"""Diagnose full-vocabulary first-token scores under counterfactual memories."""
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_spora_original_reader_fresh_audit import audit

SOURCE = ROOT / 'runs/deepseek-spora-original-reader-mixed-256-20261114'
PREVIOUS = ROOT / 'runs/deepseek-spora-original-reader-fresh-256-20261114.json'
OUTPUT = ROOT / 'runs/deepseek-spora-original-first-token-rank-20261114.json'


def describe(scores, tokenizer, target_id, other_id):
    if scores.ndim != 1 or not bool(torch.isfinite(scores).all()):
        raise ValueError('Expected finite vocabulary scores')
    if not 0 <= target_id < len(scores) or not 0 <= other_id < len(scores):
        raise ValueError('Target token outside vocabulary')
    top = scores.topk(5)
    return {'target_id': target_id, 'target_score': float(scores[target_id]),
            'target_rank': int((scores > scores[target_id]).sum()) + 1,
            'other_id': other_id, 'other_score': float(scores[other_id]),
            'margin_vs_other': float(scores[target_id] - scores[other_id]),
            'margin_vs_best': float(scores[target_id] - top.values[0]),
            'top': [{'id': int(i), 'text': tokenizer.decode([int(i)]), 'score': float(v)}
                    for i, v in zip(top.indices.tolist(), top.values.tolist())]}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    candidate_sha = p.sha(SOURCE / 'reader.pt')
    previous_sha = p.sha(PREVIOUS)
    info = json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))
    previous = json.loads(PREVIOUS.read_text(encoding='utf-8'))
    candidate = torch.load(SOURCE / 'reader.pt', map_location='cpu', weights_only=True)
    if (base_sha != p.WEIGHT_SHA or candidate_sha != info['candidate_sha256']
            or candidate_sha != previous['candidate_sha256'] or candidate['base_sha256'] != base_sha
            or candidate['source_checkpoint_sha256'] != checkpoint_sha or not candidate['original_layers']
            or previous['base_sha256'] != base_sha or previous['checkpoint_sha256'] != checkpoint_sha):
        raise ValueError('Research provenance mismatch')
    saved_rows = {(r['question'], r['native_value_id']): r for r in previous['rows']}
    if len(saved_rows) != len(previous['rows']):
        raise ValueError('Duplicate recorded cases')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix='spora-original-first-rank-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        base = p.load_full_streamed().eval()
        originals = {i: base.model.layers[i] for i in (6, 12)}
        chat = p.runtime.load_chat(base_loader=lambda: base)
        source_digest = p.frozen_digest(chat.organism, chat.state)
        replacements = {i: base.model.layers[i] for i in originals}
        for i in originals:
            base.model.layers[i] = originals[i]
        baseline = {k: v.detach().cpu().clone() for k, v in chat.decoder.reader.state_dict().items()}
        heads = {'baseline': baseline, 'candidate': candidate['state']}
        for name, key, question, first, second in audit.CASES:
            prefix = closed_think_inputs(chat.tokenizer, question, 'cuda')
            ids_by_value = {value: chat.tokenizer.encode(p.WORDS[value], add_special_tokens=False)
                            for value in (first, second)}
            if (any(not ids for ids in ids_by_value.values())
                    or ids_by_value[first][0] == ids_by_value[second][0]):
                raise ValueError('Target first tokens must be distinct')
            with torch.inference_mode():
                no_memory = chat.decoder(**prefix, use_cache=False).logits[0, -1].float()
            states = {}
            for value in (first, second):
                state = p.fresh_controlled_state(chat.organism)
                state, decision = p.observe_sourced(chat.organism, state, key, value, 'direct', 1.0)
                if not decision['current_write']:
                    raise ValueError('Controlled write rejected')
                digest = p.frozen_digest(chat.organism, state)
                args, evidence = p.runtime._prepare_memory_read(chat.organism, state, key, 'current', None)
                if int(evidence.argmax(-1)[0]) != value:
                    raise ValueError('Native evidence differs from controlled write')
                other = second if value == first else first
                row = {'name': name, 'question': question, 'key': key, 'value': p.WORDS[value],
                       'value_id': value, 'target_tokens': ids_by_value[value], 'other_value': p.WORDS[other],
                       'modes': {'no_memory': describe(no_memory, chat.tokenizer, ids_by_value[value][0],
                                                       ids_by_value[other][0])}}
                old_row = saved_rows[(question, value)]
                for label, weight in heads.items():
                    chat.decoder.reader.load_state_dict(weight, strict=True)
                    with patch.object(chat.decoder.reader, 'forward', wraps=chat.decoder.reader.forward) as reader:
                        with torch.inference_mode():
                            scores = chat.decoder(**prefix, **args).logits[0, -1].float()
                    if {c.args[2] for c in reader.call_args_list} != {0, 1}:
                        raise ValueError('Both reader sites must run')
                    result = describe(scores, chat.tokenizer, ids_by_value[value][0], ids_by_value[other][0])
                    generated = old_row['answers'][label]
                    result['generated_first_id'] = generated['token_ids'][0] if generated['token_ids'] else None
                    result['generated_answer'] = generated['answer']
                    result['forward_winner_matches_generation'] = result['top'][0]['id'] == result['generated_first_id']
                    row['modes'][label] = result
                if p.frozen_digest(chat.organism, state) != digest:
                    raise ValueError('Temporary state changed')
                rows.append(row)
                states[value] = row
                print(json.dumps({'name': name, 'value': row['value'],
                                  'ranks': {k: v['target_rank'] for k, v in row['modes'].items()}}, ensure_ascii=True), flush=True)
            for label in ('baseline', 'candidate'):
                # The second row stores the reverse token order, so addition
                # gives margin(first state) - margin(second state).
                margin_difference = states[first]['modes'][label]['margin_vs_other'] + states[second]['modes'][label]['margin_vs_other']
                delta = 1.0 - margin_difference
                pair_loss = max(delta, 0.0) + math.log1p(math.exp(-abs(delta)))
                for value in (first, second):
                    states[value]['modes'][label]['paired_margin_difference'] = margin_difference
                    states[value]['modes'][label]['paired_loss'] = pair_loss
        chat.decoder.reader.load_state_dict(baseline, strict=True)
        for i in originals:
            base.model.layers[i] = replacements[i]
        if p.frozen_digest(chat.organism, chat.state) != source_digest:
            raise ValueError('Source individual changed')
    for path, sha in ((p.MODEL / 'model.safetensors', base_sha), (p.runtime.CHECKPOINT, checkpoint_sha),
                      (SOURCE / 'reader.pt', candidate_sha), (PREVIOUS, previous_sha)):
        if p.sha(path) != sha:
            raise ValueError('Protected input changed')
    summary = {}
    for label in ('no_memory', 'baseline', 'candidate'):
        subset = [r['modes'][label] for r in rows]
        summary[label] = {'cases': len(subset), 'target_rank_1': sum(v['target_rank'] == 1 for v in subset),
                          'target_rank_over_10': sum(v['target_rank'] > 10 for v in subset)}
        if label != 'no_memory':
            summary[label]['winner_matches_recorded_first'] = sum(v['forward_winner_matches_generation'] for v in subset)
    report = {'summary': summary, 'rows': rows, 'base_sha256': base_sha, 'checkpoint_sha256': checkpoint_sha,
              'candidate_sha256': candidate_sha, 'previous_sha256': previous_sha,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Eight inspected development cases and finite known values.',
                              'Addresses and facts are manually specified, not natural routing or learning.',
                              'First word token rank is not full-answer quality or semantics.',
                              'The paired diagnostic compares these held pairs, not the training pairs.',
                              'No output adapter; forward scores compared with separately recorded generation.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
