"""Probe memory-dependent answers in short and sentence-style requests."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from spora.memory_output import MemoryOutputAdapter

SOURCE = ROOT / 'runs/deepseek-spora-memory-output-adapter-v2-20261111'
OUTPUT = ROOT / 'runs/deepseek-spora-output-sentence-counterfactual-20261114.json'
CASES = (
    (6, 'short', '你记得{name}偏爱喝什么饮品吗？直接说名称。', (2, 5)),
    (6, 'sentence', '请用一句完整的话说明你记得的{name}的饮料偏好。', (2, 5)),
    (7, 'short', '依照已有记录，{name}通常喜欢喝什么？只说名称。', (3, 7)),
    (7, 'sentence', '用一句话告诉我，记录中{name}的饮品喜好是什么。', (3, 7)),
)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    candidate_path = SOURCE / 'adapter.pt'
    candidate_sha = p.sha(candidate_path)
    if base_sha != p.WEIGHT_SHA or candidate_sha != json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))['adapter_sha256']:
        raise ValueError('Base or candidate checksum mismatch')
    payload = torch.load(candidate_path, map_location='cpu', weights_only=True)
    if payload['base_sha256'] != base_sha:
        raise ValueError('Candidate base mismatch')
    for layer in (6, 12):
        path = ROOT / 'runs/deepseek-spora-diverse-memory-replay-20261111' / f'layer-{layer}.pt'
        if payload['candidate_sha256'][str(layer)] != p.sha(path):
            raise ValueError('Candidate layer mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix='spora-output-sentence-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        original_digest = p.frozen_digest(chat.organism, chat.state)
        old = chat.decoder.output_adapter
        new = MemoryOutputAdapter(**payload['config']).to('cuda').eval()
        new.load_state_dict(payload['state'], strict=True)
        for index, style, template, targets in CASES:
            question = template.format(name=NAMES[index])
            prompt = closed_think_inputs(chat.tokenizer, question, 'cuda')
            state = p.fresh_controlled_state(chat.organism)
            for target in targets:
                state, decision = p.observe_sourced(chat.organism, state, KEYS[index], target, 'direct', 1.0)
                if not decision['current_write']:
                    raise ValueError('Controlled observation rejected')
                digest = p.frozen_digest(chat.organism, state)
                args, evidence = p.runtime._prepare_memory_read(chat.organism, state, KEYS[index], 'current', None)
                native = int(evidence.argmax(-1)[0])
                results = {}
                for label in ('no_memory', 'reader_only', 'current', 'v2'):
                    adapter = {'current': old, 'v2': new}.get(label)
                    if adapter is not None:
                        chat.decoder.output_adapter = adapter
                    memory_args = {} if label == 'no_memory' else args
                    with torch.inference_mode():
                        ids = chat.decoder.generate(**prompt, **memory_args,
                                                    memory_output_logits=evidence if adapter is not None else None,
                                                    max_new_tokens=40, do_sample=False,
                                                    pad_token_id=chat.tokenizer.pad_token_id,
                                                    eos_token_id=chat.tokenizer.eos_token_id)
                    tail = ids[0, prompt['input_ids'].shape[1]:].tolist()
                    answer = chat.tokenizer.decode(tail, skip_special_tokens=True).strip()
                    results[label] = {'answer': answer, 'token_ids': tail,
                                      'ended': chat.tokenizer.eos_token_id in tail,
                                      'exact': answer.rstrip('。.!！ ') == p.WORDS[target],
                                      'contains_expected': p.WORDS[target] in answer,
                                      'other_values': [w for w in p.WORDS if w != p.WORDS[target] and w in answer]}
                if p.frozen_digest(chat.organism, state) != digest:
                    raise ValueError('Temporary state changed during generation')
                row = {'question': question, 'style': style, 'key': KEYS[index], 'target': p.WORDS[target],
                       'target_id': target, 'native_id': native, 'results': results}
                rows.append(row)
                print(json.dumps({'done': len(rows), 'native_correct': native == target,
                                  'answers': {k: v['answer'] for k, v in results.items()}}, ensure_ascii=True), flush=True)
        chat.decoder.output_adapter = old
        if p.frozen_digest(chat.organism, chat.state) != original_digest:
            raise ValueError('Original state changed')
    if p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(p.runtime.CHECKPOINT) != checkpoint_sha or p.sha(candidate_path) != candidate_sha:
        raise ValueError('Protected file changed')
    summary = {'cases': len(rows), 'native_correct': sum(r['target_id'] == r['native_id'] for r in rows)}
    for label in ('no_memory', 'reader_only', 'current', 'v2'):
        summary[label] = {style: {'cases': len(subset),
                                  'exact': sum(r['results'][label]['exact'] for r in subset),
                                  'contains_expected': sum(r['results'][label]['contains_expected'] for r in subset),
                                  'other_value_mentions': sum(bool(r['results'][label]['other_values']) for r in subset),
                                  'ended': sum(r['results'][label]['ended'] for r in subset)}
                          for style in ('short', 'sentence')
                          for subset in [[r for r in rows if r['style'] == style]]}
    result = {'summary': summary, 'rows': rows, 'base_sha256': base_sha, 'checkpoint_sha256': checkpoint_sha,
              'candidate_sha256': candidate_sha, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Small authored development cases; not independent population estimates.',
                              'Names 6 and 7 were outside v2 output-head training but were used in previous evaluations.',
                              'Manual numeric addresses and writes bypass natural language routing.',
                              'Value containment and keyword checks are incomplete semantic metrics.',
                              'No-memory baseline still uses the replaced model layers but without read arguments.']}
    with OUTPUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
