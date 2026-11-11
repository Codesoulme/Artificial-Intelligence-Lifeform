"""Read-only OASST1 reviewed-prompt comparison of raw, old and new layer pairs."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoTokenizer
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_layer18_general_long_audit import repetition
from scripts.deepseek_spora_memory_answer_distill import _load_candidate
from scripts.deepseek_spora_pair_general_replay import OUTPUT as CANDIDATE, PAIR

OUTPUT = ROOT / 'runs/deepseek-spora-pair-oasst-reviewed-eval-20261112.json'
DATA = ROOT / 'runs/oasst1-chinese-candidate-v5/dialogue.jsonl'
REVIEW = ROOT / 'reports/P38-oasst1-chinese-v5-review.json'
DATA_SHA = 'cae912cd969441cd3c4ca2569967d1e95f6367d6c9f7dc4a9887aa1377de1315'
REVIEW_SHA = '1768cc466a1bc4dce2746c8988425c04e48f9033392b8c3fa4e07280be6f3fd1'
MAX_TOKENS = 48


def selected_prompts():
    if sha(DATA) != DATA_SHA or sha(REVIEW) != REVIEW_SHA:
        raise ValueError('Reviewed prompt export or review hash changed')
    review = json.loads(REVIEW.read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in DATA.read_text(encoding='utf-8').splitlines()]
    if len(rows) != len(review['row_reviews']) or review['approved_for_pilot'] is not False:
        raise ValueError('Unexpected reviewed data status')
    seen_text, seen_tree, selected = set(), set(), []
    for record in review['row_reviews']:
        if record['decision'] != 'pass':
            continue
        index = record['line'] - 1
        row = rows[index]
        if record['path_sha256'] != row['sha256'] or record['group'] != row['group']:
            raise ValueError('Review and source row mismatch')
        messages = row['messages']
        if len(messages) != 2 or messages[0]['role'] != 'user' or messages[1]['role'] != 'assistant':
            continue
        question = messages[0]['content'].strip()
        normalized = ''.join(question.split())
        if not question or len(question) > 100 or normalized in seen_text or row['group'] in seen_tree:
            continue
        seen_text.add(normalized)
        seen_tree.add(row['group'])
        selected.append({'line': record['line'], 'group': row['group'], 'path_sha256': row['sha256'],
                         'question': question})
    if not selected:
        raise ValueError('No reviewed prompts selected')
    return selected


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    items = selected_prompts()
    info = json.loads((CANDIDATE / 'result.json').read_text(encoding='utf-8'))
    for i in (6, 12):
        if (sha(PAIR / f'layer-{i}.pt') != info['source_pair_sha256'][str(i)]
                or sha(CANDIDATE / f'layer-{i}.pt') != info['candidate_sha256'][str(i)]):
            raise ValueError('Pair weight provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    accepted, too_long = [], []
    for item in items:
        prefix = closed_think_inputs(tokenizer, item['question'], 'cuda')
        if prefix['input_ids'].shape[1] + MAX_TOKENS > 128:
            too_long.append({'line': item['line'], 'input_tokens': prefix['input_ids'].shape[1]})
        else:
            accepted.append((item, prefix))
    if not accepted:
        raise ValueError('All reviewed prompts exceed position limit')
    base = load_full_streamed().eval()
    originals = {i: base.model.layers[i] for i in (6, 12)}
    rows_by_mode = {}
    for mode in ('raw', 'old_pair', 'replay_pair'):
        if mode == 'raw':
            for i in (6, 12):
                base.model.layers[i] = originals[i]
        else:
            directory = PAIR if mode == 'old_pair' else CANDIDATE
            for i in (6, 12):
                base.model.layers[i] = _load_candidate(directory / f'layer-{i}.pt', i, base).eval()
        if any(p.is_meta for p in base.parameters()) or any(b.is_meta for b in base.buffers()):
            raise ValueError('Unmaterialized layer')
        results = []
        for item, inputs in accepted:
            with torch.inference_mode():
                ids = base.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False, use_cache=False,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            tail = ids[0, inputs['input_ids'].shape[1]:].tolist()
            ended = bool(tail and tokenizer.eos_token_id is not None and tail[-1] == tokenizer.eos_token_id)
            content = tail[:-1] if ended else tail
            results.append({**item, 'answer': tokenizer.decode(content, skip_special_tokens=True).strip(),
                            'token_count': len(content), 'terminated_eos': ended,
                            'repeat_3gram_fraction': repetition(content, 3)})
            print(json.dumps({'mode': mode, 'done': len(results), 'total': len(accepted),
                              'repeat': round(results[-1]['repeat_3gram_fraction'], 3)}), flush=True)
        rows_by_mode[mode] = results
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base changed')
    summary = {}
    for mode, rows in rows_by_mode.items():
        summary[mode] = {'cases': len(rows), 'ended_with_eos': sum(r['terminated_eos'] for r in rows),
                         'any_repeated_3gram': sum(r['repeat_3gram_fraction'] > 0 for r in rows),
                         'mean_3gram_repeat_fraction': sum(r['repeat_3gram_fraction'] for r in rows) / len(rows),
                         'exact_same_as_raw': sum(r['answer'] == raw['answer'] for r, raw in zip(rows, rows_by_mode['raw']))}
    report = {'status': 'reviewed_oasst_prompt_only_development_audit', 'summary': summary, 'rows': rows_by_mode,
              'excluded_too_long': too_long, 'base_sha256': WEIGHT_SHA, 'data_sha256': DATA_SHA,
              'review_sha256': REVIEW_SHA, 'source_repository': 'OpenAssistant/oasst1',
              'source_revision': 'fdf72ae0827c1cda404aff25b6603abec9e3399b',
              'license_as_declared': 'Apache-2.0; preserve applicable notices',
              'source_pair_sha256': info['source_pair_sha256'], 'candidate_sha256': info['candidate_sha256'],
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Evaluation uses only user prompts from rows previously reviewed pass, not assistant targets.',
                              'Source export was not approved for training; this is not training.',
                              'Small convenience sample with deduplication, not a blind independent final benchmark.',
                              'Output semantic quality is not automatically scored; teacher may also be wrong.',
                              'Closed reasoning tag, greedy decoding and 48-token cap; no memory or identity path.',
                              'Research streamed loader; base parity assessed on limited samples.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
