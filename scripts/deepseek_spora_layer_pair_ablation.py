"""Compare original, each single graft and both grafts on shared prompts."""
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
from scripts.deepseek_spora_layer18_general_long_audit import MAX_TOKENS, PAIR, QUESTIONS, repetition
from scripts.deepseek_spora_memory_answer_distill import _load_candidate

OUTPUT = ROOT / 'runs/deepseek-spora-layer-pair-ablation-20261112.json'
REFERENCE = ROOT / 'runs/deepseek-spora-layer18-broad-pilot-20261112/result.json'
MODES = {'raw': (), 'layer6': (6,), 'layer12': (12,), 'pair': (6, 12)}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    reference = json.loads(REFERENCE.read_text(encoding='utf-8'))
    for i in (6, 12):
        if sha(PAIR / f'layer-{i}.pt') != reference['upstream_pair_sha256'][str(i)]:
            raise ValueError('Replacement provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed().eval()
    originals = {i: base.model.layers[i] for i in (6, 12)}
    replacements = {i: _load_candidate(PAIR / f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    inputs = [closed_think_inputs(tokenizer, question, 'cuda') for question in QUESTIONS]
    if any(x['input_ids'].shape[1] + MAX_TOKENS > 128 for x in inputs):
        raise ValueError('Position limit exceeded')
    all_rows = {}
    for mode, replaced in MODES.items():
        for i in (6, 12):
            base.model.layers[i] = replacements[i] if i in replaced else originals[i]
        if any(p.is_meta for p in base.parameters()) or any(b.is_meta for b in base.buffers()):
            raise ValueError('Unmaterialized model')
        rows = []
        for q, encoded in zip(QUESTIONS, inputs):
            with torch.inference_mode():
                ids = base.generate(**encoded, max_new_tokens=MAX_TOKENS, do_sample=False, use_cache=False,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            tail = ids[0, encoded['input_ids'].shape[1]:].tolist()
            ended = bool(tail and tokenizer.eos_token_id is not None and tail[-1] == tokenizer.eos_token_id)
            content = tail[:-1] if ended else tail
            row = {'question': q, 'answer': tokenizer.decode(content, skip_special_tokens=True).strip(),
                   'token_count': len(content), 'terminated_eos': ended,
                   'repeat_3gram_fraction': repetition(content, 3)}
            rows.append(row)
            print(json.dumps({'mode': mode, 'index': len(rows), 'tokens': len(content),
                              'repeat_3gram': round(row['repeat_3gram_fraction'], 3)}), flush=True)
        all_rows[mode] = rows
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base changed')
    summary = {}
    for mode, rows in all_rows.items():
        summary[mode] = {'cases': len(rows), 'ended_with_eos': sum(r['terminated_eos'] for r in rows),
                         'any_repeated_3gram': sum(r['repeat_3gram_fraction'] > 0 for r in rows),
                         'mean_3gram_repeat_fraction': sum(r['repeat_3gram_fraction'] for r in rows) / len(rows),
                         'exact_same_as_raw': sum(r['answer'] == b['answer'] for r, b in zip(rows, all_rows['raw']))}
    report = {'status': 'layer_pair_single_ablation_general_development', 'summary': summary, 'rows': all_rows,
              'base_sha256': WEIGHT_SHA, 'pair_sha256': reference['upstream_pair_sha256'],
              'reference_sha256': sha(REFERENCE), 'max_new_tokens': MAX_TOKENS,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Twelve authored questions repeatedly inspected during development.',
                              'Changes in text or repetition do not directly measure correctness.',
                              'Greedy generation with reasoning tag closed, maximum 48 new tokens.',
                              'No memory readers, individual state, or automatic factual correctness assessment.',
                              'Single-block effects are conditional on this fixed pair and are not additive.',
                              'Full research streamed base loading; parity only checked on limited samples.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
