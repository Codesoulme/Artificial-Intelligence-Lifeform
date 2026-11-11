"""Compare ungrafted DeepSeek against a previous identical-question audit."""
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
from scripts.deepseek_spora_layer18_general_long_audit import OUTPUT as PRIOR, QUESTIONS, MAX_TOKENS, repetition

OUTPUT = ROOT / 'runs/deepseek-spora-general-raw-baseline-20261112.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    if (prior['base_sha256'] != WEIGHT_SHA or prior['max_new_tokens'] != MAX_TOKENS
            or [r['question'] for r in prior['rows']['pair']] != list(QUESTIONS)):
        raise ValueError('Prior comparison is not aligned')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed().eval()
    if any(p.is_meta for p in base.parameters()) or any(b.is_meta for b in base.buffers()):
        raise ValueError('Unmaterialized raw base')
    rows = []
    for q, pair in zip(QUESTIONS, prior['rows']['pair']):
        inputs = closed_think_inputs(tokenizer, q, 'cuda')
        with torch.inference_mode():
            ids = base.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False, use_cache=False,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        tail = ids[0, inputs['input_ids'].shape[1]:].tolist()
        terminated = bool(tail and tokenizer.eos_token_id is not None and tail[-1] == tokenizer.eos_token_id)
        content = tail[:-1] if terminated else tail
        row = {'question': q, 'answer': tokenizer.decode(content, skip_special_tokens=True).strip(),
               'token_count': len(content), 'terminated_eos': terminated,
               'repeat_2gram_fraction': repetition(content, 2), 'repeat_3gram_fraction': repetition(content, 3),
               'same_as_pair': tokenizer.decode(content, skip_special_tokens=True).strip() == pair['answer']}
        rows.append(row)
        print(json.dumps({'index': len(rows), 'tokens': len(content), 'eos': terminated,
                          'repeat_3gram': round(row['repeat_3gram_fraction'], 3),
                          'same_as_pair': row['same_as_pair']}), flush=True)
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected weights changed')
    summary = {'cases': len(rows), 'ended_with_eos': sum(r['terminated_eos'] for r in rows),
               'any_repeated_3gram': sum(r['repeat_3gram_fraction'] > 0 for r in rows),
               'mean_3gram_repeat_fraction': sum(r['repeat_3gram_fraction'] for r in rows) / len(rows),
               'exact_same_as_pair': sum(r['same_as_pair'] for r in rows)}
    report = {'status': 'raw_deepseek_general_development_baseline', 'summary': summary, 'rows': rows,
              'prior_summary': prior['summary'], 'base_sha256': WEIGHT_SHA, 'prior_sha256': sha(PRIOR),
              'max_new_tokens': MAX_TOKENS, 'seconds': time.perf_counter() - start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Twelve authored questions, previously examined in development.',
                              'Runs separately under the same settings, not a simultaneous process.',
                              'Reasoning tag is closed; result does not characterize open-thinking behavior.',
                              'Raw base uses streamed local weights; loader parity verified on limited prior samples.',
                              'No memory reader and no identity state; answer quality requires content review.',
                              'N-gram and EOS statistics are not semantic accuracy or full-chat measures.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
