"""Measure bounded native-thinking completion without recording reasoning text."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoTokenizer
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_memory_answer_distill import _load_candidate
from scripts.deepseek_spora_pair_oasst_reviewed_eval import DATA_SHA, REVIEW_SHA, selected_prompts

OUTPUT = ROOT / 'runs/deepseek-spora-native-think-budget-audit-20261112.json'
PARENT = ROOT / 'runs/deepseek-spora-pair-general-replay-20261112'
CANDIDATE = ROOT / 'runs/deepseek-spora-pair-constraints-pilot-20261112'
OLD_PAIR = ROOT / 'runs/deepseek-spora-memory-answer-distill-20261111'
LINES = (2, 4, 19, 26, 28, 32, 61, 70)
MAX_NEW = 80


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    by_line = {r['line']: r for r in selected_prompts()}
    if any(line not in by_line for line in LINES):
        raise ValueError('Selected reviewed prompt unavailable')
    prompts = [by_line[line] for line in LINES]
    parent_report = json.loads((PARENT / 'result.json').read_text(encoding='utf-8'))
    candidate_report = json.loads((CANDIDATE / 'result.json').read_text(encoding='utf-8'))
    for i in (6, 12):
        if (sha(OLD_PAIR / f'layer-{i}.pt') != parent_report['source_pair_sha256'][str(i)]
                or sha(CANDIDATE / f'layer-{i}.pt') != candidate_report['candidate_sha256'][str(i)]):
            raise ValueError('Replacement digest mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    close_ids = tokenizer('</think>', add_special_tokens=False)['input_ids']
    if len(close_ids) != 1:
        raise ValueError('Unexpected thinking delimiter tokenization')
    close_id = close_ids[0]
    encoded = []
    for item in prompts:
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': item['question']}], tokenize=False,
                                             add_generation_prompt=True)
        if '<think>' not in text or '</think>' in text:
            raise ValueError('Unexpected native thinking prompt')
        inp = tokenizer(text, return_tensors='pt', add_special_tokens=False).to('cuda')
        if inp['input_ids'].shape[1] + MAX_NEW > 128:
            raise ValueError('Prompt exceeds replacement position range')
        encoded.append(inp)
    base = load_full_streamed().eval()
    originals = {i: base.model.layers[i] for i in (6, 12)}
    outputs = {}
    for mode, directory in (('raw', None), ('old_pair', OLD_PAIR), ('candidate', CANDIDATE)):
        for i in (6, 12):
            base.model.layers[i] = originals[i] if directory is None else _load_candidate(directory / f'layer-{i}.pt', i, base).eval()
        if any(p.is_meta for p in base.parameters()) or any(b.is_meta for b in base.buffers()):
            raise ValueError('Unmaterialized layer')
        rows = []
        for item, inputs in zip(prompts, encoded):
            with torch.inference_mode():
                ids = base.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False, use_cache=False,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            tail = ids[0, inputs['input_ids'].shape[1]:].tolist()
            ended = bool(tail and tokenizer.eos_token_id is not None and tail[-1] == tokenizer.eos_token_id)
            close_index = tail.index(close_id) if close_id in tail else None
            final_ids = [] if close_index is None else tail[close_index + 1:]
            if final_ids and ended:
                final_ids = final_ids[:-1]
            final = tokenizer.decode(final_ids, skip_special_tokens=True).strip() if final_ids else ''
            row = {'line': item['line'], 'group': item['group'], 'question': item['question'],
                   'input_tokens': inputs['input_ids'].shape[1], 'generated_tokens': len(tail),
                   'thinking_closed': close_index is not None,
                   'tokens_before_close': close_index if close_index is not None else None,
                   'final_answer': final, 'has_final_answer': bool(final), 'ended_with_eos': ended}
            rows.append(row)
            print(json.dumps({'mode': mode, 'line': item['line'], 'closed': row['thinking_closed'],
                              'has_final': row['has_final_answer'], 'generated': len(tail)}), flush=True)
        outputs[mode] = rows
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected weights changed')
    summary = {mode: {'cases': len(rows), 'thinking_closed': sum(r['thinking_closed'] for r in rows),
                      'has_final_answer': sum(r['has_final_answer'] for r in rows),
                      'ended_with_eos': sum(r['ended_with_eos'] for r in rows)} for mode, rows in outputs.items()}
    report = {'status': 'native_think_budget_development_audit', 'summary': summary, 'rows': outputs,
              'base_sha256': WEIGHT_SHA, 'old_pair_sha256': parent_report['source_pair_sha256'],
              'candidate_sha256': candidate_report['candidate_sha256'], 'data_sha256': DATA_SHA,
              'review_sha256': REVIEW_SHA, 'max_new_tokens': MAX_NEW, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Eight selected previously reviewed OASST prompts, not an independent benchmark.',
                              'Generated reasoning text is intentionally not persisted; final text after delimiter only.',
                              'No final answer within budget does not prove the model could not answer with more context.',
                              'No internal persistent thinking, background processing, memory reads or subjective experience demonstrated.',
                              'Greedy decoding, no cache, 80 new token cap; replacement position table limited to 128.',
                              'Original base and both pairs run with the same research loader and prompts.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
