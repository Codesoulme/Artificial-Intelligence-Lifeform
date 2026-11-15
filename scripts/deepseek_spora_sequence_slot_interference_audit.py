"""Measure whether a sequence-slot probe overrides unrelated prompts.

This experiment uses oracle-selected token vectors and is not an evaluation of
natural routing or evidence that facts are learned in the base parameters.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch

from scripts import deepseek_local_baseline as baseline
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.research_semantic_slot import install_sequence_scaled, use_slot_sequence


OUT = ROOT / 'runs/deepseek-spora-sequence-slot-interference-20261116.json'
QUESTIONS = (
    ('memory', '小赵喜欢喝什么饮料？'),
    ('arithmetic', '只回答结果：二加二等于多少？'),
    ('greeting', '请只说你好。'),
)
VALUES = ('酸奶', '椰汁')


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    base_sha = baseline.sha(baseline.MODEL / 'model.safetensors')
    if base_sha != baseline.WEIGHT_SHA:
        raise ValueError('Unexpected base weight')
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = baseline.AutoTokenizer.from_pretrained(
        baseline.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    model.requires_grad_(False)
    install_sequence_scaled(model, scale=32.)
    embedding = model.model.embed_tokens.weight.detach()
    rows = []
    for group, question in QUESTIONS:
        prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
        options = dict(**prompt, use_cache=True, max_new_tokens=24, do_sample=False,
                       pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        for value in (None, *VALUES):
            ids = None if value is None else tokenizer.encode(value, add_special_tokens=False)
            with torch.inference_mode():
                if ids is None:
                    output = model.generate(**options)
                else:
                    sequence = embedding[torch.tensor(ids + [tokenizer.eos_token_id], device=embedding.device)].float()
                    with use_slot_sequence(sequence):
                        output = model.generate(**options)
            tail = output[0, prompt['input_ids'].shape[1]:].tolist()
            rows.append({'group': group, 'question': question, 'injected_value': value,
                         'expected_injected_token_ids': ids,
                         'generated_token_ids': tail,
                         'generated_text': tokenizer.decode(tail, skip_special_tokens=True).strip(),
                         'exact_injection': ids is not None and tail == ids + [tokenizer.eos_token_id]})
            print(json.dumps({'group': group, 'injected_value': value,
                              'exact_injection': rows[-1]['exact_injection']}, ensure_ascii=True), flush=True)
    if baseline.sha(baseline.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Base file changed')
    summary = {}
    for group, _ in QUESTIONS:
        group_rows = [r for r in rows if r['group'] == group]
        off = next(r for r in group_rows if r['injected_value'] is None)
        on = [r for r in group_rows if r['injected_value'] is not None]
        summary[group] = {'injections': len(on),
                          'exact_injected_output': sum(r['exact_injection'] for r in on),
                          'changed_from_off': sum(r['generated_token_ids'] != off['generated_token_ids'] for r in on)}
    result = {'status': 'sequence_slot_forced_interference_audit', 'summary': summary,
              'rows': rows, 'base_sha256': base_sha,
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Oracle token embeddings are injected on every generation step.',
                              'Three authored prompts and two values are a development probe.',
                              'This intentionally forces memory on unrelated queries; it does not measure routing accuracy.',
                              'No natural observations, persistent individual, or independent chat benchmark.']}
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
