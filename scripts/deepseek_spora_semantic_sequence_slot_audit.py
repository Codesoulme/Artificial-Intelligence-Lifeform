"""Audit complete value generation with one semantic vector per decode step."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch

from scripts import deepseek_local_baseline as p
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.research_semantic_slot import install_sequence_scaled, use_slot_sequence


OUT = ROOT / 'runs/deepseek-spora-semantic-sequence-slot-audit-20261116.json'
SCALE = 32.0
CASES = ('酸奶', '椰汁', '柠檬水', '矿泉水')


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Unexpected base weight')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True,
                                                trust_remote_code=False)
    model = load_full_streamed().eval()
    install_sequence_scaled(model, scale=SCALE)
    embedding = model.model.embed_tokens.weight.detach().float()
    question = '小林喜欢喝什么饮料？'
    prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')

    def make_sequence(value):
        ids = tokenizer.encode(value, add_special_tokens=False)
        return ids, [embedding[token_id].detach() for token_id in ids + [tokenizer.eos_token_id]]

    rows = []
    for value in CASES:
        expected_ids, sequence = make_sequence(value)
        kwargs = dict(**prompt, use_cache=True, max_new_tokens=len(expected_ids) + 3,
                      do_sample=False, pad_token_id=tokenizer.pad_token_id,
                      eos_token_id=tokenizer.eos_token_id)
        with torch.inference_mode(), use_slot_sequence(sequence):
            output = model.generate(**kwargs)
        tail = output[0, prompt['input_ids'].shape[1]:].tolist()
        rows.append({'value': value, 'expected_ids': expected_ids, 'generated_ids': tail,
                     'expected_text': tokenizer.decode(expected_ids, skip_special_tokens=True),
                     'generated_text': tokenizer.decode(tail, skip_special_tokens=True),
                     'exact_tokens': tail == expected_ids + [tokenizer.eos_token_id],
                     'value_prefix_exact': tail[:len(expected_ids)] == expected_ids,
                     'token_count': len(expected_ids)})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    ordinary_before = [pilot.generate(model, tokenizer, text) for text in ('你好，请简单介绍自己。',)]
    ordinary_after = [pilot.generate(model, tokenizer, text) for text in ('你好，请简单介绍自己。',)]
    if ordinary_before != ordinary_after or p.sha(p.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Base or memory-off control changed')
    result = {'status': 'semantic_sequence_slot_audit', 'scale': SCALE, 'rows': rows,
              'summary': {'cases': len(rows), 'value_prefix_exact': sum(row['value_prefix_exact'] for row in rows),
                          'full_eos_exact': sum(row['exact_tokens'] for row in rows),
                          'multi_token_cases': sum(row['token_count'] > 1 for row in rows)},
              'base_sha256': base_sha, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['The token-vector sequence is supplied by an oracle writer.',
                              'This is a capacity probe, not a learned sequence writer or natural router.',
                              'Only one relation and four controlled values are evaluated.']}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
