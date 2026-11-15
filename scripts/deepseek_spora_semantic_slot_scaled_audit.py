"""Audit full generation with the direct semantic-slot capacity probe."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch

from scripts import deepseek_local_baseline as p
from scripts import deepseek_spora_ttt_crossed_values_pilot as crossed
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.research_semantic_slot import install_scaled, use_slot


OUT = ROOT / 'runs/deepseek-spora-semantic-slot-scaled-audit-v3-20261116.json'
SCALE = 32.0


def exact(answer, value):
    return answer.strip().rstrip('。.!！ ').strip() == value


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
    install_scaled(model, scale=SCALE)
    embedding = model.model.embed_tokens.weight.detach().float()

    def value_slot(value):
        token_ids = tokenizer.encode(value, add_special_tokens=False)
        return embedding[token_ids[0]].detach().float()

    def generate(question, slot=None):
        prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
        kwargs = dict(**prompt, use_cache=True, max_new_tokens=24, do_sample=False,
                      pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        with torch.inference_mode():
            if slot is None:
                output = model.generate(**kwargs)
            else:
                with use_slot(slot):
                    output = model.generate(**kwargs)
        tail = output[0, prompt['input_ids'].shape[1]:].tolist()
        return {'answer': tokenizer.decode(tail, skip_special_tokens=True).strip(),
                'token_ids': tail}

    rows = []
    values = crossed.TRAIN_VALUES + crossed.HELD_VALUES
    for name in crossed.HELD_NAMES:
        question = crossed.TEMPLATES[0].format(name=name)
        baseline = generate(question)
        for value in values:
            answer = generate(question, value_slot(value))
            rows.append({'name': name, 'value': value,
                         'group': 'seen_value' if value in crossed.TRAIN_VALUES else 'unseen_value',
                         'answer': answer, 'no_memory': baseline,
                         'exact': exact(answer['answer'], value),
                         'baseline_exact': exact(baseline['answer'], value)})
    ordinary = ('你好，请简单介绍自己。', '请写一句鼓励学习的话。')
    ordinary_before = [generate(question) for question in ordinary]
    ordinary_after = [generate(question) for question in ordinary]
    if ordinary_before != ordinary_after or p.sha(p.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Base or memory-off control changed')
    report = {'status': 'semantic_slot_scaled_capacity_audit', 'scale': SCALE,
              'summary': {group: {'cases': sum(row['group'] == group for row in rows),
                                   'exact': sum(row['group'] == group and row['exact'] for row in rows),
                                   'baseline_exact': sum(row['group'] == group and row['baseline_exact'] for row in rows)}
                          for group in ('seen_value', 'unseen_value')},
              'rows': rows, 'base_sha256': base_sha,
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['The value embedding and value boundary are supplied by the probe.',
                              'This is a capacity and language bridge audit, not a learned writer.',
                              'Only one relation and two held-out names are evaluated.',
                              'No persistence, correction, natural routing, or open-domain chat.']}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
