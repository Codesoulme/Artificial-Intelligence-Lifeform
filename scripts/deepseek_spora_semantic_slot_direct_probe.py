"""Capacity probe for direct semantic-slot injection into the frozen head."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch

from scripts import deepseek_local_baseline as p
from scripts import deepseek_spora_ttt_crossed_values_pilot as crossed
from scripts.deepseek_spora_full_stream_loader import load_full_streamed


OUT = ROOT / 'runs/deepseek-spora-semantic-slot-direct-probe-20261116.json'
SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)


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
    embedding = model.model.embed_tokens.weight.detach().float()
    rows = []
    values = crossed.TRAIN_VALUES + crossed.HELD_VALUES
    for name in crossed.HELD_NAMES:
        question = crossed.TEMPLATES[0].format(name=name)
        inputs = crossed.pilot.closed_think_inputs(tokenizer, question, 'cuda')
        with torch.inference_mode():
            hidden = model.model(**inputs, use_cache=False, return_dict=True).last_hidden_state[:, -1].float()
        head_dtype = model.lm_head.weight.dtype
        native = model.lm_head(hidden.to(head_dtype))[0].float()
        for value in values:
            token_id = tokenizer.encode(value, add_special_tokens=False)[0]
            semantic = embedding[token_id]
            for scale in SCALES:
                logits = model.lm_head((hidden + scale * semantic).to(head_dtype))[0].float()
                rank = int((logits > logits[token_id]).sum()) + 1
                rows.append({'name': name, 'value': value, 'scale': scale,
                             'group': 'seen_value' if value in crossed.TRAIN_VALUES else 'unseen_value',
                             'native_rank': int((native > native[token_id]).sum()) + 1,
                             'injected_rank': rank,
                             'winner': tokenizer.decode([int(logits.argmax())], skip_special_tokens=False)})
        print(json.dumps({'name': name, 'cases': len(values)}, ensure_ascii=False), flush=True)
    if p.sha(p.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Base model changed')
    summary = {}
    for scale in SCALES:
        subset = [row for row in rows if row['scale'] == scale]
        summary[str(scale)] = {group: {
            'cases': sum(row['group'] == group for row in subset),
            'native_top1': sum(row['group'] == group and row['native_rank'] == 1 for row in subset),
            'injected_top1': sum(row['group'] == group and row['injected_rank'] == 1 for row in subset),
            'injected_top10': sum(row['group'] == group and row['injected_rank'] <= 10 for row in subset),
        } for group in ('seen_value', 'unseen_value')}
    result = {'status': 'semantic_slot_direct_capacity_probe', 'scales': list(SCALES),
              'summary': summary, 'rows': rows, 'base_sha256': base_sha,
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Direct semantic vector injection is an oracle, not a trained writer.',
                              'Only first-token ranking is measured; no full generation or persistence.',
                              'The fact value and value token are supplied by the probe.']}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
