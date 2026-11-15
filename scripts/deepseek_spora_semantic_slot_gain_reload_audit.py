"""Reload the gain bridge independently and recheck held values."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch

from scripts import deepseek_local_baseline as p
from scripts import deepseek_spora_semantic_slot_gain_pilot as candidate
from scripts import deepseek_spora_ttt_crossed_values_pilot as crossed
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.research_semantic_slot import install_gain, use_slot


SOURCE = candidate.OUT
OUT = ROOT / 'runs/deepseek-spora-semantic-slot-gain-reload-20261116.json'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    metadata = json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))
    bridge_path = SOURCE / 'bridge.pt'
    bridge_sha = p.sha(bridge_path)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    payload = torch.load(bridge_path, map_location='cpu', weights_only=True)
    if (metadata['bridge_sha256'] != bridge_sha or base_sha != p.WEIGHT_SHA
            or payload['base_sha256'] != base_sha):
        raise ValueError('Bridge or base provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True,
                                                trust_remote_code=False)
    model = load_full_streamed().eval()
    bridge = install_gain(model, rank=payload['rank'], initial_gain=32.)
    bridge.load_state_dict(payload['bridge'], strict=True)
    model.requires_grad_(False)
    embedding = model.model.embed_tokens.weight.detach().float()

    def first_token(name, value):
        prompt = pilot.closed_think_inputs(tokenizer, crossed.TEMPLATES[0].format(name=name), 'cuda')
        token_id = tokenizer.encode(value, add_special_tokens=False)[0]
        slot = embedding[token_id].detach()
        with torch.inference_mode(), use_slot(slot):
            logits = model(**prompt, use_cache=False, return_dict=True).logits[0, -1].float()
        return {'name': name, 'value': value,
                'rank': int((logits > logits[token_id]).sum()) + 1,
                'winner': tokenizer.decode([int(logits.argmax())], skip_special_tokens=False)}

    rows = [first_token(name, value)
            for name in crossed.HELD_NAMES
            for value in crossed.TRAIN_VALUES + crossed.HELD_VALUES]
    ordinary = ('你好，请简单介绍自己。', '请写一句鼓励学习的话。')
    ordinary_before = [pilot.generate(model, tokenizer, question) for question in ordinary]
    ordinary_after = [pilot.generate(model, tokenizer, question) for question in ordinary]
    if ordinary_before != ordinary_after or p.sha(p.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Base or memory-off control changed')
    report = {'status': 'semantic_slot_gain_reload_audit',
              'summary': {'cases': len(rows), 'top1': sum(row['rank'] == 1 for row in rows),
                          'seen_top1': sum(row['value'] in crossed.TRAIN_VALUES and row['rank'] == 1 for row in rows),
                          'unseen_top1': sum(row['value'] in crossed.HELD_VALUES and row['rank'] == 1 for row in rows)},
              'rows': rows, 'base_sha256': base_sha, 'bridge_sha256': bridge_sha,
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['The semantic slot is supplied by the controlled writer.',
                              'This checks first-token reload, not full dialogue or persistence of arbitrary observations.',
                              'No natural routing, correction, unknown-fact refusal, or multi-fact test.']}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
