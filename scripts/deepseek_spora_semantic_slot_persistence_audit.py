"""Persist oracle semantic slots and verify an independent reload."""
import json
import sys
import tempfile
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
from spora.semantic_slot_state import SemanticSlotState, load, save


SOURCE = candidate.OUT
OUT = ROOT / 'runs/deepseek-spora-semantic-slot-persistence-audit-20261116.json'


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
    embedding = model.model.embed_tokens.weight.detach().float().cpu()
    values = crossed.TRAIN_VALUES + crossed.HELD_VALUES
    state = SemanticSlotState(embedding.shape[-1], len(values))
    for index, value in enumerate(values):
        token_ids = tokenizer.encode(value, add_special_tokens=False)
        state = state.put(index, embedding[token_ids[0]])

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / 'semantic-slots.pt'
        state_digest = save(checkpoint, state, base_sha256=base_sha,
                            bridge_sha256=bridge_sha, metadata={'purpose': 'reload-audit'})
        restored, restored_metadata = load(checkpoint, expected_base_sha256=base_sha,
                                           expected_bridge_sha256=bridge_sha, device='cuda')
        if restored.revision != state.revision or not torch.equal(restored.occupied.cpu(), state.occupied.cpu()):
            raise ValueError('Restored semantic slot metadata changed')

        rows = []
        for name in crossed.HELD_NAMES:
            prompt = pilot.closed_think_inputs(tokenizer, crossed.TEMPLATES[0].format(name=name), 'cuda')
            for index, value in enumerate(values):
                token_id = tokenizer.encode(value, add_special_tokens=False)[0]
                with torch.inference_mode(), use_slot(restored.get(index, device='cuda')):
                    logits = model(**prompt, use_cache=False, return_dict=True).logits[0, -1].float()
                rows.append({'name': name, 'value': value,
                             'group': 'seen_value' if value in crossed.TRAIN_VALUES else 'unseen_value',
                             'rank': int((logits > logits[token_id]).sum()) + 1,
                             'winner': tokenizer.decode([int(logits.argmax())], skip_special_tokens=False)})
    if p.sha(p.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Base model changed')
    report = {'status': 'semantic_slot_persistence_audit',
              'summary': {'cases': len(rows), 'top1': sum(row['rank'] == 1 for row in rows),
                          'seen_top1': sum(row['group'] == 'seen_value' and row['rank'] == 1 for row in rows),
                          'unseen_top1': sum(row['group'] == 'unseen_value' and row['rank'] == 1 for row in rows)},
              'rows': rows, 'state_digest': state_digest, 'restored_metadata': restored_metadata,
              'base_sha256': base_sha, 'bridge_sha256': bridge_sha,
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['The writer still supplies value token embeddings directly.',
                              'The slot bank is a state-file prototype, not yet integrated into live observation routing.',
                              'Only first-token decoding and one relation are evaluated.']}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
