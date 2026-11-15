"""Audit multiple relation slots, relation-local correction, and reload."""
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
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.research_semantic_slot import install_gain, use_slot
from spora.semantic_slot_runtime import NaturalSemanticSlotMemory
from spora.semantic_slot_state import SemanticSlotState


SOURCE = candidate.OUT
OUT = ROOT / 'runs/deepseek-spora-natural-multirelation-audit-20261116.json'
FACTS = ('小林喜欢喝酸奶。', '小林喜欢的颜色是蓝色。', '小梅喜欢喝红茶。')


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
    embedding = model.model.embed_tokens.weight.detach()

    def evaluate(memory, cases, phase):
        rows = []
        for question, expected in cases:
            try:
                route = memory.query_slot(question)
            except ValueError as error:
                rows.append({'phase': phase, 'question': question, 'expected': expected,
                             'status': 'rejected', 'reason': str(error)})
                continue
            row = {'phase': phase, 'question': question, 'expected': expected,
                   'route': {'subject': route['subject'], 'relation': route['relation'],
                             'slot': route['slot']}}
            if route['slot'] is None:
                row.update({'status': 'no_route', 'top1': False})
                rows.append(row)
                continue
            token_id = tokenizer.encode(expected, add_special_tokens=False)[0]
            prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
            with torch.inference_mode(), use_slot(route['value']):
                logits = model(**prompt, use_cache=False, return_dict=True).logits[0, -1].float()
            row.update({'status': 'read', 'rank': int((logits > logits[token_id]).sum()) + 1,
                        'winner': tokenizer.decode([int(logits.argmax())], skip_special_tokens=False)})
            row['top1'] = row['rank'] == 1
            rows.append(row)
        return rows

    state = SemanticSlotState(embedding.shape[-1], 6)
    memory = NaturalSemanticSlotMemory(state, base_sha256=base_sha, bridge_sha256=bridge_sha)
    writes = [memory.write(fact, tokenizer, embedding) for fact in FACTS]
    initial_cases = (
        ('小林喜欢喝什么饮料？', '酸奶'),
        ('小林喜欢什么颜色？', '蓝色'),
        ('小梅喜欢喝什么饮料？', '红茶'),
        ('小赵喜欢什么颜色？', '未知'),
        ('小林喜欢什么动物？', '未知'),
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'slots.pt'
        memory.save(path, metadata={'phase': 'initial'})
        restored = NaturalSemanticSlotMemory.load(path, expected_base_sha256=base_sha,
                                                  expected_bridge_sha256=bridge_sha, device='cuda')
        initial_rows = evaluate(restored, initial_cases, 'initial')

        correction = restored.write('小林喜欢的颜色是绿色。', tokenizer, embedding)
        restored.save(path, metadata={'phase': 'corrected'})
        corrected = NaturalSemanticSlotMemory.load(path, expected_base_sha256=base_sha,
                                                   expected_bridge_sha256=bridge_sha, device='cuda')
        corrected_cases = (
            ('小林喜欢喝什么饮料？', '酸奶'),
            ('小林喜欢什么颜色？', '绿色'),
            ('小梅喜欢喝什么饮料？', '红茶'),
            ('小赵喜欢什么颜色？', '未知'),
        )
        corrected_rows = evaluate(corrected, corrected_cases, 'corrected')
    if p.sha(p.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Base model changed')
    known_initial = [row for row in initial_rows if row['expected'] != '未知']
    known_corrected = [row for row in corrected_rows if row['expected'] != '未知']
    report = {'status': 'natural_multirelation_audit', 'writes': writes,
              'correction': correction, 'initial': initial_rows, 'corrected': corrected_rows,
              'summary': {'initial_known_top1': sum(row.get('top1', False) for row in known_initial),
                          'initial_known_cases': len(known_initial),
                          'corrected_known_top1': sum(row.get('top1', False) for row in known_corrected),
                          'corrected_known_cases': len(known_corrected),
                          'unknown_no_route': sum(row.get('status') == 'no_route' for row in initial_rows if row['expected'] == '未知'),
                          'unsupported_rejected': sum(row.get('status') == 'rejected' for row in initial_rows)},
              'base_sha256': base_sha, 'bridge_sha256': bridge_sha,
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Two authored relation patterns only; no learned syntax parser.',
                              'The writer stores the first token of each value span.',
                              'First-token retrieval is measured; full response boundaries and free chat are not.',
                              'No conflict arbitration or sleep consolidation.']}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
