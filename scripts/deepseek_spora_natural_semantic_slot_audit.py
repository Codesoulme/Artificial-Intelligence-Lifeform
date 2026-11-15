"""Audit natural-language semantic writes, routing, correction, and reload."""
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
OUT = ROOT / 'runs/deepseek-spora-natural-semantic-slot-audit-20261116.json'
FACTS = ('小林喜欢喝酸奶。', '小梅喜欢喝红茶。')
CORRECTION = '小林喜欢喝豆浆。'
QUESTIONS = ('小林喜欢喝什么饮料？', '小梅喜欢喝什么饮料？', '小赵喜欢喝什么饮料？')


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

    def first_token(memory, question):
        route = memory.query_slot(question)
        prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
        if route['slot'] is None:
            with torch.inference_mode():
                logits = model(**prompt, use_cache=False, return_dict=True).logits[0, -1].float()
            return {'route': route, 'rank': None,
                    'winner': tokenizer.decode([int(logits.argmax())], skip_special_tokens=False)}
        value = route['value']
        with torch.inference_mode(), use_slot(value):
            logits = model(**prompt, use_cache=False, return_dict=True).logits[0, -1].float()
        return {'route': {'subject': route['subject'], 'slot': route['slot']},
                'rank': None, 'winner': tokenizer.decode([int(logits.argmax())], skip_special_tokens=False)}

    def check(memory, expected):
        rows = []
        for question, value in expected.items():
            result = first_token(memory, question)
            token_id = tokenizer.encode(value, add_special_tokens=False)[0]
            if result['route']['slot'] is None:
                result['exact_first_token'] = False
            else:
                with torch.inference_mode(), use_slot(result.pop('_slot', memory.query_slot(question)['value'])):
                    prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
                    logits = model(**prompt, use_cache=False, return_dict=True).logits[0, -1].float()
                result['rank'] = int((logits > logits[token_id]).sum()) + 1
                result['exact_first_token'] = result['rank'] == 1
            result.update({'question': question, 'expected': value})
            rows.append(result)
        return rows

    # Use the model's tokenizer and embedding as the natural writer's input;
    # the writer does not receive a fixed value vocabulary or a value index.
    state = SemanticSlotState(embedding.shape[-1], 4)
    memory = NaturalSemanticSlotMemory(state, base_sha256=base_sha, bridge_sha256=bridge_sha)
    writes = [memory.write(fact, tokenizer, embedding) for fact in FACTS]
    before_expected = {QUESTIONS[0]: '酸奶', QUESTIONS[1]: '红茶', QUESTIONS[2]: '未知'}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'slots.pt'
        memory.save(path, metadata={'phase': 'initial'})
        initial = NaturalSemanticSlotMemory.load(path, expected_base_sha256=base_sha,
                                                 expected_bridge_sha256=bridge_sha, device='cuda')
        initial_rows = []
        for question, value in before_expected.items():
            route = initial.query_slot(question)
            row = {'phase': 'initial', 'question': question, 'expected': value,
                   'route': {'subject': route['subject'], 'slot': route['slot']}}
            if route['slot'] is not None and value != '未知':
                token_id = tokenizer.encode(value, add_special_tokens=False)[0]
                prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
                with torch.inference_mode(), use_slot(route['value']):
                    logits = model(**prompt, use_cache=False, return_dict=True).logits[0, -1].float()
                row['rank'] = int((logits > logits[token_id]).sum()) + 1
                row['exact_first_token'] = row['rank'] == 1
            else:
                row['rank'] = None
                row['exact_first_token'] = route['slot'] is not None
            initial_rows.append(row)

        correction = initial.write(CORRECTION, tokenizer, embedding)
        initial.save(path, metadata={'phase': 'corrected'})
        restored = NaturalSemanticSlotMemory.load(path, expected_base_sha256=base_sha,
                                                  expected_bridge_sha256=bridge_sha, device='cuda')
        corrected_rows = []
        expected_after = {QUESTIONS[0]: '豆浆', QUESTIONS[1]: '红茶', QUESTIONS[2]: '未知'}
        for question, value in expected_after.items():
            route = restored.query_slot(question)
            row = {'phase': 'corrected', 'question': question, 'expected': value,
                   'route': {'subject': route['subject'], 'slot': route['slot']}}
            if route['slot'] is not None and value != '未知':
                token_id = tokenizer.encode(value, add_special_tokens=False)[0]
                prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
                with torch.inference_mode(), use_slot(route['value']):
                    logits = model(**prompt, use_cache=False, return_dict=True).logits[0, -1].float()
                row['rank'] = int((logits > logits[token_id]).sum()) + 1
                row['exact_first_token'] = row['rank'] == 1
            else:
                row['rank'] = None
                row['exact_first_token'] = route['slot'] is not None
            corrected_rows.append(row)

    if p.sha(p.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Base model changed')
    report = {'status': 'natural_semantic_slot_audit', 'writes': writes,
              'correction': correction, 'initial': initial_rows, 'corrected': corrected_rows,
              'summary': {'initial_known_top1': sum(row['exact_first_token'] for row in initial_rows if row['expected'] != '未知'),
                          'corrected_known_top1': sum(row['exact_first_token'] for row in corrected_rows if row['expected'] != '未知'),
                          'unknown_initial_no_route': sum(row['route']['slot'] is None for row in initial_rows if row['expected'] == '未知'),
                          'unknown_corrected_no_route': sum(row['route']['slot'] is None for row in corrected_rows if row['expected'] == '未知')},
              'base_sha256': base_sha, 'bridge_sha256': bridge_sha,
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['The parser handles one controlled Chinese relation and does not infer arbitrary syntax.',
                              'The value embedding is obtained from the parsed text span; there is no learned write encoder.',
                              'Only first-token retrieval is evaluated; full answer stopping and open chat remain unsolved.',
                              'No concurrent writer, conflict resolution, or sleep consolidation is included.']}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
