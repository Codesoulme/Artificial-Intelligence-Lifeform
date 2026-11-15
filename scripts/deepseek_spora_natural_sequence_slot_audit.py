"""Audit natural fact writes, multi-token output, correction, and reload."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch

from scripts import deepseek_local_baseline as p
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.research_semantic_slot import install_sequence_scaled, use_slot_sequence
from spora.semantic_phrase_slot_runtime import (NaturalSemanticPhraseMemory,
                                                 SEQUENCE_BRIDGE_BINDING)
from spora.semantic_phrase_slot_state import SemanticPhraseSlotState


OUT = ROOT / 'runs/deepseek-spora-natural-sequence-slot-audit-20261116.json'
FACTS = ('小林喜欢喝椰汁。', '小梅喜欢喝柠檬水。')


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
    install_sequence_scaled(model, scale=32.)
    embedding = model.model.embed_tokens.weight.detach()

    def generate(memory, question, expected):
        route = memory.query_slot(question)
        if route['slot'] is None:
            return {'question': question, 'expected': expected, 'status': 'no_route'}
        prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
        expected_ids = tokenizer.encode(expected, add_special_tokens=False)
        kwargs = dict(**prompt, use_cache=True, max_new_tokens=len(expected_ids) + 3,
                      do_sample=False, pad_token_id=tokenizer.pad_token_id,
                      eos_token_id=tokenizer.eos_token_id)
        with torch.inference_mode(), use_slot_sequence(route['value']):
            output = model.generate(**kwargs)
        tail = output[0, prompt['input_ids'].shape[1]:].tolist()
        return {'question': question, 'expected': expected, 'status': 'read',
                'route': {'subject': route['subject'], 'relation': route['relation'],
                          'slot': route['slot']},
                'expected_ids': expected_ids, 'generated_ids': tail,
                'exact_tokens': tail == expected_ids + [tokenizer.eos_token_id],
                'generated_text': tokenizer.decode(tail, skip_special_tokens=True)}

    state = SemanticPhraseSlotState(embedding.shape[-1], 4, max_tokens=16)
    memory = NaturalSemanticPhraseMemory(state, base_sha256=base_sha,
                                         bridge_sha256=SEQUENCE_BRIDGE_BINDING)
    writes = [memory.write(fact, tokenizer, embedding) for fact in FACTS]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'phrase-slots.pt'
        memory.save(path, metadata={'phase': 'initial'})
        restored = NaturalSemanticPhraseMemory.load(path, expected_base_sha256=base_sha,
                                                    expected_bridge_sha256=SEQUENCE_BRIDGE_BINDING,
                                                    device='cuda')
        initial = [generate(restored, '小林喜欢喝什么饮料？', '椰汁'),
                   generate(restored, '小梅喜欢喝什么饮料？', '柠檬水'),
                   generate(restored, '小赵喜欢喝什么饮料？', '未知')]
        correction = restored.write('小林喜欢喝矿泉水。', tokenizer, embedding)
        restored.save(path, metadata={'phase': 'corrected'})
        corrected = NaturalSemanticPhraseMemory.load(path, expected_base_sha256=base_sha,
                                                     expected_bridge_sha256=SEQUENCE_BRIDGE_BINDING,
                                                     device='cuda')
        after = [generate(corrected, '小林喜欢喝什么饮料？', '矿泉水'),
                 generate(corrected, '小梅喜欢喝什么饮料？', '柠檬水'),
                 generate(corrected, '小赵喜欢喝什么饮料？', '未知')]
    if p.sha(p.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Base model changed')
    known_initial = [row for row in initial if row['status'] == 'read']
    known_after = [row for row in after if row['status'] == 'read']
    report = {'status': 'natural_sequence_slot_audit', 'writes': writes,
              'correction': correction, 'initial': initial, 'corrected': after,
              'summary': {'initial_full_exact': sum(row['exact_tokens'] for row in known_initial),
                          'initial_known_cases': len(known_initial),
                          'corrected_full_exact': sum(row['exact_tokens'] for row in known_after),
                          'corrected_known_cases': len(known_after),
                          'unknown_no_route_initial': sum(row['status'] == 'no_route' for row in initial),
                          'unknown_no_route_corrected': sum(row['status'] == 'no_route' for row in after)},
              'base_sha256': base_sha, 'bridge_sha256': SEQUENCE_BRIDGE_BINDING,
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['The parser supports one relation pattern and the writer uses token embeddings directly.',
                              'The sequence reader is a capacity bridge, not a learned writer.',
                              'Only one relation and controlled multi-token values are evaluated.',
                              'No open-domain chat or sleep consolidation.']}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
