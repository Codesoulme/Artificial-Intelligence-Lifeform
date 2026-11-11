"""Compare output heads through the actual router with two interleaved entities."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))
import torch

from scripts.deepseek_local_baseline import WEIGHT_SHA, MODEL, sha
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_runtime_chat import load_chat
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest
from spora.memory_output import MemoryOutputAdapter
from spora.integrated_checkpoint import replacement_feature_variant_sha256
from spora.structure_transfer import SPORALayerReplacement

SOURCE = ROOT/'runs/deepseek-spora-memory-output-adapter-v2-20261111'
OUTPUT = ROOT/'runs/deepseek-spora-output-v2-routed-audit-20261111.json'
CASES = (
    ('小林现在喜欢喝什么？', 0, 0, 7),
    ('小林改口以前喜欢什么饮料？', 0, 1, 1),
    ('小林后来喜欢喝什么？', 0, 2, 7),
    ('阿宁现在喜欢喝什么？', 1, 0, 4),
    ('阿宁改口以前喜欢什么饮料？', 1, 1, 3),
    ('阿宁后来喜欢喝什么？', 1, 2, 4),
    ('小雪以前喜欢喝什么？', None, None, None),
    ('给小林写一句鼓励的话。', None, None, None),
    ('小林和阿宁分别喜欢喝什么？', None, None, None),
    ('小林的颜色偏好是什么？', None, None, None),
)


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    report = json.loads((SOURCE/'result.json').read_text(encoding='utf-8'))
    if sha(SOURCE/'adapter.pt') != report['adapter_sha256']:
        raise ValueError('Candidate digest mismatch')
    payload = torch.load(SOURCE/'adapter.pt', map_location='cpu', weights_only=True)
    if payload.get('base_sha256') != WEIGHT_SHA:
        raise ValueError('Candidate base mismatch')
    torch.set_num_threads(2)
    chat = load_chat()
    replacements = {i: layer for i, layer in enumerate(chat.base.model.layers)
                    if isinstance(layer, SPORALayerReplacement)}
    if replacement_feature_variant_sha256(MODEL, replacements) != chat.router['feature_variant_sha256']:
        raise ValueError('Router variant mismatch')
    for index in (6, 12):
        if payload['candidate_sha256'][str(index)] != sha(ROOT/'runs/deepseek-spora-diverse-memory-replay-20261111'/f'layer-{index}.pt'):
            raise ValueError('Candidate layer provenance mismatch')
    old = chat.decoder.output_adapter
    candidate = MemoryOutputAdapter(**payload['config']).to('cuda').eval()
    candidate.load_state_dict(payload['state'], strict=True)
    original_state = chat.state
    before = frozen_digest(chat.organism, original_state)
    chat.state = fresh_controlled_state(chat.organism)
    for entity, value in ((0, 1), (1, 3), (0, 7), (1, 4)):
        chat.state, decision = observe_sourced(chat.organism, chat.state, KEYS[entity], value, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('Controlled observation was not written')
    results = []
    for question, entity, intent, value in CASES:
        for label, adapter in (('old', old), ('candidate', candidate)):
            chat.decoder.output_adapter = adapter
            row = chat.answer(question)
            row.update({'adapter': label, 'expected_key': None if entity is None else KEYS[entity],
                        'expected_intent': intent, 'expected_value': value})
            row['route_correct'] = ((not row['accepted']) if entity is None else
                                    row['accepted'] and row.get('key') == KEYS[entity]
                                    and row['temporal_intent'] == intent)
            row['exact'] = bool(entity is not None and row['answer'] in
                                (WORDS[value], WORDS[value]+'.', WORDS[value]+'。'))
            results.append(row)
        print(json.dumps({'question_index': len(results)//2, 'total': len(CASES)}), flush=True)
    chat.decoder.output_adapter = old
    if frozen_digest(chat.organism, original_state) != before:
        raise ValueError('Protected source state changed')
    scores = {}
    for label in ('old', 'candidate'):
        subset = [r for r in results if r['adapter'] == label]
        scores[label] = {'positive_routes': sum(r['route_correct'] for r in subset if r['expected_value'] is not None),
                         'positive_exact': sum(r['exact'] for r in subset),
                         'negative_rejected': sum(not r['accepted'] for r in subset if r['expected_value'] is None)}
    output = {'candidate_sha256': sha(SOURCE/'adapter.pt'), 'scores': scores, 'rows': results,
              'limitations': ['Two bound names and six authored positive questions; not an independent benchmark.',
                              'Controlled interleaved facts, not automatic observation or persistence.',
                              'No general dialogue or open-domain memory claim.',
                              'Positive questions and finite drink vocabulary may overlap prior development.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(scores), flush=True)


if __name__ == '__main__':
    run()
