"""Factor memory read strength against base and SPORA layer-replacement pairs."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.deepseek_spora_block_distill import _hash
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.integrated_checkpoint import load_integrated
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest
from spora.structure_transfer import SPORALayerReplacement

SOURCE_CHECKPOINT = ROOT / 'runs/deepseek-temporal-intent-router-native-spora-site-20261110/historical-individual.pt'
OUTPUT = ROOT / 'runs/deepseek-spora-memory-strength-factor-20261111.json'
STRENGTHS = (0.1, 0.25, 0.5, 0.75)
CONDITION_PATHS = {
    'plain_pair': {
        6: ROOT / 'runs/deepseek-spora-block-distill-layer6-20261111/replacement.pt',
        12: ROOT / 'runs/deepseek-spora-block-distill-layer12-logit-warmstart-20261111/replacement.pt',
    },
    'memory_conditioned_layer6': {
        6: ROOT / 'runs/deepseek-spora-memory-conditioned-layer6-20261111/replacement.pt',
        12: ROOT / 'runs/deepseek-spora-block-distill-layer12-logit-warmstart-20261111/replacement.pt',
    },
    'joint_memory_v2': {
        6: ROOT / 'runs/deepseek-spora-joint-memory-distill-v2-20261111/layer-6.pt',
        12: ROOT / 'runs/deepseek-spora-joint-memory-distill-v2-20261111/layer-12.pt',
    },
}
QUESTIONS = {
    'current': ('{name}现在喜欢什么饮品？', '目前{name}偏爱喝什么？'),
    'past': ('{name}过去喜欢哪种饮料？', '更正前{name}喜欢喝什么？'),
    'later': ('更正后{name}改喜欢什么饮品？', '{name}后来最喜欢喝什么？'),
}


def _module(path, layer, base):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    meta = payload.get('metadata', {})
    if (meta.get('format') != 'spora-qwen-layer-replacement-candidate-v1'
            or meta.get('base_weight_sha256') != WEIGHT_SHA or meta.get('layer') != layer):
        raise ValueError(f'Candidate provenance mismatch at layer {layer}')
    replacement = SPORALayerReplacement(base.config.hidden_size, base.config.num_attention_heads,
                                        max_length=meta['max_length'], layer_idx=layer).to('cuda').eval()
    replacement.load_state_dict(payload['state'], strict=True)
    return replacement


def _address(organism, key, intent, key_index):
    keys = torch.tensor([key], dtype=torch.long, device='cuda')
    if intent == 'current':
        addresses = [torch.nn.functional.normalize(site.current_basis[key:key + 1].to('cuda').float(), dim=-1)
                     for site in organism.memories]
        return addresses, 0
    event_time = 2 * key_index if intent == 'past' else 2 * key_index + 1
    modes = torch.ones_like(keys)
    times = torch.tensor([event_time], dtype=torch.long, device='cuda')
    _, bound = organism.representation(keys, modes, times)
    return [organism.memory_address(site, bound, keys, modes) for site in organism.memories], 1


def _metric(reference, candidate):
    ref, hyp = reference.float(), candidate.float()
    p = ref.softmax(-1)
    kl = (p * (ref.log_softmax(-1) - hyp.log_softmax(-1))).sum(-1)
    return {'top1_agreement': int((ref.argmax(-1) == hyp.argmax(-1)).sum()),
            'kl_nats': float(kl.mean()), 'max_logit_abs_diff': float((ref - hyp).abs().max())}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA or sha(INDIVIDUAL) != INDIVIDUAL_SHA:
        raise ValueError('Protected model or original individual hash mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float16,
        attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    integrated, organism, _ = load_integrated(SOURCE_CHECKPOINT, base, MODEL, individual_device='cuda')
    native_strength = integrated.reader.strength
    state = fresh_controlled_state(organism)
    entity_rows = []
    for index, (name, key) in enumerate(zip(NAMES[:4], KEYS[:4])):
        for value in (1, 6):
            state, decision = observe_sourced(organism, state, key, value, 'direct', 1.0)
            if not decision['current_write']:
                raise ValueError(f'Controlled write rejected for {name}: {decision}')
        for intent, templates in QUESTIONS.items():
            addresses, slot = _address(organism, key, intent, index)
            for template in templates:
                text = template.format(name=name)
                inputs = closed_think_inputs(tokenizer, text, 'cuda')
                entity_rows.append({'name': name, 'key': key, 'intent': intent, 'text': text,
                                    'addresses': addresses, 'slot': slot, 'inputs': inputs})
    state_digest = frozen_digest(organism, state)
    originals = {layer: base.model.layers[layer] for layer in (6, 12)}
    candidates = {name: {layer: _module(path, layer, base) for layer, path in paths.items()}
                  for name, paths in CONDITION_PATHS.items()}
    candidate_digests = {name: {str(layer): sha(path) for layer, path in paths.items()}
                         for name, paths in CONDITION_PATHS.items()}
    results = []
    try:
        with torch.inference_mode():
            for strength in STRENGTHS:
                integrated.reader.strength = strength
                for row in entity_rows:
                    args = {**row['inputs'], 'memory_state': state,
                            'memory_addresses': row['addresses'],
                            'memory_slots': torch.tensor([row['slot']], dtype=torch.long, device='cuda'),
                            'current_only': False, 'use_cache': False}
                    base.model.layers[6], base.model.layers[12] = originals[6], originals[12]
                    reference = integrated(**args).logits[:, -1].detach().cpu()
                    for condition, layers in [('base', None), *candidates.items()]:
                        if layers is None:
                            continue
                        base.model.layers[6] = layers[6]
                        base.model.layers[12] = layers[12]
                        logits = integrated(**args).logits[:, -1].detach().cpu()
                        results.append({'strength': strength, 'condition': condition,
                                        'name': row['name'], 'intent': row['intent'],
                                        'question': row['text'], **_metric(reference, logits)})
    finally:
        base.model.layers[6], base.model.layers[12] = originals[6], originals[12]
        integrated.reader.strength = native_strength
    summary = {}
    for strength in STRENGTHS:
        for condition in candidates:
            rows = [r for r in results if r['strength'] == strength and r['condition'] == condition]
            summary[f'{condition}@{strength}'] = {
                'top1_agreement': sum(r['top1_agreement'] for r in rows) / len(rows),
                'mean_kl_nats': sum(r['kl_nats'] for r in rows) / len(rows),
                'max_logit_abs_diff': max(r['max_logit_abs_diff'] for r in rows),
                'queries': len(rows),
            }
    report = {
        'status': 'multi_entity_memory_strength_and_structure_factor_probe',
        'base_weight_sha256': WEIGHT_SHA,
        'original_individual_sha256': INDIVIDUAL_SHA,
        'integrated_checkpoint_sha256': sha(SOURCE_CHECKPOINT),
        'individual_frozen_digest': state_digest,
        'native_reader_strength': native_strength,
        'strengths': list(STRENGTHS),
        'entities': list(NAMES[:4]),
        'queries': len(entity_rows),
        'candidate_sha256': candidate_digests,
        'summary': summary,
        'rows': results,
        'elapsed_seconds': time.perf_counter() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'device': torch.cuda.get_device_name(0),
        'limitations': [
            'Four fixed entities, one corrected fact each and 24 authored questions.',
            'Measures next-token agreement with the same integrated base at each tested reader strength.',
            'No generated-answer quality or automatic entity/time routing evaluated.',
            'Strength values below the native 0.5 setting are diagnostic only, not an accepted runtime policy.',
        ],
    }
    if (sha(MODEL / 'model.safetensors') != WEIGHT_SHA or sha(INDIVIDUAL) != INDIVIDUAL_SHA
            or frozen_digest(organism, state) != state_digest):
        raise ValueError('Protected source or memory state changed')
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
