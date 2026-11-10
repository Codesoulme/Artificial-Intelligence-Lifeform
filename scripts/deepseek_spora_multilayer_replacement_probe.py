"""Measure error interaction when replacing separated Qwen layers by SPORA blocks."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_block_distill import _encode, _held_messages
from spora.structure_transfer import SPORALayerReplacement

LAYER_6 = ROOT / 'runs/deepseek-spora-block-distill-layer6-20261111/replacement.pt'
LAYER_12 = ROOT / 'runs/deepseek-spora-block-distill-layer12-logit-warmstart-20261111/replacement.pt'
OUTPUT = ROOT / 'runs/deepseek-spora-multilayer-replacement-20261111.json'
GENERALIZATION_OUTPUT = ROOT / 'runs/deepseek-spora-multilayer-replacement-generalization-20261111.json'
GENERALIZATION_SOURCE = ROOT / 'data/conversation-development.json'
LAYERS = (6, 12)


def _load_replacement(path, base, layer):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    metadata = payload.get('metadata', {})
    if (metadata.get('base_weight_sha256') != WEIGHT_SHA or metadata.get('layer') != layer
            or metadata.get('format') != 'spora-qwen-layer-replacement-candidate-v1'):
        raise ValueError(f'Layer {layer} candidate provenance mismatch')
    module = SPORALayerReplacement(base.config.hidden_size, base.config.num_attention_heads,
                                   max_length=metadata['max_length'], layer_idx=layer).to('cuda').eval()
    module.load_state_dict(payload['state'], strict=True)
    return module


@torch.inference_mode()
def _metrics(teacher_logits, candidate_logits):
    teacher = teacher_logits[:, :-1].float()
    candidate = candidate_logits[:, :-1].float()
    p = teacher.softmax(-1)
    kl = (p * (teacher.log_softmax(-1) - candidate.log_softmax(-1))).sum(-1)
    return {
        'top1_agreement': int((teacher.argmax(-1) == candidate.argmax(-1)).sum()),
        'tokens': int(teacher.shape[0] * teacher.shape[1]),
        'kl_sum': float(kl.sum()),
    }


def _messages(path):
    with path.open(encoding='utf-8') as stream:
        data = json.load(stream)
    return [[{'role': 'user', 'content': turn}]
            for scenario in data['scenarios'] for turn in scenario['turns']]


def run(generalization=False):
    output = GENERALIZATION_OUTPUT if generalization else OUTPUT
    held_source = GENERALIZATION_SOURCE if generalization else ROOT / 'data/conversation-development-v2.json'
    if output.exists():
        raise FileExistsError(output)
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected DeepSeek weight digest mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float16,
        attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    candidates = {6: _load_replacement(LAYER_6, base, 6), 12: _load_replacement(LAYER_12, base, 12)}
    originals = {index: base.model.layers[index] for index in LAYERS}
    conditions = {'layer6': (6,), 'layer12': (12,), 'layer6_plus_layer12': LAYERS}
    totals = {name: {'top1_agreement': 0, 'tokens': 0, 'kl_sum': 0.0} for name in conditions}
    try:
        held = _messages(held_source)
        for messages in held:
            ids = _encode(tokenizer, messages)
            mask = torch.ones_like(ids)
            teacher = base(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).logits
            for name, selected in conditions.items():
                for index in LAYERS:
                    base.model.layers[index] = candidates[index] if index in selected else originals[index]
                logits = base(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).logits
                measured = _metrics(teacher, logits)
                for key in ('top1_agreement', 'tokens'):
                    totals[name][key] += measured[key]
                totals[name]['kl_sum'] += measured['kl_sum']
    finally:
        for index in LAYERS:
            base.model.layers[index] = originals[index]
    results = {}
    for name, item in totals.items():
        results[name] = {'token_top1_agreement': item['top1_agreement'] / item['tokens'],
                         'mean_token_kl_nats': item['kl_sum'] / item['tokens'],
                         'scored_next_tokens': item['tokens']}
    report = {
        'status': 'two_layer_spora_replacement_development_probe',
        'base_weight_sha256': WEIGHT_SHA,
        'held_source_sha256': __import__('hashlib').sha256(held_source.read_bytes()).hexdigest(),
        'layers': list(LAYERS),
        'conditions': results,
        'held_sequences': len(held),
        'elapsed_seconds': time.perf_counter() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'device': torch.cuda.get_device_name(0),
        'limitations': [
            'Reuses a layer-12 candidate distilled on layer-12 activations; the paired layer-6 shift is not jointly distilled.',
            'No SPORA memory reader is active in this measurement.',
            'Project development sequences only; not a chat capability evaluation.',
            'Full-sequence logits only; KV-cache generation was checked separately for layer 12.',
        ],
    }
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generalization', action='store_true')
    run(generalization=parser.parse_args().generalization)
