"""Test one SPORA layer replacement while native SPORA memory reads are active."""
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
from scripts.deepseek_named_binding_probe import KEYS
from scripts.deepseek_temporal_intent_router import current_addresses, temporal_addresses
from scripts.deepseek_temporal_intent_router import HELD as TEMPORAL_HELD
from spora.integrated_checkpoint import load_integrated
from spora.structure_transfer import SPORALayerReplacement

INDIVIDUAL = ROOT / 'runs/deepseek-temporal-intent-router-native-spora-site-20261110/historical-individual.pt'
CANDIDATE = ROOT / 'runs/deepseek-spora-block-distill-layer6-20261111/replacement.pt'
OUTPUT = ROOT / 'runs/deepseek-spora-memory-layer-interaction-v2-20261111.json'
ROWS = (
    {'question': TEMPORAL_HELD[0][0], 'intent': 'current', 'expected_value': 6},
    {'question': TEMPORAL_HELD[1][0], 'intent': 'past', 'expected_value': 1},
    {'question': TEMPORAL_HELD[2][0], 'intent': 'later', 'expected_value': 6},
    {'question': TEMPORAL_HELD[0][1], 'intent': 'current', 'expected_value': 6},
)


def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _metrics(teacher, candidate):
    p = teacher.float().softmax(-1)
    t = teacher.float()
    c = candidate.float()
    kl = (p * (t.log_softmax(-1) - c.log_softmax(-1))).sum(-1)
    return {
        'top1_agreement': int((teacher.argmax(-1) == candidate.argmax(-1)).sum()),
        'tokens': int(teacher.shape[0]),
        'kl_sum': float(kl.sum()),
        'max_logit_abs_diff': float((teacher.float() - c).abs().max()),
    }


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected DeepSeek base hash mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float16,
        attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    integrated, organism, state = load_integrated(INDIVIDUAL, base, MODEL, individual_device='cuda')
    payload = torch.load(CANDIDATE, map_location='cpu', weights_only=True)
    if payload.get('metadata', {}).get('base_weight_sha256') != WEIGHT_SHA or payload['metadata'].get('layer') != 6:
        raise ValueError('Layer-6 candidate provenance mismatch')
    replacement = SPORALayerReplacement(base.config.hidden_size, base.config.num_attention_heads,
                                        max_length=payload['metadata']['max_length'], layer_idx=6).to('cuda').eval()
    replacement.load_state_dict(payload['state'], strict=True)
    original = base.model.layers[6]
    key = KEYS[0]
    before_rows = []
    after_rows = []
    try:
        with torch.inference_mode():
            for row in ROWS:
                if row['intent'] == 'current':
                    addresses = current_addresses(organism, key, 'cuda')
                    slot = 0
                else:
                    event_time = 0 if row['intent'] == 'past' else 1
                    addresses = temporal_addresses(organism, key, event_time, 'cuda')
                    slot = 1
                inputs = closed_think_inputs(tokenizer, row['question'], 'cuda')
                kwargs = {'memory_state': state, 'memory_addresses': addresses,
                          'memory_slots': torch.tensor([slot], dtype=torch.long, device='cuda'),
                          'current_only': False, 'use_cache': False}
                teacher_logits = integrated(**inputs, **kwargs).logits[:, -1]
                base.model.layers[6] = replacement
                candidate_logits = integrated(**inputs, **kwargs).logits[:, -1]
                base.model.layers[6] = original
                before_rows.append((row, teacher_logits.detach().cpu()))
                after_rows.append((row, candidate_logits.detach().cpu()))
    finally:
        base.model.layers[6] = original
    results = []
    for (row, teacher), (_, candidate) in zip(before_rows, after_rows):
        item = _metrics(teacher, candidate)
        results.append({**row, **item,
                        'teacher_top1_token': int(teacher.argmax(-1)),
                        'candidate_top1_token': int(candidate.argmax(-1))})
    total_tokens = sum(item['tokens'] for item in results)
    report = {
        'status': 'native_memory_plus_layer6_replacement_development_probe',
        'base_weight_sha256': WEIGHT_SHA,
        'individual_sha256': sha(INDIVIDUAL),
        'candidate_sha256': _sha(CANDIDATE),
        'layers': [5, 6, 17],
        'memory_reader_active': True,
        'rows': results,
        'summary': {
            'top1_agreement': sum(item['top1_agreement'] for item in results) / total_tokens,
            'mean_token_kl_nats': sum(item['kl_sum'] for item in results) / total_tokens,
            'scored_tokens': total_tokens,
            'max_logit_abs_diff': max(item['max_logit_abs_diff'] for item in results),
        },
        'elapsed_seconds': time.perf_counter() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'device': torch.cuda.get_device_name(0),
        'limitations': [
            'Four authored temporal prompts, one fact address and one integrated state.',
            'Compares next-token distributions, not generated answers or memory-value accuracy.',
            'The layer-6 replacement was distilled without memory injection; this measures transfer under memory context only.',
            'No writes, multi-layer replacement, or concurrency tested.',
        ],
    }
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
