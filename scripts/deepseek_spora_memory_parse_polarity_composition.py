"""Compose subject, polarity, and open-value gates without persistent writes."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_value_span_probe import SpanHead, build as build_values, feature_batch, decode as decode_values
from scripts.deepseek_spora_observation_transfer_audit import CHECKPOINT
from scripts.deepseek_spora_cluener_locator_retrain import OUTPUT as LOCATOR
from scripts.deepseek_spora_polarity_gate_probe import CLASSES, OUTPUT as POLARITY
from spora.integrated_checkpoint import load_integrated, replacement_feature_variant_sha256
from spora.structure_transfer import SPORALayerReplacement

VALUE = ROOT / 'runs/deepseek-spora-value-span-probe-v3-20261111/span-head.pt'
OUTPUT = ROOT / 'runs/deepseek-spora-memory-parse-polarity-composition-20261111.json'


def pool(vectors, mask):
    weights = mask[:, None, :, None].float()
    means = (vectors.float() * weights).sum(2) / weights.sum(2).clamp_min(1)
    return torch.cat([F.layer_norm(means[:, i], (means.shape[-1],)) for i in range(2)], dim=-1)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    locator_file = LOCATOR / 'locator.pt'
    polarity_file = POLARITY / 'gate.pt'
    value_file = VALUE
    paths = (MODEL / 'model.safetensors', CHECKPOINT, locator_file, polarity_file, value_file)
    source = {str(path): sha(path) for path in paths}
    if source[str(MODEL / 'model.safetensors')] != WEIGHT_SHA:
        raise ValueError('Provenance mismatch')
    locator_data = torch.load(locator_file, map_location='cpu', weights_only=True)
    polarity_data = torch.load(polarity_file, map_location='cpu', weights_only=True)
    value_data = torch.load(value_file, map_location='cpu', weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    decoder, organism, state = load_integrated(CHECKPOINT, base, MODEL, individual_device='cuda')
    del decoder, organism, state
    blocks = {i: layer for i, layer in enumerate(base.model.layers) if isinstance(layer, SPORALayerReplacement)}
    variant = replacement_feature_variant_sha256(MODEL, blocks)
    if (locator_data.get('feature_variant_sha256') != variant or polarity_data.get('feature_variant_sha256') != variant
            or value_data.get('source_checkpoint_sha256') != source[str(CHECKPOINT)]):
        raise ValueError('Component structural provenance mismatch')
    rows = build_values('held')
    vectors, masks, offsets, starts = feature_batch(base, tokenizer, rows)
    token_vectors = torch.cat([F.layer_norm(vectors[:, i].float(), (vectors.shape[-1],)) for i in range(2)], dim=-1)
    locator = nn.Linear(locator_data['state']['weight'].shape[1], 1).eval()
    locator.load_state_dict(locator_data['state'], strict=True)
    polarity = nn.Linear(vectors.shape[-1] * 2, len(CLASSES)).eval()
    polarity.load_state_dict(polarity_data['state'], strict=True)
    value = SpanHead(vectors.shape[-1]).eval()
    value.load_state_dict(value_data['state'], strict=True)
    with torch.inference_mode():
        subject_scores = locator(token_vectors).squeeze(-1).sigmoid()
        polarity_probabilities = polarity(pool(vectors, masks)).softmax(-1)
    value_rows = decode_values(rows, value, vectors, offsets, starts, masks, value_data['threshold'])
    result = []
    for index, (row, subject_score, probabilities, value_row) in enumerate(zip(rows, subject_scores, polarity_probabilities, value_rows)):
        _, subject_single = __import__('scripts.deepseek_learned_name_span', fromlist=['single_span_mask']).single_span_mask(subject_score, masks[index])
        polarity_class = int(probabilities.argmax())
        polarity_ok = polarity_class == 0 and float(probabilities[0]) >= polarity_data['threshold']
        result.append({'text': row['text'], 'positive': row['positive'], 'value': value_row['value'],
                       'subject_single': subject_single, 'polarity_class': CLASSES[polarity_class],
                       'polarity_ok': polarity_ok, 'value_accepted': value_row['accepted'],
                       'composed_accept': bool(subject_single and polarity_ok and value_row['accepted'])})
    positives = [row for row in result if row['positive']]
    negatives = [row for row in result if not row['positive']]
    summary = {'examples': len(result), 'positive_composed_accept': sum(row['composed_accept'] for row in positives),
               'positive_total': len(positives), 'negative_composed_accept': sum(row['composed_accept'] for row in negatives),
               'negative_total': len(negatives), 'polarity_negative_accept': sum(row['polarity_ok'] for row in negatives),
               'value_only_negative_accept': sum(row['value_accepted'] for row in negatives)}
    if any(sha(Path(path)) != digest for path, digest in source.items()):
        raise ValueError('Protected input changed')
    output = {'status': 'development_only_no_writes', 'summary': summary, 'source_hashes': source,
              'feature_variant_sha256': variant, 'replacement_layers': sorted(blocks), 'rows': result,
              'seconds': time.perf_counter() - started,
              'limitations': ['Synthetic held split only; no independent dialogue.',
                              'No source, temporal, correction, authorization or address binding.',
                              'Calibration was finite and does not guarantee zero false writes.',
                              'Composition remains disconnected from persistence.']}
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
