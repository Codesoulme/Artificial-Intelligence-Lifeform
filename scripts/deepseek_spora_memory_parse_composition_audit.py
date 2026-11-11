"""Audit composed subject and open-value parsing on replaced features."""
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
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_observation_retrain import SPLITS
from scripts.deepseek_spora_value_span_probe import SpanHead, build as build_values, feature_batch, decode as decode_values
from scripts.deepseek_spora_observation_transfer_audit import CHECKPOINT
from scripts.deepseek_spora_cluener_locator_retrain import OUTPUT as LOCATOR
from spora.integrated_checkpoint import load_integrated, replacement_feature_variant_sha256
from spora.structure_transfer import SPORALayerReplacement

VALUE = ROOT / 'runs/deepseek-spora-value-span-probe-v3-20261111/span-head.pt'
OUTPUT = ROOT / 'runs/deepseek-spora-memory-parse-composition-20261111.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    locator_file = LOCATOR / 'locator.pt'
    locator_data = torch.load(locator_file, map_location='cpu', weights_only=True)
    value_data = torch.load(VALUE, map_location='cpu', weights_only=True)
    source_paths = (MODEL / 'model.safetensors', CHECKPOINT, locator_file, VALUE)
    source = {str(path): sha(path) for path in source_paths}
    if source[str(MODEL / 'model.safetensors')] != WEIGHT_SHA:
        raise ValueError('Base provenance mismatch')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    torch.manual_seed(90114)
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
    if locator_data.get('feature_variant_sha256') != variant or value_data.get('source_checkpoint_sha256') != source[str(CHECKPOINT)]:
        raise ValueError('Component feature provenance mismatch')
    rows = build_values('held')
    vectors, masks, offsets, starts = feature_batch(base, tokenizer, rows)
    locator = nn.Linear(locator_data['state']['weight'].shape[1], 1).eval()
    locator.load_state_dict(locator_data['state'], strict=True)
    value = SpanHead(vectors.shape[-1]).eval()
    value.load_state_dict(value_data['state'], strict=True)
    token_vectors = torch.cat([F.layer_norm(vectors[:, i].float(), (vectors.shape[-1],)) for i in range(2)], dim=-1)
    with torch.inference_mode():
        locator_scores = locator(token_vectors).squeeze(-1).sigmoid()
    value_rows = decode_values(rows, value, vectors, offsets, starts, masks, value_data['threshold'])
    result = []
    for index, (row, score, value_row) in enumerate(zip(rows, locator_scores, value_rows)):
        _, single = __import__('scripts.deepseek_learned_name_span', fromlist=['single_span_mask']).single_span_mask(score, masks[index])
        result.append({'text': row['text'], 'positive': row['positive'], 'value_accepted': value_row['accepted'],
                       'value': value_row['value'], 'subject_single': single,
                       'composed_accept': bool(single and value_row['accepted']),
                       'subject_score': float(score[masks[index]].max())})
    positives = [row for row in result if row['positive']]
    negatives = [row for row in result if not row['positive']]
    summary = {'examples': len(result), 'positive_composed_accept': sum(row['composed_accept'] for row in positives),
               'positive_total': len(positives), 'negative_composed_accept': sum(row['composed_accept'] for row in negatives),
               'negative_total': len(negatives), 'value_only_negative_accept': sum(row['value_accepted'] for row in negatives),
               'subject_rejected_value_accept': sum(row['value_accepted'] and not row['subject_single'] for row in negatives)}
    if any(sha(Path(path)) != digest for path, digest in source.items()):
        raise ValueError('Protected input changed')
    report = {'status': 'development_only_no_writes', 'summary': summary,
              'source_hashes': source, 'feature_variant_sha256': variant,
              'replacement_layers': sorted(blocks), 'rows': result,
              'seconds': time.perf_counter() - started,
              'limitations': ['Held synthetic split only; no independent dialogue.',
                              'Subject locator and value span model use different training corpora.',
                              'No source, polarity, temporal semantics, address binding or write authorization.',
                              'Composition acceptance is not evidence that the extracted fact is true.']}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
