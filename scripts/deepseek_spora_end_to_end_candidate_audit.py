"""Audit the full replaced-feature observation route without writing memory."""
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
from scripts.deepseek_named_binding_probe import KEYS, NAMES, pooled, predict
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_cluener_locator_retrain import OUTPUT as LOCATOR
from scripts.deepseek_spora_memory_parse_polarity_composition import pool
from scripts.deepseek_spora_value_span_probe import SpanHead, build, decode as decode_values, feature_batch
from scripts.deepseek_observation_value_probe import WORDS
from scripts.deepseek_spora_observation_transfer_audit import CHECKPOINT
from scripts.deepseek_spora_polarity_gate_probe import CLASSES, OUTPUT as POLARITY
from scripts.deepseek_spora_named_binding_retrain import OUTPUT as BINDING
from scripts.deepseek_learned_name_span import single_span_mask
from spora.integrated_checkpoint import load_integrated, replacement_feature_variant_sha256
from spora.observation_candidate import ObservationCandidate, write_guard
from spora.structure_transfer import SPORALayerReplacement

VALUE = ROOT / 'runs/deepseek-spora-value-span-probe-v3-20261111/span-head.pt'
OUTPUT = ROOT / 'runs/deepseek-spora-end-to-end-candidate-audit-20261111.json'
NAME_TO_KEY = dict(zip(NAMES, KEYS))


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    paths = (MODEL / 'model.safetensors', CHECKPOINT, LOCATOR / 'locator.pt',
             BINDING / 'binding.pt', POLARITY / 'gate.pt', VALUE)
    source = {str(path): sha(path) for path in paths}
    if source[str(MODEL / 'model.safetensors')] != WEIGHT_SHA:
        raise ValueError('Base provenance mismatch')
    locator_data = torch.load(paths[2], map_location='cpu', weights_only=True)
    binding_data = torch.load(paths[3], map_location='cpu', weights_only=True)
    polarity_data = torch.load(paths[4], map_location='cpu', weights_only=True)
    value_data = torch.load(paths[5], map_location='cpu', weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    decoder, organism, state = load_integrated(CHECKPOINT, base, MODEL, individual_device='cuda')
    del decoder
    blocks = {i: layer for i, layer in enumerate(base.model.layers) if isinstance(layer, SPORALayerReplacement)}
    variant = replacement_feature_variant_sha256(MODEL, blocks)
    if any(data.get('feature_variant_sha256') != variant for data in (locator_data, binding_data, polarity_data)):
        raise ValueError('Structural feature provenance mismatch')
    if value_data.get('source_checkpoint_sha256') != source[str(CHECKPOINT)]:
        raise ValueError('Value probe provenance mismatch')
    rows = build('held')
    vectors, masks, offsets, starts = feature_batch(base, tokenizer, rows)
    token_vectors = torch.cat([F.layer_norm(vectors[:, i].float(), (vectors.shape[-1],)) for i in range(2)], dim=-1)
    locator = nn.Linear(locator_data['state']['weight'].shape[1], 1).eval()
    locator.load_state_dict(locator_data['state'], strict=True)
    polarity = nn.Linear(vectors.shape[-1] * 2, len(CLASSES)).eval()
    polarity.load_state_dict(polarity_data['state'], strict=True)
    value = SpanHead(vectors.shape[-1]).eval()
    value.load_state_dict(value_data['state'], strict=True)
    bases = torch.stack([F.normalize(site.current_basis.detach().float().cpu(), dim=-1)
                         for site in organism.memories])
    with torch.inference_mode():
        subject_scores = locator(token_vectors).squeeze(-1).sigmoid()
        polarity_probabilities = polarity(pool(vectors, masks)).softmax(-1)
    value_rows = decode_values(rows, value, vectors, offsets, starts, masks, value_data['threshold'])
    span_vectors = []
    subject_flags = []
    for index, score in enumerate(subject_scores):
        selected, single = single_span_mask(score, masks[index])
        subject_flags.append(single)
        span_vectors.append(pooled(vectors[index:index + 1], selected.unsqueeze(0))[0])
    span_vectors = torch.stack(span_vectors)
    binding_indices, binding_scores = predict(span_vectors, binding_data['weights'], bases)
    rows_out = []
    for index, (row, probabilities, value_row) in enumerate(zip(rows, polarity_probabilities, value_rows)):
        polarity_index = int(probabilities.argmax())
        polarity_ok = polarity_index == 0 and float(probabilities[0]) >= polarity_data['threshold']
        address = binding_indices[index].tolist()
        address_scores = binding_scores[index].tolist()
        address_ok = subject_flags[index] and address[0] == address[1] and min(address_scores) >= binding_data['threshold']
        name = next((name for name in NAME_TO_KEY if name in row['text']), None)
        value_text = value_row['value']
        value_index = WORDS.index(value_text) if value_text in WORDS else None
        candidate = ObservationCandidate(NAME_TO_KEY.get(name) if address_ok else None, value_text,
                                         value_index, CLASSES[polarity_index], 'direct', 1.0,
                                         state.cursor, float(probabilities[0]), row['text'])
        guard = write_guard(candidate, key_count=organism.config.key_count,
                            value_count=organism.config.value_count, current_cursor=state.cursor)
        rows_out.append({'text': row['text'], 'positive': row['positive'], 'name': name,
                         'predicted_address': address, 'address_ok': address_ok,
                         'polarity': CLASSES[polarity_index], 'polarity_ok': polarity_ok,
                         'value': value_text, 'value_index': value_index,
                         'value_accepted': value_row['accepted'], 'write_guard': guard})
    positives = [row for row in rows_out if row['positive']]
    negatives = [row for row in rows_out if not row['positive']]
    summary = {'examples': len(rows_out), 'known_positive_write_allowed': sum(row['write_guard']['write'] for row in positives if row['value_index'] is not None),
               'known_positive_total': sum(row['value_index'] is not None for row in positives),
               'unknown_positive_deferred': sum(not row['write_guard']['write'] for row in positives if row['value_index'] is None),
               'negative_write_allowed': sum(row['write_guard']['write'] for row in negatives),
               'negative_total': len(negatives), 'positive_address_correct': sum(row['address_ok'] for row in positives if row['name'] in NAME_TO_KEY)}
    if any(sha(Path(path)) != digest for path, digest in source.items()):
        raise ValueError('Protected input changed')
    report = {'status': 'development_only_no_writes', 'summary': summary, 'source_hashes': source,
              'feature_variant_sha256': variant, 'replacement_layers': sorted(blocks), 'rows': rows_out,
              'seconds': time.perf_counter() - started,
              'limitations': ['Held synthetic split; no independent memory dialogue.',
                              'The direct source and current time are supplied by the audit, not inferred by the parser.',
                              'Unknown open-vocabulary values are intentionally deferred by the finite write guard.',
                              'No state mutation, checkpoint write, or response generation occurred.']}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
