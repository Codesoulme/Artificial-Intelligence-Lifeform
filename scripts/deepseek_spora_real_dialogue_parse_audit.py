"""Read-only audit of the composed parser on selected natural Chinese turns."""
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
from scripts.deepseek_spora_cluener_locator_retrain import OUTPUT as LOCATOR
from scripts.deepseek_spora_value_span_probe import SpanHead, feature_batch, decode as decode_values
from scripts.deepseek_spora_observation_transfer_audit import CHECKPOINT
from scripts.deepseek_spora_polarity_gate_probe import CLASSES, OUTPUT as POLARITY
from scripts.deepseek_spora_memory_parse_polarity_composition import pool
from spora.integrated_checkpoint import load_integrated, replacement_feature_variant_sha256
from spora.structure_transfer import SPORALayerReplacement

VALUE = ROOT / 'runs/deepseek-spora-value-span-probe-v3-20261111/span-head.pt'
OUTPUT = ROOT / 'runs/deepseek-spora-real-dialogue-parse-audit-20261111.json'

# Selected from the local DuLeMon test archive. Labels describe whether a direct
# persistent observation should be accepted, not whether the utterance is true.
CASES = [
    {'text': '没有，我不喜欢喝咖啡，什么样的都不喜欢，我喜欢喝奶茶', 'label': 'mixed_correction', 'write': False},
    {'text': '我爱喝橙汁', 'label': 'unaddressed_affirmation', 'write': False},
    {'text': '对呀，我喜欢一边看电视，一边喝呢', 'label': 'no_value_affirmation', 'write': False},
    {'text': '原来你喜欢这样的生活，那你是不是经常喝醉？', 'label': 'question', 'write': False},
    {'text': '嗯，我在喝呢，你喜欢喝咖啡吗？', 'label': 'question', 'write': False},
    {'text': '哈哈，我喜欢喝炒奶，你喝过吗？', 'label': 'unaddressed_affirmation', 'write': False},
    {'text': '小度，你喜欢喝什么饮料啊？', 'label': 'question', 'write': False},
]


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    locator_file = LOCATOR / 'locator.pt'
    polarity_file = POLARITY / 'gate.pt'
    paths = (MODEL / 'model.safetensors', CHECKPOINT, locator_file, polarity_file, VALUE)
    source = {str(path): sha(path) for path in paths}
    if source[str(MODEL / 'model.safetensors')] != WEIGHT_SHA:
        raise ValueError('Provenance mismatch')
    locator_data = torch.load(locator_file, map_location='cpu', weights_only=True)
    polarity_data = torch.load(polarity_file, map_location='cpu', weights_only=True)
    value_data = torch.load(VALUE, map_location='cpu', weights_only=True)
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
        raise ValueError('Structural component provenance mismatch')
    vectors, masks, offsets, starts = feature_batch(base, tokenizer, CASES)
    token_vectors = torch.cat([F.layer_norm(vectors[:, i].float(), (vectors.shape[-1],)) for i in range(2)], dim=-1)
    locator = nn.Linear(locator_data['state']['weight'].shape[1], 1).eval()
    locator.load_state_dict(locator_data['state'], strict=True)
    polarity = nn.Linear(vectors.shape[-1] * 2, len(CLASSES)).eval()
    polarity.load_state_dict(polarity_data['state'], strict=True)
    value = SpanHead(vectors.shape[-1]).eval()
    value.load_state_dict(value_data['state'], strict=True)
    with torch.inference_mode():
        subject_scores = locator(token_vectors).squeeze(-1).sigmoid()
        probabilities = polarity(pool(vectors, masks)).softmax(-1)
    value_rows = decode_values(CASES, value, vectors, offsets, starts, masks, value_data['threshold'])
    rows = []
    for index, (case, score, probability, value_row) in enumerate(zip(CASES, subject_scores, probabilities, value_rows)):
        _, subject_single = __import__('scripts.deepseek_learned_name_span', fromlist=['single_span_mask']).single_span_mask(score, masks[index])
        class_index = int(probability.argmax())
        polarity_ok = class_index == 0 and float(probability[0]) >= polarity_data['threshold']
        accepted = bool(subject_single and polarity_ok and value_row['accepted'])
        rows.append({**case, 'subject_single': subject_single, 'polarity': CLASSES[class_index],
                     'polarity_ok': polarity_ok, 'value_accepted': value_row['accepted'],
                     'value': value_row['value'], 'accepted': accepted,
                     'acceptance_reason': 'all_gates' if accepted else 'guarded_reject'})
    report = {'status': 'real_dialogue_read_only_parse_audit', 'source': 'DuLeMon local test selection',
              'source_rows': [0, 1, 2, 3, 4, 5, 6], 'summary': {'examples': len(rows),
              'accepted': sum(row['accepted'] for row in rows), 'expected_writes': sum(row['write'] for row in rows),
              'polarity_affirmative': sum(row['polarity_ok'] for row in rows),
              'value_nonabstain': sum(row['value_accepted'] for row in rows)},
              'source_hashes': source, 'feature_variant_sha256': variant,
              'replacement_layers': sorted(blocks), 'rows': rows,
              'seconds': time.perf_counter() - started,
              'limitations': ['Seven selected turns, manually labeled for write eligibility.',
                              'No entity is supplied for first-person statements, so rejecting them is intentional.',
                              'The local archive selection and labels are not an independent benchmark.',
                              'No persistent memory write or response generation was performed.']}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
