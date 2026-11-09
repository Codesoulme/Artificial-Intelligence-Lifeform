"""Read-only cross-domain audit of the experimental person-name locator."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import pyarrow.parquet as parquet
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_learned_name_span import OUTPUT as LOCATOR, single_span_mask
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_named_binding_probe import NAMES, UNKNOWN
from scripts.deepseek_pooled_address import extract
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.named_address import feature_files_sha256

SOURCE = ROOT / 'runs/cluener-validation-41aef965.parquet'
SOURCE_SHA = 'b374c9749c6f755aaebe7a8d7c24c5c3307183fa4d8e9aa3413b4ebdd052c03c'
OUTPUT = ROOT / 'runs/deepseek-cluener-span-audit-20261110.json'
REVISION = '41aef9650901fa2169213f38d49bd6491eb9b77a'


def labeled_names(row):
    text = row['text']
    if not isinstance(text, str) or not isinstance(row['entities'], list):
        raise ValueError('Invalid dataset row')
    spans = []
    for entity in row['entities']:
        if entity['label'] != 'name':
            continue
        start, end, value = entity['start_offset'], entity['end_offset'], entity['entity']
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                or text[start:end] != value):
            raise ValueError('Invalid source name offsets')
        spans.append((start, end))
    return sorted(set(spans))


def select_rows(source):
    groups = {'single': [], 'multiple': [], 'none': []}
    seen = set()
    for index, row in enumerate(source):
        text = row['text']
        if not isinstance(text, str) or not 8 <= len(text) <= 120 or any(name in text for name in NAMES + UNKNOWN):
            continue
        spans = labeled_names(row)
        if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
            continue
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        group = 'none' if not spans else 'single' if len(spans) == 1 else 'multiple'
        groups[group].append({'source_index': index, 'text': text, 'spans': spans, 'text_sha256': digest})
    selected = []
    for group, count in (('single', 20), ('multiple', 10), ('none', 10)):
        candidates = sorted(groups[group], key=lambda r: r['text_sha256'])
        if len(candidates) < count:
            raise ValueError('Insufficient source examples')
        selected.extend(dict(row, group=group) for row in candidates[:count])
    return selected, {key: len(value) for key, value in groups.items()}


def character_mask(offsets, selected, text_start, text_length):
    if offsets.ndim != 2 or offsets.shape != (len(selected), 2) or selected.dtype != torch.bool:
        raise ValueError('Invalid token offsets')
    result = torch.zeros(text_length, dtype=torch.bool)
    for (start, end), use in zip(offsets.tolist(), selected.tolist()):
        if use:
            lo, hi = max(start - text_start, 0), min(end - text_start, text_length)
            if hi > lo:
                result[lo:hi] = True
    return result


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    locator_file, locator_report = LOCATOR / 'locator.pt', LOCATOR / 'result.json'
    protected = {p: sha(p) for p in (MODEL / 'model.safetensors', INDIVIDUAL, SOURCE, locator_file, locator_report)}
    if (protected[MODEL / 'model.safetensors'] != WEIGHT_SHA or protected[INDIVIDUAL] != INDIVIDUAL_SHA
            or protected[SOURCE] != SOURCE_SHA):
        raise ValueError('Protected source mismatch')
    with locator_report.open(encoding='utf-8') as stream:
        prior = json.load(stream)
    locator_data = torch.load(locator_file, map_location='cpu', weights_only=True)
    config_sha = feature_files_sha256(MODEL)
    if (prior['locator_sha256'] != protected[locator_file]
            or locator_data.get('format') != 'deepseek-name-span-locator-development-v1'
            or locator_data.get('weight_sha256') != WEIGHT_SHA or locator_data.get('feature_config_sha256') != config_sha
            or locator_data.get('layers') != [6, 18] or locator_data.get('threshold') != 0.5):
        raise ValueError('Locator provenance mismatch')
    source = parquet.read_table(SOURCE).to_pylist()
    selected, population = select_rows(source)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    questions = [row['text'] for row in selected]
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False,
                                           add_generation_prompt=True) for q in questions]
    encoded = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False, return_offsets_mapping=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    features, question_masks = extract(base, tokenizer, questions, batch_size=1)
    if features.shape[2] != encoded['offset_mapping'].shape[1] or features.shape[-1] != locator_data['feature_width']:
        raise ValueError('Feature/tokenizer mismatch')
    locator = nn.Linear(features.shape[-1] * 2, 1).eval()
    locator.load_state_dict(locator_data['state'], strict=True)
    vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
    with torch.inference_mode():
        scores = locator(vectors).squeeze(-1).sigmoid()
    rows = []
    for i, item in enumerate(selected):
        raw = (scores[i] >= 0.5) & question_masks[i]
        single, single_ok = single_span_mask(scores[i], question_masks[i])
        start = texts[i].find(item['text'])
        if start < 0:
            raise ValueError('Source text missing from formatted prompt')
        predicted = character_mask(encoded['offset_mapping'][i], raw, start, len(item['text']))
        gold = torch.zeros(len(item['text']), dtype=torch.bool)
        for left, right in item['spans']:
            gold[left:right] = True
        representable = torch.zeros_like(gold)
        for left, right in item['spans']:
            offsets = encoded['offset_mapping'][i]
            selected_gold = (offsets[:, 0] < start + right) & (offsets[:, 1] > start + left) & question_masks[i]
            representable |= character_mask(offsets, selected_gold, start, len(item['text']))
        rows.append(dict(item, predicted_spans_indices=torch.nonzero(raw).flatten().tolist(),
                         predicted_characters=int(predicted.sum()), gold_characters=int(gold.sum()),
                         true_positive_characters=int((predicted & gold).sum()),
                         exact=bool(torch.equal(predicted, gold)), gold_token_representable=bool(torch.equal(representable, gold)),
                         single_contiguous=single_ok, single_selected_tokens=int(single.sum())))
    if feature_files_sha256(MODEL) != config_sha or any(sha(p) != d for p, d in protected.items()):
        raise ValueError('Protected input changed')
    summary = {}
    for group in ('single', 'multiple', 'none'):
        subset = [row for row in rows if row['group'] == group]
        tp = sum(row['true_positive_characters'] for row in subset)
        predicted = sum(row['predicted_characters'] for row in subset)
        gold = sum(row['gold_characters'] for row in subset)
        summary[group] = {'examples': len(subset), 'exact': sum(row['exact'] for row in subset),
                          'gold_token_representable': sum(row['gold_token_representable'] for row in subset),
                          'single_contiguous': sum(row['single_contiguous'] for row in subset),
                          'predicted_characters': predicted, 'gold_characters': gold, 'true_positive_characters': tp,
                          'character_precision': tp / predicted if predicted else None,
                          'character_recall': tp / gold if gold else None}
    report = {'status': 'cluener_cross_domain_development_audit', 'dataset': 'xusenlin/clue-ner', 'revision': REVISION,
              'source_sha256': SOURCE_SHA, 'source_population': population, 'selection': 'sha256-sorted unique text per group',
              'locator_sha256': protected[locator_file], 'weight_sha256': WEIGHT_SHA,
              'summary': summary, 'rows': rows, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['CLUENER validation split is news, not chat; sampled 20 single, 10 multiple and 10 no-name rows.',
                              'Dataset annotations may be incomplete; labels and exact character boundaries may contain errors.',
                              'Tokenizer can merge name and surrounding characters; representability is reported separately.',
                              'The public benchmark or related text may have appeared in pretrained model data.',
                              'No training, memory reading, response generation or open-world rejection evaluated.',
                              'Single contiguous span is a heuristic; it does not prove one person is present.',
                              'Mirror declares Apache-2.0; underlying news text rights are not independently verified.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
