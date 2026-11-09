"""Continue a frozen-base name locator with a small separate CLUENER training sample."""
import copy
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

from scripts.deepseek_cluener_span_audit import SOURCE as VALIDATION, SOURCE_SHA as VALIDATION_SHA, character_mask, select_rows
from scripts.deepseek_learned_name_span import (HELD_MULTIPLE, HELD_NEGATIVE, OUTPUT as LOCATOR, TRAIN_NEGATIVE,
                                                single_span_mask)
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_name_span_probe import oracle_span_mask
from scripts.deepseek_named_binding_probe import HELD_TEMPLATES, KEYS, NAMES, TRAIN_TEMPLATES, UNKNOWN
from scripts.deepseek_named_incremental_probe import NEW_KEYS
from scripts.deepseek_pooled_address import extract
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.named_address import feature_files_sha256

TRAIN = ROOT / 'runs/cluener-train-41aef965.parquet'
TRAIN_SHA = '29fa263e67cf4a27fa4a9e378cdba9bb5189a3e6d9002829cdc72e2976cfe1cd'
AUDIT = ROOT / 'runs/deepseek-cluener-span-audit-20261110.json'
OUTPUT = ROOT / 'runs/deepseek-cluener-locator-20261110'


def remove_validation_texts(training, validation):
    held = {row['text'] for row in validation if isinstance(row.get('text'), str)}
    return [row for row in training if row.get('text') not in held]


def token_gold(offsets, allowed, text_start, text_length, spans):
    gold_chars = torch.zeros(text_length, dtype=torch.bool)
    selected = torch.zeros(len(allowed), dtype=torch.bool)
    for start, end in spans:
        gold_chars[start:end] = True
        selected |= (offsets[:, 0] < text_start + end) & (offsets[:, 1] > text_start + start) & allowed
    return selected, torch.equal(character_mask(offsets, selected, text_start, text_length), gold_chars), gold_chars


def assess_news(rows, scores, offsets, masks, texts):
    results = []
    for i, row in enumerate(rows):
        start = texts[i].find(row['text'])
        gold_tokens, representable, gold = token_gold(offsets[i], masks[i], start, len(row['text']), row['spans'])
        predicted_tokens = (scores[i] >= 0.5) & masks[i]
        predicted = character_mask(offsets[i], predicted_tokens, start, len(row['text']))
        _, single = single_span_mask(scores[i], masks[i])
        results.append({'group': row['group'], 'text_sha256': row['text_sha256'],
                        'exact': bool(torch.equal(predicted, gold)), 'representable': representable,
                        'single_contiguous': single, 'predicted_characters': int(predicted.sum()),
                        'gold_characters': int(gold.sum()), 'true_positive_characters': int((predicted & gold).sum()),
                        'gold_token_count': int(gold_tokens.sum())})
    groups = {}
    for group in ('single', 'multiple', 'none'):
        subset = [r for r in results if r['group'] == group]
        tp = sum(r['true_positive_characters'] for r in subset)
        predicted = sum(r['predicted_characters'] for r in subset)
        gold = sum(r['gold_characters'] for r in subset)
        groups[group] = {'examples': len(subset), 'exact': sum(r['exact'] for r in subset),
                         'representable': sum(r['representable'] for r in subset),
                         'single_contiguous': sum(r['single_contiguous'] for r in subset),
                         'character_precision': tp / predicted if predicted else None,
                         'character_recall': tp / gold if gold else None}
    return {'groups': groups, 'rows': results}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    locator_file, locator_report = LOCATOR / 'locator.pt', LOCATOR / 'result.json'
    protected = {p: sha(p) for p in (MODEL / 'model.safetensors', INDIVIDUAL, TRAIN, VALIDATION, AUDIT,
                                       locator_file, locator_report)}
    if (protected[MODEL / 'model.safetensors'] != WEIGHT_SHA or protected[INDIVIDUAL] != INDIVIDUAL_SHA
            or protected[TRAIN] != TRAIN_SHA or protected[VALIDATION] != VALIDATION_SHA):
        raise ValueError('Protected source mismatch')
    with locator_report.open(encoding='utf-8') as stream:
        prior_locator = json.load(stream)
    with AUDIT.open(encoding='utf-8') as stream:
        prior_audit = json.load(stream)
    old_data = torch.load(locator_file, map_location='cpu', weights_only=True)
    config_sha = feature_files_sha256(MODEL)
    if (prior_locator['locator_sha256'] != protected[locator_file] or prior_audit['locator_sha256'] != protected[locator_file]
            or old_data.get('format') != 'deepseek-name-span-locator-development-v1'
            or old_data.get('weight_sha256') != WEIGHT_SHA or old_data.get('feature_config_sha256') != config_sha
            or old_data.get('layers') != [6, 18] or old_data.get('threshold') != 0.5):
        raise ValueError('Locator provenance mismatch')
    validation = parquet.read_table(VALIDATION).to_pylist()
    training = remove_validation_texts(parquet.read_table(TRAIN).to_pylist(), validation)
    news_train, population = select_rows(training)
    news_eval, _ = select_rows(validation)
    if [r['text_sha256'] for r in news_eval] != [r['text_sha256'] for r in prior_audit['rows']]:
        raise ValueError('Validation selection changed')
    train_single = [(name, template.format(name=name)) for name in NAMES for template in TRAIN_TEMPLATES]
    eval_single = [(name, template.format(name=name)) for name in NAMES + UNKNOWN for template in HELD_TEMPLATES]
    questions = ([r['text'] for r in news_train] + [q for _, q in train_single] + list(TRAIN_NEGATIVE)
                 + [r['text'] for r in news_eval] + [q for _, q in eval_single] + list(HELD_NEGATIVE) + list(HELD_MULTIPLE))
    news_count = len(news_train)
    synth_start = news_count
    train_count = news_count + len(train_single) + len(TRAIN_NEGATIVE)
    eval_news_start = train_count
    eval_single_start = eval_news_start + len(news_eval)
    if set(questions[:train_count]) & set(questions[train_count:]):
        raise ValueError('Train/validation overlap')
    torch.manual_seed(4632)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False,
                                           add_generation_prompt=True) for q in questions]
    encoded = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False, return_offsets_mapping=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    features, masks = extract(base, tokenizer, questions, batch_size=1)
    offsets = encoded['offset_mapping']
    if features.shape[2] != offsets.shape[1] or features.shape[-1] != old_data['feature_width']:
        raise ValueError('Feature/tokenizer mismatch')
    labels = torch.zeros_like(masks[:train_count], dtype=torch.float32)
    usable = torch.ones(train_count, dtype=torch.bool)
    for i, row in enumerate(news_train):
        start = texts[i].find(row['text'])
        tokens, representable, _ = token_gold(offsets[i], masks[i], start, len(row['text']), row['spans'])
        if not representable:
            usable[i] = False
        labels[i] = tokens.float()
    synthetic_indices = list(range(synth_start, synth_start + len(train_single)))
    oracle_train, crossings = oracle_span_mask(offsets[synthetic_indices], encoded['attention_mask'][synthetic_indices],
        [texts[i] for i in synthetic_indices], [questions[i] for i in synthetic_indices], [name for name, _ in train_single])
    if any(crossings):
        raise ValueError('Synthetic training span not representable')
    labels[synthetic_indices] = oracle_train.float()
    eval_indices = list(range(eval_single_start, eval_single_start + len(eval_single)))
    oracle_eval, _ = oracle_span_mask(offsets[eval_indices], encoded['attention_mask'][eval_indices],
        [texts[i] for i in eval_indices], [questions[i] for i in eval_indices], [name for name, _ in eval_single])
    vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
    train_mask = masks[:train_count] & usable[:, None]
    train_x, train_y = vectors[:train_count][train_mask], labels[train_mask]
    old = nn.Linear(train_x.shape[-1], 1).eval()
    old.load_state_dict(old_data['state'], strict=True)
    new = copy.deepcopy(old).train()
    optimizer = torch.optim.AdamW(new.parameters(), lr=0.0001, weight_decay=0.01)
    pos_weight = (train_y.numel() - train_y.sum()) / train_y.sum()
    log = []
    for step in range(400):
        loss = F.binary_cross_entropy_with_logits(new(train_x).squeeze(-1), train_y, pos_weight=pos_weight)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite locator loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in (0, 99, 199, 399):
            log.append({'step': step + 1, 'loss': float(loss.detach())})
    new.eval()
    results = {}
    with torch.inference_mode():
        for label, locator in (('old', old), ('new', new)):
            scores = locator(vectors[train_count:]).squeeze(-1).sigmoid()
            news = assess_news(news_eval, scores[:len(news_eval)], offsets[eval_news_start:eval_single_start],
                               masks[eval_news_start:eval_single_start], texts[eval_news_start:eval_single_start])
            synthetic = {'single': {'examples': len(eval_single), 'exact': 0},
                         'no_entity': {'examples': len(HELD_NEGATIVE), 'single_contiguous': 0},
                         'multiple': {'examples': len(HELD_MULTIPLE), 'single_contiguous': 0}}
            base_index = len(news_eval)
            for i in range(len(eval_single)):
                mask, _ = single_span_mask(scores[base_index + i], masks[eval_single_start + i])
                synthetic['single']['exact'] += int(torch.equal(mask, oracle_eval[i]))
            for group, start, count in (('no_entity', base_index + len(eval_single), len(HELD_NEGATIVE)),
                                        ('multiple', base_index + len(eval_single) + len(HELD_NEGATIVE), len(HELD_MULTIPLE))):
                for i in range(start, start + count):
                    _, single = single_span_mask(scores[i], masks[train_count + i])
                    synthetic[group]['single_contiguous'] += int(single)
            results[label] = {'news': news, 'synthetic': synthetic}
    for group in ('single', 'multiple', 'none'):
        if results['old']['news']['groups'][group] != prior_audit['summary'][group]:
            # Prior uses a different key for representability; compare the shared metrics below.
            before = results['old']['news']['groups'][group]
            previous = prior_audit['summary'][group]
            if any(before[k] != previous[k] for k in ('examples', 'exact', 'single_contiguous', 'character_precision', 'character_recall')):
                raise ValueError('Baseline did not reproduce previous audit')
    if feature_files_sha256(MODEL) != config_sha or any(sha(p) != d for p, d in protected.items()):
        raise ValueError('Protected input changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    snapshot = OUTPUT / 'locator.pt'
    torch.save({'format': 'deepseek-name-span-locator-development-v1', 'state': new.state_dict(),
                'feature_width': features.shape[-1], 'weight_sha256': WEIGHT_SHA,
                'feature_config_sha256': config_sha, 'layers': [6, 18], 'threshold': 0.5,
                'parent_sha256': protected[locator_file]}, snapshot)
    report = {'status': 'cluener_locator_development_only', 'dataset': 'xusenlin/clue-ner',
              'train_sha256': TRAIN_SHA, 'validation_sha256': VALIDATION_SHA,
              'parent_locator_sha256': protected[locator_file], 'locator_sha256': sha(snapshot),
              'train_news_selected': len(news_train), 'train_news_usable': int(usable[:news_count].sum()),
              'train_news_population_after_validation_dedup': population,
              'train_synthetic': train_count - news_count, 'training_question_tokens': len(train_y),
              'training_positive_tokens': int(train_y.sum()), 'log': log, 'results': results,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Forty selected news examples and forty synthetic examples; token-boundary failures are excluded from training.',
                              'Validation set was used for an earlier audit, so this is development evaluation, not an untouched final test.',
                              'Exact text overlap with the whole validation split was removed from training; paraphrase/source overlap is unknown.',
                              'News benchmark is not a memory-chat distribution; public data may have appeared in base pretraining.',
                              'Synthetic examples cover few names and templates; no end-to-end address or language answer measured.',
                              'Multi-entity contiguous output is not a reliable proxy for safe memory routing.',
                              'Mirror declares Apache-2.0; underlying news text rights have not been independently verified.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({label: {'news': r['news']['groups'], 'synthetic': r['synthetic']}
                      for label, r in results.items()}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
