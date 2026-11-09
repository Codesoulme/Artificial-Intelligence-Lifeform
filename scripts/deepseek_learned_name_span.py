"""Train a token span locator on frozen DeepSeek features; development only."""
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

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_name_span_probe import OUTPUT as PROJECTION, oracle_span_mask
from scripts.deepseek_named_binding_migrate import OUTPUT as SOURCE
from scripts.deepseek_named_binding_probe import HELD_TEMPLATES, KEYS, NAMES, TRAIN_TEMPLATES, UNKNOWN, pooled, predict
from scripts.deepseek_named_incremental_probe import NEW_KEYS
from scripts.deepseek_pooled_address import extract
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-learned-name-span-20261110'
TRAIN_NEGATIVE = ('当前的数字是多少？', '请查询现在的数值。', '告诉我记录的数字。', '目前存的数值是几？',
                  '你会说中文吗？', '请解释什么是记忆。', '今天天气怎么样？', '帮我想一个故事。')
HELD_NEGATIVE = ('现在是什么数字？', '刚才记住的内容有哪些？', '给我一个放松建议。',
                 '你可以帮我写诗吗？', '目前保存了多少？', '我想知道最新消息。')
HELD_MULTIPLE = ('小林和阿宁各记了什么数？', '小雨与阿哲的数字相同吗？', '比较小梅和阿远的值。',
                 '小岚与阿明各是多少？', '阿松和小雪现在记录的数字？', '小青和阿禾的数值是什么？')


def single_span_mask(token_scores, allowed, threshold=0.5):
    if token_scores.ndim != 1 or allowed.shape != token_scores.shape or allowed.dtype != torch.bool:
        raise ValueError('Matching token scores and boolean mask required')
    if not bool(torch.isfinite(token_scores).all()):
        raise ValueError('Nonfinite token scores')
    selected = (token_scores >= threshold) & allowed
    indices = torch.nonzero(selected).flatten()
    contiguous = bool(len(indices) and int(indices[-1] - indices[0] + 1) == len(indices))
    return selected if contiguous else torch.zeros_like(selected), contiguous


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    individual, source_report = SOURCE / 'individual.pt', SOURCE / 'result.json'
    projections, projection_report = PROJECTION / 'projections.pt', PROJECTION / 'result.json'
    protected = {p: sha(p) for p in (MODEL / 'model.safetensors', INDIVIDUAL, individual, source_report,
                                       projections, projection_report)}
    if protected[MODEL / 'model.safetensors'] != WEIGHT_SHA or protected[INDIVIDUAL] != INDIVIDUAL_SHA:
        raise ValueError('Protected model mismatch')
    with source_report.open(encoding='utf-8') as stream:
        source = json.load(stream)
    with projection_report.open(encoding='utf-8') as stream:
        prior = json.load(stream)
    config_sha = feature_files_sha256(MODEL)
    if (source['saved_individual_sha256'] != protected[individual] or source['feature_config_sha256'] != config_sha
            or prior['projection_sha256'] != protected[projections] or prior['weight_sha256'] != WEIGHT_SHA):
        raise ValueError('Source provenance mismatch')
    projection = torch.load(projections, map_location='cpu', weights_only=True)
    if (projection.get('format') != 'oracle-name-span-development-v1' or projection.get('weight_sha256') != WEIGHT_SHA
            or projection.get('individual_sha256') != INDIVIDUAL_SHA or projection.get('feature_config_sha256') != config_sha):
        raise ValueError('Projection provenance mismatch')
    threshold = prior['results']['oracle_name_span']['threshold']
    torch.manual_seed(4631)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, state, _ = load(individual, 'cpu')
    before = frozen_digest(organism, state)
    bases = torch.stack([F.normalize(site.current_basis.detach().float(), dim=-1) for site in organism.memories])
    train_single = [(name, key, template.format(name=name)) for name, key in zip(NAMES, KEYS) for template in TRAIN_TEMPLATES]
    eval_single = [(name, key, template.format(name=name)) for name, key in zip(NAMES + UNKNOWN[:2], KEYS + NEW_KEYS)
                   for template in HELD_TEMPLATES]
    eval_single += [(name, None, template.format(name=name)) for name in UNKNOWN[2:] for template in HELD_TEMPLATES]
    train_questions = [q for _, _, q in train_single] + list(TRAIN_NEGATIVE)
    eval_questions = [q for _, _, q in eval_single] + list(HELD_NEGATIVE) + list(HELD_MULTIPLE)
    if set(train_questions) & set(eval_questions):
        raise ValueError('Training/evaluation overlap')
    questions = train_questions + eval_questions
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False,
                                           add_generation_prompt=True) for q in questions]
    encoding = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False, return_offsets_mapping=True)
    indices = list(range(len(train_single))) + list(range(len(train_questions), len(train_questions) + len(eval_single)))
    oracle, crossings = oracle_span_mask(encoding['offset_mapping'][indices], encoding['attention_mask'][indices],
        [texts[i] for i in indices], [questions[i] for i in indices],
        [r[0] for r in train_single + eval_single])
    if any(crossings):
        raise ValueError('Tokenizer crosses labeled span boundary')
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    features, question_masks = extract(base, tokenizer, questions, batch_size=1)
    if features.shape[2] != oracle.shape[1]:
        raise ValueError('Tokenization mismatch')
    train_len = len(train_questions)
    train_labels = torch.zeros_like(question_masks[:train_len], dtype=torch.float32)
    train_labels[:len(train_single)] = oracle[:len(train_single)].float()
    if not bool((train_labels.bool() <= question_masks[:train_len]).all()):
        raise ValueError('Name lies outside question')
    # Each token is classified using only the frozen two-layer representation.
    token_vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],))
                               for i in range(2)], dim=-1)
    train_x = token_vectors[:train_len][question_masks[:train_len]]
    train_y = train_labels[question_masks[:train_len]]
    locator = nn.Linear(train_x.shape[-1], 1)
    optimizer = torch.optim.AdamW(locator.parameters(), lr=0.001, weight_decay=0.01)
    positive_weight = (train_y.numel() - train_y.sum()) / train_y.sum()
    log = []
    for step in range(600):
        logits = locator(train_x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, train_y, pos_weight=positive_weight)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite locator loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in (0, 199, 399, 599):
            log.append({'step': step + 1, 'loss': float(loss.detach())})
    locator.eval()
    rows = []
    with torch.inference_mode():
        all_scores = locator(token_vectors[train_len:]).squeeze(-1).sigmoid()
        for i, q in enumerate(eval_questions):
            mask, single = single_span_mask(all_scores[i], question_masks[train_len + i])
            if i < len(eval_single):
                name, key, _ = eval_single[i]
                expected_mask = oracle[len(train_single) + i]
                group = 'old' if i < 16 else 'new' if i < 20 else 'unbound'
                exact = bool(torch.equal(mask, expected_mask))
            else:
                name, key, exact = None, None, None
                group = 'no_entity' if i < len(eval_single) + len(HELD_NEGATIVE) else 'multi_entity'
            row = {'question': q, 'group': group, 'expected_key': key, 'single_span': single,
                   'exact_span': exact, 'selected_token_indices': torch.nonzero(mask).flatten().tolist(),
                   'accepted': False}
            if single:
                vectors = pooled(features[train_len+i:train_len+i+1], mask.unsqueeze(0))
                predicted, sims = predict(vectors, projection['weights']['oracle_name_span'], bases)
                row['predicted'] = predicted[0].tolist()
                row['similarities'] = sims[0].tolist()
                row['accepted'] = bool((sims[0] >= threshold).all() and predicted[0, 0] == predicted[0, 1])
            row['address_correct'] = (key is not None and row['accepted'] and row.get('predicted') == [key, key])
            rows.append(row)
    if (frozen_digest(organism, state) != before or feature_files_sha256(MODEL) != config_sha
            or any(sha(p) != d for p, d in protected.items())):
        raise ValueError('Protected input changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    snapshot = OUTPUT / 'locator.pt'
    torch.save({'format': 'deepseek-name-span-locator-development-v1', 'state': locator.state_dict(),
                'feature_width': features.shape[-1], 'weight_sha256': WEIGHT_SHA,
                'feature_config_sha256': config_sha, 'layers': [6, 18], 'threshold': 0.5}, snapshot)
    summary = {}
    for group in ('old', 'new', 'unbound', 'no_entity', 'multi_entity'):
        subset = [r for r in rows if r['group'] == group]
        summary[group] = {'examples': len(subset), 'single_span': sum(r['single_span'] for r in subset),
                          'exact_span': sum(r['exact_span'] is True for r in subset),
                          'accepted': sum(r['accepted'] for r in subset),
                          'address_correct': sum(r['address_correct'] for r in subset)}
    report = {'status': 'learned_span_locator_development_only', 'weight_sha256': WEIGHT_SHA,
              'individual_sha256': INDIVIDUAL_SHA, 'projection_sha256': protected[projections],
              'locator_sha256': sha(snapshot), 'training_names': len(NAMES),
              'training_positive_questions': len(train_single), 'training_negative_questions': len(TRAIN_NEGATIVE),
              'training_positive_tokens': int(train_y.sum()), 'training_question_tokens': len(train_y),
              'log': log, 'summary': summary, 'rows': rows, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Span classifier trained on eight names and a few hand-authored templates.',
                              'Locator evaluation includes four names not seen by the locator; two were supervised for the binding projection.',
                              'All names, projections and most templates appeared in prior development, not independent final evaluation.',
                              'Multiple disconnected predicted spans are deferred; one missed name may still be misrouted.',
                              'No context resolution, aliases, same-name disambiguation or natural-language fact storage.',
                              'No end-to-end response generation, write routing, persistence, or crash recovery evaluated.',
                              'Token and address thresholds are uncalibrated and are not probabilities of correctness.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
