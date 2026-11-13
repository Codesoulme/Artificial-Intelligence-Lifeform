"""Train an isolated name router on original language-layer representations."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch import nn
from torch.nn import functional as F
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_learned_name_span import single_span_mask
from scripts.deepseek_name_span_probe import oracle_span_mask
from scripts.deepseek_named_binding_probe import KEYS, NAMES, pooled, fit_binding, predict
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_pre_read_guard_audit import RecordingGuard
from spora.identity_fingerprints import IdentityFingerprints
from spora.named_address import address_basis_sha256, feature_files_sha256

OUT = ROOT / 'runs/deepseek-spora-original-layer-name-router-v3-20261114'
SEED = 47092
TRAIN = ('{name}现在爱喝什么饮料？', '请查询{name}喜欢的饮品。',
         '记录中{name}最喜欢喝什么？', '{name}的饮料偏好是什么？')
HELD = ('你记得{name}喜欢喝哪种饮品吗？', '从记录看，{name}常喝什么饮料？')
UNKNOWN = ('小雪', '阿禾', '小青', '小王')
NEAR = ('小林华', '阿宁悦', '小雨晴', '阿哲轩', '小梅子', '阿远航', '小岚宁', '阿明宇')
TRAIN_NEGATIVE = ('请写一句鼓励的话。', '我现在想喝饮料。', '记忆中保存了什么？',
                  '今天应该喝水吗？', '小林和阿宁喜欢什么？', '小雨与阿哲的饮品是什么？')
HELD_NEGATIVE = ('讲一个关于饮料的故事。', '我想知道现在几点。',
                 '小梅和阿远分别喜欢什么饮料？', '阿明和小岚爱喝什么？')
LOCATOR_ONLY = ()


def locator_only_rows(examples):
    rows = []
    for item in examples:
        if (not isinstance(item, (tuple, list)) or len(item) != 2
                or not all(isinstance(x, str) and x for x in item)):
            raise ValueError('Expected a nonempty name and question')
        name, text = item
        if name in NAMES or text.count(name) != 1:
            raise ValueError('Locator-only examples require one unbound name')
        rows.append({'group': 'train_locator_only', 'name': name, 'key': None, 'text': text})
    return rows


def unpad_selection(selected, single_length):
    if (not isinstance(selected, torch.Tensor) or selected.ndim != 1 or selected.dtype != torch.bool
            or type(single_length) is not int or not 0 < single_length <= len(selected)
            or bool(selected[single_length:].any())):
        raise ValueError('Selected span extends into padding or has invalid shape')
    return selected[:single_length]


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Base checksum mismatch')
    training = [{'group': 'train_known', 'name': name, 'key': key, 'text': t.format(name=name)}
                for name, key in zip(NAMES, KEYS) for t in TRAIN]
    training += locator_only_rows(LOCATOR_ONLY)
    training += [{'group': 'train_negative', 'name': None, 'key': None, 'text': text}
                 for text in TRAIN_NEGATIVE]
    held = [{'group': 'known', 'name': name, 'key': key, 'text': t.format(name=name)}
            for name, key in zip(NAMES, KEYS) for t in HELD]
    held += [{'group': 'near_name', 'name': name, 'key': None, 'text': HELD[0].format(name=name)}
             for name in NEAR]
    held += [{'group': 'unknown', 'name': name, 'key': None, 'text': HELD[1].format(name=name)}
             for name in UNKNOWN]
    held += [{'group': 'no_or_multiple', 'name': None, 'key': None, 'text': text}
             for text in HELD_NEGATIVE]
    rows = training + held
    if len({r['text'] for r in rows}) != len(rows):
        raise ValueError('Overlapping examples')
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='spora-original-name-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        base = p.load_full_streamed().eval()
        originals = {i: base.model.layers[i] for i in (6, 12)}
        chat = p.runtime.load_chat(base_loader=lambda: base)
        source_digest = p.frozen_digest(chat.organism, chat.state)
        replacements = {i: base.model.layers[i] for i in originals}
        for index in originals:
            base.model.layers[index] = originals[index]
        tokenizer = chat.tokenizer
        tokenizer.padding_side = 'right'
        def label_rows(examples):
            texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': row['text']}],
                     tokenize=False, add_generation_prompt=True) for row in examples]
            encoding = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False,
                                 return_offsets_mapping=True)
            positives = [i for i, row in enumerate(examples) if row['name'] is not None]
            spans, crossings = oracle_span_mask(encoding['offset_mapping'][positives],
                                                encoding['attention_mask'][positives],
                                                [texts[i] for i in positives],
                                                [examples[i]['text'] for i in positives],
                                                [examples[i]['name'] for i in positives])
            truth = torch.zeros_like(encoding['attention_mask'], dtype=torch.bool)
            truth[positives] = spans
            return truth, {positives[i] for i, crossed in enumerate(crossings) if crossed}

        truth, unrepresentable = label_rows(rows)
        dropped_train = [training[i]['text'] for i in sorted(unrepresentable) if i < len(training)]
        training = [row for i, row in enumerate(training) if i not in unrepresentable]
        rows = training + held
        truth, unrepresentable = label_rows(rows)
        if any(i < len(training) for i in unrepresentable):
            raise ValueError('Unrepresentable training name span remains')
        if {row['key'] for row in training if row['key'] is not None} != set(KEYS):
            raise ValueError('Dropping spans removed a registered training identity')
        features, allowed = extract(base, tokenizer, [r['text'] for r in rows], batch_size=1)
        if truth.shape != allowed.shape or bool((truth & ~allowed).any()):
            raise ValueError('Feature mask mismatch')
        vector = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
        n_train = len(training)
        x, y = vector[:n_train][allowed[:n_train]], truth[:n_train][allowed[:n_train]].float()
        locator = nn.Linear(x.shape[-1], 1)
        optimizer = torch.optim.AdamW(locator.parameters(), lr=0.0005, weight_decay=0.01)
        positive_weight = (y.numel() - y.sum()) / y.sum()
        for _ in range(400):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(locator(x).squeeze(-1), y, pos_weight=positive_weight)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite locator loss')
            loss.backward()
            optimizer.step()
        locator.eval()
        bases = torch.stack([F.normalize(site.current_basis.detach().cpu().float(), dim=-1)
                             for site in chat.organism.memories])
        train_indices = [i for i, row in enumerate(training) if row['key'] is not None]
        vectors = pooled(features[train_indices], truth[train_indices])
        target = torch.stack([bases[:, training[i]['key']] for i in train_indices])
        projection = fit_binding(vectors, target)
        _, train_scores = predict(vectors, projection, bases)
        threshold = float(train_scores.min()) - 1e-6
        if not 0 <= threshold <= 1:
            raise ValueError('Invalid address threshold')
        fingerprints = IdentityFingerprints(chat.organism.config.key_count).to(next(chat.organism.parameters()).device)
        for name, key in zip(NAMES, KEYS):
            fingerprints.bind(key, name)
        guard = RecordingGuard(fingerprints, tokenizer)
        with torch.inference_mode():
            token_scores = locator(vector[n_train:]).squeeze(-1).sigmoid()
        results = []
        for j, row in enumerate(held):
            selected, single = single_span_mask(token_scores[j], allowed[n_train + j], 0.5)
            result = {'group': row['group'], 'question': row['text'], 'expected_key': row['key'],
                      'span_representable': n_train+j not in unrepresentable if row['name'] else None,
                      'span_exact': bool(torch.equal(selected, truth[n_train+j])) if row['name'] and n_train+j not in unrepresentable else None,
                      'single_span': single, 'accepted': False, 'guarded_accepted': False}
            if single:
                span_vector = pooled(features[n_train+j:n_train+j+1], selected.unsqueeze(0))
                indices, scores = predict(span_vector, projection, bases)
                result['indices'], result['similarities'] = indices[0].tolist(), scores[0].tolist()
                result['accepted'] = bool(indices[0, 0] == indices[0, 1] and (scores[0] >= threshold).all())
                if result['accepted']:
                    single_text = tokenizer.apply_chat_template([{'role': 'user', 'content': row['text']}],
                                                               tokenize=False, add_generation_prompt=True)
                    single_length = len(tokenizer(single_text, add_special_tokens=False)['input_ids'])
                    result['guarded_accepted'] = bool(guard.authorize(
                        row['text'], unpad_selection(selected, single_length), int(indices[0, 0])))
                    result['guard_span'] = guard.last
                    if result['span_exact'] is True and guard.last['selected_text'] != row['name']:
                        raise ValueError('Batch and single-question token offsets disagree')
            result['correct_key'] = bool(row['key'] is not None and result['accepted'] and result.get('indices') == [row['key']] * 2)
            result['guarded_correct_key'] = bool(result['correct_key'] and result['guarded_accepted'])
            results.append(result)
        for index in originals:
            base.model.layers[index] = replacements[index]
        if p.frozen_digest(chat.organism, chat.state) != source_digest:
            raise ValueError('Protected individual changed')
        basis_sha = address_basis_sha256(chat.organism)
        locator_state = {key: value.detach().cpu().clone() for key, value in locator.state_dict().items()}
    if p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(p.runtime.CHECKPOINT) != checkpoint_sha:
        raise ValueError('Protected files changed')
    summary = {}
    for group in sorted({r['group'] for r in results}):
        subset = [r for r in results if r['group'] == group]
        summary[group] = {'cases': len(subset), 'representable': sum(r['span_representable'] is True for r in subset),
                          'exact_span': sum(r['span_exact'] is True for r in subset),
                          'accepted': sum(r['accepted'] for r in subset), 'correct_key': sum(r['correct_key'] for r in subset),
                          'guarded_accepted': sum(r['guarded_accepted'] for r in subset),
                          'guarded_correct_key': sum(r['guarded_correct_key'] for r in subset)}
    OUT.mkdir()
    path = OUT / 'router.pt'
    torch.save({'format': 'spora-original-layer-name-router-research-v1', 'base_sha256': base_sha,
                'checkpoint_sha256': checkpoint_sha, 'feature_config_sha256': feature_files_sha256(p.MODEL),
                'address_basis_sha256': basis_sha, 'hidden_indices': [6, 18], 'locator_threshold': 0.5,
                'address_threshold': threshold, 'locator_state': locator_state, 'projection': projection.cpu()}, path)
    report = {'summary': summary, 'rows': results, 'train_count': len(training), 'dropped_train': dropped_train,
              'locator_only_train_count': sum(row['group'] == 'train_locator_only' for row in training),
              'threshold': threshold,
              'loss': float(loss.detach()), 'unrepresentable_held': len([i for i in unrepresentable if i >= n_train]),
              'base_sha256': base_sha, 'checkpoint_sha256': checkpoint_sha, 'basis_sha256': basis_sha,
              'router_sha256': p.sha(path), 'seed': SEED, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Manual name enrollment and numeric address binding, not autonomous identity learning.',
                              'Small authored templates and known names with possible prior overlap.',
                              'Fingerprint guards extracted spans only; shared truncation or homonyms can cause false acceptance.',
                              'Address threshold from training examples, not independent calibration.',
                              'No read intent, temporal classification, final answer or persistence tested.',
                              'Experimental checkpoint is not compatible with the default replacement-bound router.']}
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
