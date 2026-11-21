"""Train an isolated name router on original language-layer representations."""
import argparse
import hashlib
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
from spora.integrated_checkpoint import load_integrated
from spora.named_address import address_basis_sha256, feature_files_sha256
from transformers import AutoTokenizer

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


class SequenceLocator(nn.Module):
    """Locate names from the complete user question before a memory lookup."""

    def __init__(self, width):
        super().__init__()
        self.project = nn.Linear(width, 48)
        self.context = nn.GRU(48, 32, batch_first=True, bidirectional=True)
        self.classify = nn.Linear(64, 1)

    def forward(self, tokens):
        hidden, _ = self.context(F.gelu(self.project(tokens)))
        return self.classify(hidden)


class SpanLocator(nn.Module):
    """Compare complete name spans and an explicit no-single-name class."""

    def __init__(self, width):
        super().__init__()
        self.project = nn.Linear(width, 48)
        self.context = nn.GRU(48, 32, batch_first=True, bidirectional=True)
        self.start = nn.Linear(64, 17)
        self.end = nn.Linear(64, 17)
        self.reject = nn.Linear(64, 1)

    def forward(self, tokens, allowed):
        hidden, _ = self.context(F.gelu(self.project(tokens)))
        starts = self.start(hidden)
        ends = self.end(hidden)
        pairs = (starts[..., :16] @ ends[..., :16].transpose(-2, -1)) / 4.
        pairs = pairs + starts[..., 16, None] + ends[..., 16].unsqueeze(-2)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        length = positions[None, :] - positions[:, None]
        valid = (length >= 0) & (length < 8) & allowed.unsqueeze(-1) & allowed.unsqueeze(-2)
        pairs = pairs.masked_fill(~valid, -torch.inf).flatten(1)
        pooled = (hidden * allowed[..., None]).sum(1) / allowed.sum(1, keepdim=True).clamp_min(1)
        return torch.cat((pairs, self.reject(pooled)), dim=1)


def span_targets(truth):
    width = truth.shape[1]
    first = truth.to(torch.int64).argmax(dim=1)
    last = width - 1 - truth.flip(1).to(torch.int64).argmax(dim=1)
    return torch.where(truth.any(1), first * width + last, width * width)


def choose_span(probabilities, allowed, threshold):
    if probabilities.numel() == allowed.numel():
        return single_span_mask(probabilities, allowed, threshold)
    width = len(allowed)
    pairs = probabilities[:-1]
    index = int(pairs.argmax())
    start, end = divmod(index, width)
    selected = torch.zeros_like(allowed)
    single = bool((pairs[index] - probabilities[-1]).sigmoid() >= threshold)
    if single:
        selected[start:end + 1] = True
        if not bool(allowed[selected].all()):
            return torch.zeros_like(allowed), False
    return selected, single


def lookahead(tokens):
    """Concatenate the current and next two normalized token features."""
    normalized = F.layer_norm(tokens.float(), (tokens.shape[-1],))
    padded = F.pad(normalized, (0, 0, 0, 2))
    return torch.cat((padded[:-2], padded[1:-1], padded[2:]), dim=-1)


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


def annotated_name_rows():
    from pyarrow import parquet
    from scripts.deepseek_cluener_locator_train import TRAIN, TRAIN_SHA, VALIDATION, VALIDATION_SHA, remove_validation_texts
    from scripts.deepseek_cluener_span_audit import labeled_names

    sources = {TRAIN: TRAIN_SHA, VALIDATION: VALIDATION_SHA}
    if any(p.sha(path) != digest for path, digest in sources.items()):
        raise ValueError('Unexpected CLUENER source digest')
    source = remove_validation_texts(parquet.read_table(TRAIN).to_pylist(), parquet.read_table(VALIDATION).to_pylist())
    unique = {item['text']: item for item in source if 8 <= len(item['text']) <= 48}
    groups = {'single': [], 'none': [], 'multiple': []}
    for text in sorted(unique, key=lambda value: hashlib.sha256(value.encode()).digest()):
        spans = labeled_names(unique[text])
        name = None
        if len(spans) == 1:
            start, end = spans[0]
            name = text[start:end]
            if name in NAMES or text.count(name) != 1:
                continue
        group = 'single' if name is not None else 'multiple' if spans else 'none'
        groups[group].append({'group': 'cluener_' + group, 'name': name, 'key': None, 'text': text})
    return [row for group, limit in (('single', 96), ('none', 32), ('multiple', 24))
            for row in groups[group][:limit]], sources


def unpad_selection(selected, single_length):
    if (not isinstance(selected, torch.Tensor) or selected.ndim != 1 or selected.dtype != torch.bool
            or type(single_length) is not int or not 0 < single_length <= len(selected)
            or bool(selected[single_length:].any())):
        raise ValueError('Selected span extends into padding or has invalid shape')
    return selected[:single_length]


def main(*, checkpoint=None, output=None, include_following_tokens=False,
         diverse_boundaries=False, fresh_boundary_holdout=False, diverse_query_context=False,
         sequence_locator=False, span_locator=False, contrast_query_nouns=False,
         cluener_replay=False):
    output = OUT if output is None else output
    if output.exists():
        raise FileExistsError(output)
    source = p.runtime.CHECKPOINT if checkpoint is None else checkpoint
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    checkpoint_sha = p.sha(source)
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Base checksum mismatch')
    known_templates = TRAIN
    if diverse_query_context:
        known_templates += ('根据最新的一条信息，{name}最爱的是哪款饮料？',
                            '在当前的备注里，{name}首选什么饮品？',
                            '我想核对一下最近的记录，{name}最爱喝哪一种？',
                            '翻一下旧的那条，{name}过去总喝哪种饮料？',
                            '更换偏好以前的笔记里，{name}常喝什么饮品？',
                            '我想知道更早记下的饮料，{name}当时的选择是什么？')
    if contrast_query_nouns:
        known_templates += (
            '翻开我刚写的备忘，{name}最爱喝的饮品叫什么？',
            '关于饮品的最新一条记录，{name}喜欢的是哪个？',
            '记下的饮料里，哪款是{name}的首选饮品？',
            '对于上次的饮品记录，请说出{name}当初爱喝的饮品。',
        )
    training = [{'group': 'train_known', 'name': name, 'key': key, 'text': t.format(name=name)}
                for name, key in zip(NAMES, KEYS, strict=True) for t in known_templates]
    boundary_examples = LOCATOR_ONLY
    if diverse_boundaries:
        boundary_examples += tuple(
            (name + suffix, template.format(name=name + suffix))
            for name in NAMES for suffix in ('森', '宁', '航', '扬')
            if name + suffix not in NEAR
            for template in ('能回忆起{name}想喝的饮品吗？', '记录显示{name}平时爱喝哪种饮料？')
        )
    training += locator_only_rows(boundary_examples)
    training += [{'group': 'train_negative', 'name': None, 'key': None, 'text': text}
                 for text in TRAIN_NEGATIVE]
    extra_sources = {}
    if cluener_replay:
        news, extra_sources = annotated_name_rows()
        training += news
    if contrast_query_nouns:
        training += [{'group': 'train_query_noun', 'name': None, 'key': None, 'text': text} for text in (
            '看看桌上的单子，这里列的是哪些饮品？',
            '翻开饮料菜单，先介绍最热门的那款饮品。',
            '关于杯子和饮品，这份手册怎么区分？',
            '最爱喝哪款饮品是一个偏好问题吗？',
            '现在看到的这页笔记里有几个饮品名称？',
            '从更新后的目录看，一共有几款饮品？',
            '这条备忘说的饮料和饮品是同一个意思吗？',
            '在先前的笔记里，饮品这一栏共有几条？',
        )]
    held = [{'group': 'known', 'name': name, 'key': key, 'text': t.format(name=name)}
            for name, key in zip(NAMES, KEYS) for t in HELD]
    held += [{'group': 'near_name', 'name': name, 'key': None, 'text': HELD[0].format(name=name)}
             for name in NEAR]
    held += [{'group': 'unknown', 'name': name, 'key': None, 'text': HELD[1].format(name=name)}
             for name in UNKNOWN]
    held += [{'group': 'no_or_multiple', 'name': None, 'key': None, 'text': text}
             for text in HELD_NEGATIVE]
    if fresh_boundary_holdout:
        held += [{'group': 'fresh_longer_name', 'name': name + suffix, 'key': None,
                  'text': f'能告诉我{name}{suffix}偏爱的饮料吗？'}
                 for name in NAMES for suffix in ('月', '溪')]
        if ({row['name'] for row in training if row['name'] is not None}
                & {row['name'] for row in held if row['key'] is None}):
            raise ValueError('Unregistered holdout names overlap training')
    rows = training + held
    if len({r['text'] for r in rows}) != len(rows):
        raise ValueError('Overlapping examples')
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='spora-original-name-') as folder:
        base = p.load_full_streamed().eval()
        replacements = {}
        if checkpoint is None:
            p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
            originals = {i: base.model.layers[i] for i in (6, 12)}
            chat = p.runtime.load_chat(base_loader=lambda: base)
            organism, state, tokenizer = chat.organism, chat.state, chat.tokenizer
            replacements = {i: base.model.layers[i] for i in originals}
            for index, block in originals.items():
                base.model.layers[index] = block
        else:
            decoder, organism, state = load_integrated(source, base, p.MODEL)
            if not decoder.compatible_block_indices:
                raise ValueError('Candidate must contain compatible decoder blocks')
            tokenizer = AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
        source_digest = p.frozen_digest(organism, state)
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
        features, allowed = extract(base, tokenizer, [r['text'] for r in rows], batch_size=2 if cluener_replay else 1)
        if truth.shape != allowed.shape or bool((truth & ~allowed).any()):
            raise ValueError('Feature mask mismatch')
        vector = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
        if include_following_tokens:
            vector = torch.stack([lookahead(row) for row in vector.masked_fill(~allowed[..., None], 0.)])
        n_train = len(training)
        x, y = vector[:n_train][allowed[:n_train]], truth[:n_train][allowed[:n_train]].float()
        if sequence_locator or span_locator:
            vector = vector.masked_fill(~allowed[..., None], 0.)
            locator = (SpanLocator if span_locator else SequenceLocator)(vector.shape[-1])
        else:
            locator = nn.Linear(x.shape[-1], 1)
        optimizer = torch.optim.AdamW(locator.parameters(), lr=0.0005, weight_decay=0.01)
        positive_weight = (y.numel() - y.sum()) / y.sum()
        if span_locator:
            positive_widths = truth[:n_train].sum(1)
            if bool((positive_widths > 8).any()):
                raise ValueError('A training name span exceeds the candidate width')
            targets = span_targets(truth[:n_train])
        for _ in range(400):
            optimizer.zero_grad(set_to_none=True)
            if span_locator:
                indices = torch.randint(n_train, (16,))
                logits = locator(vector[indices], allowed[indices])
                loss = F.cross_entropy(logits, targets[indices])
            elif sequence_locator:
                indices = torch.randint(n_train, (16,))
                mask = allowed[indices]
                logits = locator(vector[indices]).squeeze(-1)[mask]
                labels = truth[indices][mask].float()
            else:
                logits, labels = locator(x).squeeze(-1), y
            if not span_locator:
                loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=positive_weight)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite locator loss')
            loss.backward()
            optimizer.step()
        locator.eval()
        bases = torch.stack([F.normalize(site.current_basis.detach().cpu().float(), dim=-1)
                             for site in organism.memories])
        train_indices = [i for i, row in enumerate(training) if row['key'] is not None]
        vectors = pooled(features[train_indices], truth[train_indices])
        target = torch.stack([bases[:, training[i]['key']] for i in train_indices])
        projection = fit_binding(vectors, target)
        _, train_scores = predict(vectors, projection, bases)
        threshold = float(train_scores.min()) - 1e-6
        if not 0 <= threshold <= 1:
            raise ValueError('Invalid address threshold')
        fingerprints = IdentityFingerprints(organism.config.key_count).to(next(organism.parameters()).device)
        for name, key in zip(NAMES, KEYS):
            fingerprints.bind(key, name)
        guard = RecordingGuard(fingerprints, tokenizer)
        with torch.inference_mode():
            token_scores = (locator(vector[n_train:], allowed[n_train:]) if span_locator
                            else locator(vector[n_train:]).squeeze(-1).sigmoid())
        results = []
        for j, row in enumerate(held):
            selected, single = choose_span(token_scores[j], allowed[n_train + j], 0.5)
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
        for index, block in replacements.items():
            base.model.layers[index] = block
        if p.frozen_digest(organism, state) != source_digest:
            raise ValueError('Protected individual changed')
        basis_sha = address_basis_sha256(organism)
        locator_state = {key: value.detach().cpu().clone() for key, value in locator.state_dict().items()}
    if (p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(source) != checkpoint_sha
            or any(p.sha(path) != digest for path, digest in extra_sources.items())):
        raise ValueError('Protected files changed')
    summary = {}
    for group in sorted({r['group'] for r in results}):
        subset = [r for r in results if r['group'] == group]
        summary[group] = {'cases': len(subset), 'representable': sum(r['span_representable'] is True for r in subset),
                          'exact_span': sum(r['span_exact'] is True for r in subset),
                          'accepted': sum(r['accepted'] for r in subset), 'correct_key': sum(r['correct_key'] for r in subset),
                          'guarded_accepted': sum(r['guarded_accepted'] for r in subset),
                          'guarded_correct_key': sum(r['guarded_correct_key'] for r in subset)}
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'router.pt'
    architecture = 'span' if span_locator else 'sequence' if sequence_locator else 'linear'
    torch.save({'format': 'spora-original-layer-name-router-research-v1', 'base_sha256': base_sha,
                'checkpoint_sha256': checkpoint_sha, 'feature_config_sha256': feature_files_sha256(p.MODEL),
                'address_basis_sha256': basis_sha, 'hidden_indices': [6, 18], 'locator_threshold': 0.5,
                'locator_following_tokens': 2 if include_following_tokens else 0,
                'locator_architecture': architecture,
                'address_threshold': threshold, 'locator_state': locator_state, 'projection': projection.cpu()}, path)
    report = {'summary': summary, 'rows': results, 'train_count': len(training), 'dropped_train': dropped_train,
              'locator_only_train_count': sum(row['group'] == 'train_locator_only' for row in training),
              'threshold': threshold, 'locator_following_tokens': 2 if include_following_tokens else 0,
              'locator_architecture': architecture,
              'training_groups': {group: sum(r['group'] == group for r in training)
                                  for group in sorted({r['group'] for r in training})},
              'extra_source_sha256': {str(path): digest for path, digest in extra_sources.items()},
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
    (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, default=OUT)
    locator = parser.add_mutually_exclusive_group()
    locator.add_argument('--include-following-tokens', action='store_true')
    locator.add_argument('--sequence-locator', action='store_true')
    locator.add_argument('--span-locator', action='store_true')
    parser.add_argument('--diverse-boundaries', action='store_true')
    parser.add_argument('--fresh-boundary-holdout', action='store_true')
    parser.add_argument('--diverse-query-context', action='store_true')
    parser.add_argument('--contrast-query-nouns', action='store_true')
    parser.add_argument('--cluener-replay', action='store_true')
    args = parser.parse_args()
    main(checkpoint=args.checkpoint, output=args.output, include_following_tokens=args.include_following_tokens,
         diverse_boundaries=args.diverse_boundaries, fresh_boundary_holdout=args.fresh_boundary_holdout,
         diverse_query_context=args.diverse_query_context, sequence_locator=args.sequence_locator,
         span_locator=args.span_locator, contrast_query_nouns=args.contrast_query_nouns,
         cluener_replay=args.cluener_replay)


if __name__ == '__main__':
    cli()
