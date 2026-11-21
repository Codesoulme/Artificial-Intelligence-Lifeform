"""Learn full-name boundaries and memory query roles as separate tasks."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.deepseek_named_binding_probe import NAMES
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_qwen_character_name_span_20261122 import (
    INDICES, MODEL, WEIGHT_SHA, character_inputs, examples, sha, splits as name_splits,
)
from spora.integrated_checkpoint import load_integrated
from spora.language_training import atomic_save
from spora.named_address import feature_files_sha256
from spora.research_entity_query import MentionTagger, QueryTargetSelector, mention_spans
from transformers import AutoTokenizer

FORMAT = 'spora-factorized-name-query-v1'
MENTION_PREFIX = '提取原文出现的所有人名，保留每个人的完整姓名。\n原文：'
ROLE_PREFIX = '判断记忆查询针对谁，区分要查询的人和句中顺便提到的人。\n原文：'
SEED = 20261122
TRAINING_NAMES = ('阿珂', '阿珂尧', '小青', '小青禾', '阿慈', '阿慈菀', '小槿', '小槿秋',
                  '周安雅', '申桥', '秦月弦', '楚蘅', '梁晓飞', '小朔', '阿葵', '穆夏')


def annotated_mentions(text, names):
    """Build annotations from the controlled names in an authored fixture."""
    ordered = sorted(set(names), key=lambda name: (-len(name), name))
    spans = []
    start = 0
    while start < len(text):
        name = next((name for name in ordered if text.startswith(name, start)), None)
        if name is None:
            start += 1
        else:
            spans.append((start, start + len(name)))
            start += len(name)
    return spans


def news_rows():
    from pyarrow import parquet
    from scripts.deepseek_cluener_locator_train import TRAIN, TRAIN_SHA, VALIDATION, VALIDATION_SHA, remove_validation_texts
    from scripts.deepseek_cluener_span_audit import labeled_names

    sources = {TRAIN: TRAIN_SHA, VALIDATION: VALIDATION_SHA}
    if any(sha(path) != digest for path, digest in sources.items()):
        raise ValueError('CLUENER source digest mismatch')
    train = parquet.read_table(TRAIN).to_pylist()
    validation = parquet.read_table(VALIDATION).to_pylist()
    def sample(source, limits):
        counts = [0, 0, 0]
        result, seen = [], set()
        for row in sorted(source, key=lambda row: hashlib.sha256(row['text'].encode()).digest()):
            text = row['text']
            if text in seen or not 8 <= len(text) <= 44 or any(name in text for name in NAMES):
                continue
            spans = labeled_names(row)
            group = min(len(spans), 2)
            if counts[group] >= limits[group] or any(l[1] > r[0] for l, r in zip(spans, spans[1:])):
                continue
            counts[group] += 1
            seen.add(text)
            result.append((text, spans, None))
            if counts == list(limits):
                break
        if counts != list(limits):
            raise ValueError('Insufficient CLUENER examples')
        return result

    return sample(remove_validation_texts(train, validation), (56, 200, 32)), sample(validation, (16, 40, 16)), sources


def data_splits():
    calibration_names = ('孟禾', '黎恬', '小墨', '阿扬澄')
    fields = ('饮品', '座位', '旧称', '饮料', '常用尺寸')
    authored = lambda pairs, names: [(text, annotated_mentions(text, names), target) for text, target in pairs]
    train = authored(examples(TRAINING_NAMES, fields), TRAINING_NAMES)
    calibration = authored(examples(calibration_names, ('咖啡', '密码提示')), calibration_names)
    mention_train, mention_test, sources = news_rows()

    fixture_names = (*NAMES, *(name + suffix for suffixes in ('远铭舟诚洁越汐瑜', '珉宣雁炜瑾筝禾修', '溪遥彦岑羽朗言川')
                                for name, suffix in zip(NAMES, suffixes, strict=True)))
    previous = name_splits()
    held = {name: authored(previous[name], fixture_names) for name in ('original', 'fresh_v1', 'fresh_v2', 'boundary')}
    return train, calibration, mention_train, mention_test, held, sources


def role_contrasts():
    rows = []
    for i, (short, long) in enumerate(zip(TRAINING_NAMES[:8:2], TRAINING_NAMES[1:8:2], strict=True)):
        field = ('饮品', '取货地点', '座位', '标签颜色')[i]
        for first, second in ((short, long), (long, short)):
            options = (
                (f'{first}帮我提了个醒，我只需要{second}曾登记的{field}。', second),
                (f'先查{first}的{field}。{second}那份等会儿再打开。', first),
                (f'麻烦确认{first}当时采用的{field}，别读{second}的资料。', first),
                (f'这里{first}是旁听人，真正要核查{field}的是{second}。', second),
                (f'{first}和{second}的{field}先不要打开，帮我改写这句话。', None),
                (f'{first}和{second}更换{field}的日期各是几号？只问日期。', None),
            )
            rows.extend((text, annotated_mentions(text, (short, long)), target) for text, target in options)
    return rows


def features_for(decoder, tokenizer, rows, prefix):
    questions = [row[0] for row in rows]
    hidden, _ = extract(decoder.base, tokenizer, questions, batch_size=2, hidden_indices=INDICES,
                        user_prefix=prefix, last_logit_only=True)
    return character_inputs(tokenizer, questions, hidden, prefix)


def tag_targets(rows, width):
    targets = torch.zeros((len(rows), width), dtype=torch.long)
    for i, (_, spans, _) in enumerate(rows):
        for left, right in spans:
            targets[i, left] = 1
            targets[i, left + 1:right] = 2
    return targets


def candidate_masks(spans, width):
    positions = torch.arange(width)
    masks = torch.zeros((len(spans), max(1, max(map(len, spans))), width), dtype=torch.bool)
    for index, candidates in enumerate(spans):
        for slot, (left, right) in enumerate(candidates):
            masks[index, slot] = (positions >= left) & (positions < right)
    return masks


def fit_tagger(features, mask, rows):
    model = MentionTagger(features.shape[-1])
    targets = tag_targets(rows, features.shape[1])
    weights = torch.tensor([1., 3., 3.])
    train(model, len(rows), 14,
          lambda ix: F.cross_entropy(model(features[ix], mask[ix])[mask[ix]], targets[ix][mask[ix]], weight=weights),
          'mention')
    return model.eval()


def fit_selector(features, mask, rows, initial=None):
    model = QueryTargetSelector(features.shape[-1])
    if initial is not None:
        model.load_state_dict(initial, strict=True)
    candidates = candidate_masks([spans for _, spans, _ in rows], features.shape[1])
    targets = []
    for text, spans, target in rows:
        indices = [i for i, (left, right) in enumerate(spans) if text[left:right] == target]
        if target is not None and len(indices) != 1:
            raise ValueError('Every positive target must have a unique complete mention')
        targets.append(indices[0] if indices else candidates.shape[1])
    targets = torch.tensor(targets)
    train(model, len(rows), 18,
          lambda ix: F.cross_entropy(model(features[ix], mask[ix], candidates[ix]), targets[ix]), 'role')
    return model.eval()


def train(model, count, epochs, loss_for, label):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    generator = torch.Generator().manual_seed(SEED)
    for epoch in range(epochs):
        for indices in torch.randperm(count, generator=generator).split(24):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(indices)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite entity query loss')
            loss.backward()
            optimizer.step()
        if epoch + 1 == epochs:
            print(label, 'final batch loss', float(loss.detach()), flush=True)


@torch.inference_mode()
def detected_spans(model, features, mask):
    tags = torch.cat([model(x, valid).argmax(-1) for x, valid in zip(features.split(16), mask.split(16), strict=True)])
    return [mention_spans(row[valid].tolist()) for row, valid in zip(tags, mask, strict=True)]


@torch.inference_mode()
def selected_targets(model, features, mask, spans, margin=0.):
    candidates = candidate_masks(spans, features.shape[1])
    logits = torch.cat([model(x, valid, cand) for x, valid, cand in zip(
        features.split(16), mask.split(16), candidates.split(16), strict=True)])
    best, index = logits[:, :-1].max(-1)
    confidence = best - logits[:, -1]
    return [(None if score <= margin else row[int(slot)], float(score) if torch.isfinite(score) else None)
            for row, slot, score in zip(spans, index, confidence, strict=True)]


def main(checkpoint, output, *, selector_baseline=None):
    if output.exists():
        raise FileExistsError(output)
    train_rows, calibration, news_train, news_held, held, sources = data_splits()
    baseline = None
    if selector_baseline is not None:
        path = selector_baseline / 'router.pt'
        metadata = json.loads((selector_baseline / 'result.json').read_text(encoding='utf-8'))
        digest = sha(path)
        baseline = torch.load(path, map_location='cpu', weights_only=True)
        if (digest != metadata['router_sha256'] or baseline['format'] != FORMAT
                or baseline['checkpoint_sha256'] != sha(checkpoint) or baseline['base_sha256'] != WEIGHT_SHA
                or baseline['feature_config_sha256'] != feature_files_sha256(MODEL)
                or tuple(baseline['hidden_indices']) != INDICES):
            raise ValueError('Name selector baseline does not match the checkpoint')
        sources[path] = digest
        train_rows += role_contrasts()
    all_held = [row for rows in held.values() for row in rows]
    train_text = {row[0] for row in train_rows + news_train}
    if train_text.intersection(row[0] for row in calibration + news_held + all_held):
        raise ValueError('Train and evaluation text overlaps')
    seed = SEED
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    checkpoint_sha = sha(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    decoder, _, _ = load_integrated(checkpoint, load_full_streamed(), MODEL)
    decoder.eval().requires_grad_(False)
    mention_training = train_rows + news_train if baseline is None else []
    x, mask = features_for(decoder, tokenizer, mention_training + news_held + all_held, MENTION_PREFIX)
    count = len(mention_training)
    if baseline is None:
        tagger = fit_tagger(x[:count], mask[:count], mention_training)
    else:
        tagger = MentionTagger(baseline['width']).eval()
        tagger.load_state_dict(baseline['tagger'], strict=True)
    predicted = detected_spans(tagger, x[count:], mask[count:])
    del x, mask
    training_count, calibration_count = len(train_rows), len(calibration)
    x, mask = features_for(decoder, tokenizer, train_rows + calibration + all_held, ROLE_PREFIX)
    del decoder
    selector = fit_selector(x[:training_count], mask[:training_count], train_rows,
                            initial=None if baseline is None else baseline['selector'])
    middle = slice(training_count, training_count + calibration_count)
    oracle_cal = selected_targets(selector, x[middle], mask[middle], [row[1] for row in calibration])
    margin = max([0.] + [conf for (_, conf), (_, _, target) in zip(oracle_cal, calibration, strict=True)
                         if target is None and conf is not None])
    middle_end = training_count + calibration_count
    x, mask = x[middle_end:], mask[middle_end:]
    auto_spans = predicted[len(news_held):]
    ranks = selected_targets(selector, x, mask, auto_spans, margin)
    oracle = selected_targets(selector, x, mask, [row[1] for row in all_held], margin)
    if sha(checkpoint) != checkpoint_sha or any(sha(path) != digest for path, digest in sources.items()):
        raise ValueError('Source changed during entity query training')
    output.mkdir(parents=True, exist_ok=False)
    weights = output / 'router.pt'
    atomic_save({'format': FORMAT, 'checkpoint_sha256': checkpoint_sha, 'base_sha256': WEIGHT_SHA,
                 'feature_config_sha256': feature_files_sha256(MODEL), 'hidden_indices': list(INDICES),
                 'width': x.shape[-1], 'margin': margin, 'tagger': tagger.state_dict(), 'selector': selector.state_dict()},
                weights)
    def scores(rows, spans, chosen=None, provided=None):
        info = {'rows': len(rows), 'exact_mention_sets': 0, 'gold_mentions': 0, 'predicted_mentions': 0,
                'correct_mentions': 0, 'oracle_correct': 0, 'correct': 0, 'positive': 0, 'wrong_enrolled': 0,
                'correct_positive': 0, 'oracle_correct_positive': 0}
        details = []
        for i, ((text, gold, target), found) in enumerate(zip(rows, spans, strict=True)):
            info['exact_mention_sets'] += gold == found
            info['gold_mentions'] += len(gold)
            info['predicted_mentions'] += len(found)
            info['correct_mentions'] += len(set(gold) & set(found))
            if chosen is not None:
                span, conf = chosen[i]
                original, _ = provided[i]
                selected = text[slice(*span)] if span else None
                oracle_name = text[slice(*original)] if original else None
                info['oracle_correct'] += oracle_name == target
                info['correct'] += selected == target
                info['positive'] += target is not None
                info['correct_positive'] += target is not None and selected == target
                info['oracle_correct_positive'] += target is not None and oracle_name == target
                info['wrong_enrolled'] += selected in NAMES and selected != target
                details.append({'question': text, 'expected': target, 'selected': selected, 'oracle': oracle_name,
                                'mentions': [text[l:r] for l, r in found], 'expected_mentions': [text[l:r] for l, r in gold],
                                'confidence': conf})
        return {'summary': info, 'details': details}

    report = {'router_sha256': sha(weights), 'seed': seed, 'margin': margin,
              'mention_steps_retrained': baseline is None,
              'mention_train_count': len(mention_training) if baseline is None else metadata['mention_train_count'],
              'role_train_count': training_count,
              'news_holdout': scores(news_held, predicted[:len(news_held)]), 'held': {}}
    offset = 0
    for name, rows in held.items():
        part = slice(offset, offset + len(rows))
        report['held'][name] = scores(rows, auto_spans[part], ranks[part], oracle[part])
        print(name, report['held'][name]['summary'], flush=True)
        offset += len(rows)
    (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selector-baseline', type=Path)
    args = parser.parse_args()
    main(args.checkpoint, args.output, selector_baseline=args.selector_baseline)
