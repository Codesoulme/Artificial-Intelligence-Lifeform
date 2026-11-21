"""Compare character identity and contextual features for SPORA name detection."""

import argparse
import json
from pathlib import Path

import torch

from scripts.train_qwen_entity_query_20261122 import (
    FORMAT, INDICES, MENTION_PREFIX, MODEL, NAMES, ROLE_PREFIX, SEED, TRAINING_NAMES, WEIGHT_SHA,
    annotated_mentions, data_splits, detected_spans, features_for, fit_tagger, lexical_character_features,
    selected_targets, sha,
)
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_checkpoint import load_integrated
from spora.language_training import atomic_save
from spora.named_address import feature_files_sha256
from spora.research_entity_query import MentionTagger, QueryTargetSelector
from transformers import AutoTokenizer


def measure(rows, predicted, selected=None):
    counts = {'questions': len(rows), 'target_questions': 0, 'exact_targets': 0, 'wrong_enrolled_targets': 0,
              'gold_mentions': 0, 'predicted_mentions': 0, 'correct_mentions': 0, 'exact_mention_sets': 0}
    details = []
    for index, ((text, gold, target), mentions) in enumerate(zip(rows, predicted, strict=True)):
        counts['gold_mentions'] += len(gold)
        counts['predicted_mentions'] += len(mentions)
        counts['correct_mentions'] += len(set(gold) & set(mentions))
        counts['exact_mention_sets'] += gold == mentions
        selected_name = None
        if selected is not None:
            span, _ = selected[index]
            selected_name = text[slice(*span)] if span is not None else None
            counts['target_questions'] += target is not None
            counts['exact_targets'] += target is not None and selected_name == target
            counts['wrong_enrolled_targets'] += selected_name in NAMES and selected_name != target
        details.append({'question': text, 'expected': target, 'target': selected_name,
                        'mentions': [text[a:b] for a, b in mentions], 'expected_mentions': [text[a:b] for a, b in gold]})
    return {'summary': counts, 'rows': details}


def boundary_contrasts():
    pairs = zip(TRAINING_NAMES[:8:2], TRAINING_NAMES[1:8:2], strict=True)
    templates = (
        '等会儿给{name}搬把凳子。', '桌上的座签给{name}留着。',
        '你把替{name}收着的那袋饼干送来吧。', '写在最外面的是{name}领的号码。',
        '门卫说{name}已经来取餐了。', '这张新卡归{name}，那张还给{other}。',
        '把{other}的单放一边，看看{name}这一张。', '见到{name}就转达{other}在门外等候。',
        '让我核实清单上{name}选的尺寸和编号。', '我只对照{name}尚未盖章的那一页。',
        '别把{name}错当成{other}。', '让{other}先挑，给{name}也留一份。',
    )
    return [(text, annotated_mentions(text, (short, long)), None)
            for short, long in pairs
            for name, other in ((short, long), (long, short))
            for template in templates
            for text in (template.format(name=name, other=other),)]


def run(checkpoint, baseline_dir, output, *, bidirectional=False, contrast=False):
    if output.exists():
        raise FileExistsError(output)
    baseline_path = baseline_dir / 'router.pt'
    checkpoint_sha, baseline_sha = sha(checkpoint), sha(baseline_path)
    base = torch.load(baseline_path, map_location='cpu', weights_only=True)
    report = json.loads((baseline_dir / 'result.json').read_text(encoding='utf-8'))
    if (baseline_sha != report['router_sha256'] or base['format'] != FORMAT or tuple(base['hidden_indices']) != INDICES
            or base['checkpoint_sha256'] != checkpoint_sha or base['base_sha256'] != WEIGHT_SHA
            or base['feature_config_sha256'] != feature_files_sha256(MODEL)):
        raise ValueError('Reference name router provenance mismatch')

    authored, _, news_train, news_validation, held, sources = data_splits()
    additional = boundary_contrasts() if contrast else []
    train = authored + news_train + additional
    evaluations = [row for rows in held.values() for row in rows]
    all_rows = train + news_validation + evaluations
    if {row[0] for row in train}.intersection(row[0] for row in news_validation + evaluations):
        raise ValueError('Name detector train and validation overlap')

    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    decoder, _, _ = load_integrated(checkpoint, load_full_streamed(), MODEL)
    decoder.eval().requires_grad_(False)
    lexical, mask = lexical_character_features(decoder, tokenizer, all_rows)
    context, context_mask = features_for(decoder, tokenizer, all_rows, MENTION_PREFIX)
    if not torch.equal(mask, context_mask):
        raise ValueError('Lexical and contextual feature masks differ')
    role, role_mask = features_for(decoder, tokenizer, evaluations, ROLE_PREFIX)
    del decoder
    selector = QueryTargetSelector(base['width']).eval()
    selector.load_state_dict(base['selector'], strict=True)
    known = len(train)
    news_count = len(news_validation)
    if base.get('mention_features', 'context') != 'context':
        raise ValueError('Reference router must use contextual mention features')
    control = MentionTagger(base.get('mention_width', base['width']),
                            bidirectional=base.get('mention_bidirectional', False)).eval()
    control.load_state_dict(base['tagger'], strict=True)
    detected = {'context': detected_spans(control, context[known:], mask[known:])}
    states = {}
    if not bidirectional and not contrast:
        torch.manual_seed(SEED)
        tagger = fit_tagger(lexical[:known], mask[:known], train)
        detected['lexical'] = detected_spans(tagger, lexical[known:], mask[known:])
        states['lexical'] = (lexical.shape[-1], tagger.state_dict())
    hybrid = torch.cat((context, lexical), dim=-1)
    del context, lexical
    torch.manual_seed(SEED)
    tagger = fit_tagger(hybrid[:known], mask[:known], train, bidirectional=bidirectional)
    detected['hybrid'] = detected_spans(tagger, hybrid[known:], mask[known:])
    states['hybrid'] = (hybrid.shape[-1], tagger.state_dict())
    del hybrid

    sources = {**sources, checkpoint: checkpoint_sha, baseline_path: baseline_sha}
    if any(sha(path) != digest for path, digest in sources.items()):
        raise ValueError('Name training source changed')
    output.mkdir(parents=True, exist_ok=False)
    for variant, spans in detected.items():
        news = measure(news_validation, spans[:news_count])
        natural = spans[news_count:]
        chosen = selected_targets(selector, role, role_mask, natural, base['margin'])
        groups = {}
        offset = 0
        for name, rows in held.items():
            part = slice(offset, offset + len(rows))
            groups[name] = measure(rows, natural[part], chosen[part])
            offset += len(rows)
        folder = output / variant
        folder.mkdir()
        digest = baseline_sha
        if variant != 'context':
            width, weights = states[variant]
            path = folder / 'router.pt'
            atomic_save(dict(base, mention_features=variant, mention_width=width,
                             mention_bidirectional=bidirectional, tagger=weights), path)
            digest = sha(path)
        outcome = {'router_sha256': digest, 'baseline_router_sha256': baseline_sha,
                   'sources': {str(path): digest for path, digest in sources.items()},
                   'mention_train_count': known if variant != 'context' else report['mention_train_count'],
                   'boundary_contrast_count': len(additional) if variant != 'context' else 0,
                   'mention_bidirectional': bidirectional and variant != 'context',
                   'news': news, 'groups': groups}
        (folder / 'result.json').write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(variant, news['summary'], {name: item['summary'] for name, item in groups.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--boundary-contrasts', action='store_true')
    args = parser.parse_args()
    run(args.checkpoint, args.baseline, args.output, bidirectional=args.bidirectional,
        contrast=args.boundary_contrasts)
