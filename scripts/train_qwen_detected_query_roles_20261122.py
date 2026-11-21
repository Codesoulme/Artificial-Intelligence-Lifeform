"""Fine-tune query roles with complete and imperfect detected name candidates."""

import argparse
import json
from pathlib import Path

import torch

from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_qwen_entity_query_20261122 import (
    FORMAT, INDICES, MODEL, ROLE_PREFIX, SEED, WEIGHT_SHA, data_splits, detected_spans,
    features_for, fit_selector, mention_features, role_contrasts, selected_targets, sha,
)
from spora.integrated_checkpoint import load_integrated
from spora.language_training import atomic_save
from spora.named_address import feature_files_sha256
from spora.research_entity_query import MentionTagger
from transformers import AutoTokenizer


def training_candidates(rows, predictions):
    """Reject incomplete candidate sets and contrast exact names with truncated names."""
    views = []
    for (text, exact, target), predicted in zip(rows, predictions, strict=True):
        distractors = set(predicted) - set(exact)
        for start, end in exact:
            if end - start >= 3:
                distractors.add((start, end - 1))
        complete = sorted(set(exact) | distractors)
        # An exact name in another span remains a legitimate candidate. Avoid
        # teaching contradictory labels when a synthetic prefix repeats it.
        if target is not None:
            complete = [span for span in complete if span in exact or text[slice(*span)] != target]
        views.append((text, complete, target))
        if target is not None:
            without_target = [span for span in complete if text[slice(*span)] != target]
            if without_target:
                views.append((text, without_target, None))
    return views


def main(checkpoint, source, output):
    if output.exists():
        raise FileExistsError(output)
    path = source / 'router.pt'
    source_hash, checkpoint_hash = sha(path), sha(checkpoint)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    report = json.loads((source / 'result.json').read_text(encoding='utf-8'))
    if (payload['format'] != FORMAT or source_hash != report['router_sha256']
            or payload['checkpoint_sha256'] != checkpoint_hash or payload['base_sha256'] != WEIGHT_SHA
            or payload['feature_config_sha256'] != feature_files_sha256(MODEL)
            or tuple(payload['hidden_indices']) != INDICES):
        raise ValueError('Source router does not match the DeepSeek checkpoint')
    train, calibration, _, _, held, _ = data_splits()
    train += role_contrasts()
    evaluation = [row for group in held.values() for row in group]
    if {row[0] for row in train}.intersection(row[0] for row in calibration + evaluation):
        raise ValueError('Role training overlaps with held-out text')
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    decoder, _, _ = load_integrated(checkpoint, load_full_streamed(), MODEL)
    decoder.eval().requires_grad_(False)
    rows = train + calibration
    features, mask = mention_features(decoder, tokenizer, rows, payload['mention_features'])
    tagger = MentionTagger(payload['mention_width'], bidirectional=payload.get('mention_bidirectional', False)).eval()
    tagger.load_state_dict(payload['tagger'], strict=True)
    mentions = detected_spans(tagger, features, mask)
    del features, mask
    features, mask = features_for(decoder, tokenizer, rows, ROLE_PREFIX)
    del decoder
    training = training_candidates(train, mentions[:len(train)])
    index_by_text = {row[0]: i for i, row in enumerate(train)}
    indices = torch.tensor([index_by_text[row[0]] for row in training])
    selector = fit_selector(features[indices], mask[indices], training, initial=payload['selector'])
    cut = len(train)
    calibration_spans = mentions[cut:]
    raw = selected_targets(selector, features[cut:], mask[cut:], calibration_spans, 0.)
    bad = [confidence for (span, confidence), (text, _, target) in zip(raw, calibration, strict=True)
           if span is not None and text[slice(*span)] != target and confidence is not None]
    margin = max([0.] + bad)
    choices = selected_targets(selector, features[cut:], mask[cut:], calibration_spans, margin)
    if sha(checkpoint) != checkpoint_hash or sha(path) != source_hash:
        raise ValueError('Source changed during query role training')
    output.mkdir(parents=True, exist_ok=False)
    weights = output / 'router.pt'
    atomic_save(dict(payload, selector=selector.state_dict(), margin=margin), weights)
    summary = {'router_sha256': sha(weights), 'source_router_sha256': source_hash, 'margin': margin,
               'train_questions': len(train), 'train_views': len(training), 'calibration_questions': len(calibration),
               'calibration_positive': sum(target is not None for _, _, target in calibration),
               'calibration_exact_positive': sum(span is not None and text[slice(*span)] == target
                                                 for (span, _), (text, _, target) in zip(choices, calibration, strict=True))}
    (output / 'result.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.checkpoint, args.source, args.output)
