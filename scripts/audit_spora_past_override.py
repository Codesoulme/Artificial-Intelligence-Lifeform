"""Calibrate a selective historical correction using the trained version residual."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoTokenizer

from scripts.audit_qwen_compatible_natural_memory_20261121 import fresh_version_cases, load_head
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_revision_contrast import build_data, held_questions, natural
from spora.integrated_checkpoint import load_integrated_runtime
from spora.language_training import atomic_save, frozen_digest
from spora.research_temporal_intent import past_override_mask, temporal_logits, time_selection


@torch.no_grad()
def calibrate(runtime, tokenizer, old, fitted):
    texts, values = zip(*build_data()['calibration'], strict=True)
    labels = torch.tensor(values)
    features, mask = extract(runtime.decoder.base, tokenizer, list(texts), batch_size=2,
                             hidden_indices=tuple(old['hidden_indices']), last_logit_only=True)
    original = temporal_logits(old, features, mask).softmax(-1)
    current = time_selection(original, old['threshold'], factorized=True)
    was_correct = current == labels
    results = []
    for strength in (.25, .5, .75, 1., 1.25):
        settings = dict(fitted['version_residual'], strength=strength)
        revised = temporal_logits(dict(old, version_residual=settings), features, mask, texts=texts).softmax(-1)
        for confidence in (.55, .65, .75, .85, .95):
            change = past_override_mask(original, revised, old['threshold'], confidence)
            correct = torch.where(change, 1, current) == labels
            gained = int((correct & ~was_correct).sum())
            lost = int((~correct & was_correct).sum())
            results.append((gained - lost, gained, -lost, confidence, -strength, int(change.sum())))
    net, gained, neg_lost, confidence, neg_strength, changes = max(results)
    config = dict(fitted['version_residual'], strength=-neg_strength, past_override=confidence)
    return dict(old, version_residual=config), {'confidence': confidence, 'strength': -neg_strength,
                                               'net': net, 'gained': gained, 'lost': -neg_lost,
                                               'changes': changes, 'calibration_questions': len(labels)}


def main(args):
    reload_path = args.output / 'reloaded.json'
    if (args.reopen and reload_path.exists()) or (not args.reopen and args.output.exists()):
        raise FileExistsError(args.output)
    started = time.perf_counter()
    torch.set_num_threads(2)
    original_sha = sha(args.checkpoint)
    route_source_sha = sha(args.route_source)
    old, old_sha = load_head(args.temporal, 'temporal.pt', 'temporal_sha256', route_source_sha)
    router, router_sha = load_head(args.router, 'router.pt', 'router_sha256', route_source_sha)
    fitted, fitted_sha = load_head(args.residual, 'temporal.pt', 'temporal_sha256', original_sha)
    prior_report = json.loads((args.residual / 'result.json').read_text(encoding='utf-8'))
    if (prior_report['baseline_temporal_sha256'] != old_sha or prior_report['router_sha256'] != router_sha
            or fitted.get('architecture') != 'factorized'
            or old['state'].keys() != fitted['state'].keys()
            or any(not torch.equal(tensor, fitted['state'][key]) for key, tensor in old['state'].items())):
        raise ValueError('The fitted residual must share the existing read and version weights')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(args.checkpoint, load_full_streamed(), MODEL)
    runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    sources = {'checkpoint_sha256': original_sha, 'baseline_temporal_sha256': old_sha,
               'router_sha256': router_sha, 'fitted_temporal_sha256': fitted_sha}
    if args.reopen:
        saved = json.loads((args.output / 'result.json').read_text(encoding='utf-8'))
        if any(saved[key] != value for key, value in sources.items()):
            raise ValueError('Reopen requires the same temporal and model sources')
        candidate, candidate_sha = load_head(args.output, 'temporal.pt', 'temporal_sha256', original_sha)
    else:
        candidate, calibration = calibrate(runtime, tokenizer, old, fitted)
        candidate['checkpoint_sha256'] = original_sha
        print('Selective correction calibration', calibration, flush=True)
    results = {}
    for name, questions in (('fresh_v2', fresh_version_cases()), ('new_holdout', held_questions())):
        reference = prior_report['natural'][name]['original']
        trial = natural(runtime, tokenizer, router, candidate, questions)
        before, after = reference['rows'], trial['rows']
        if any(left['question'] != right['question'] for left, right in zip(before, after, strict=True)):
            raise ValueError('Evaluation questions changed')
        if any((left['key'] is None) != (right['key'] is None) for left, right in zip(before, after, strict=True)):
            raise ValueError('Selective correction altered a read decision')
        gained = [right['question'] for left, right in zip(before, after, strict=True)
                  if not left['correct_pair'] and right['correct_pair']]
        lost = [right['question'] for left, right in zip(before, after, strict=True)
                if left['correct_pair'] and not right['correct_pair']]
        results[name] = {'baseline': reference['summary'], **trial, 'gained': gained, 'lost': lost}
        print('Selective end-to-end', name, trial['summary'], 'gained', len(gained), 'lost', len(lost), flush=True)
    if frozen_digest(runtime.organism, runtime.state) != individual or sha(args.checkpoint) != original_sha:
        raise ValueError('An original model or individual changed during evaluation')
    if args.reopen:
        if results != saved['results']:
            raise ValueError('Reloaded temporal candidate did not reproduce its evaluation')
        report = {**sources, 'temporal_sha256': candidate_sha, 'exact_reopen': True,
                  'rows_reproduced': sum(len(trial['rows']) for trial in results.values()),
                  'seconds': time.perf_counter() - started}
        reload_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print('Reproduced', report['rows_reproduced'], 'queries', flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    weights = args.output / 'temporal.pt'
    atomic_save(candidate, weights)
    passed = all(len(trial['gained']) >= len(trial['lost']) for trial in results.values()) and any(
        len(trial['gained']) > len(trial['lost']) for trial in results.values()
    )
    report = {**sources, 'temporal_sha256': sha(weights), 'calibration': calibration,
              'results': results, 'passed': passed, 'seconds': time.perf_counter() - started}
    (args.output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'route-source', 'router', 'temporal', 'residual', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--reopen', action='store_true')
    main(parser.parse_args())
