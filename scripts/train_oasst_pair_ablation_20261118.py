"""Matched question/context pair ablation on frozen dialogue features.

No GPU feature extraction, threshold selection on tests, or live memory changes.
The already-reviewed corpus is development material, not fresh blind evidence.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'runs/open-base-deps')]
import torch
from torch import nn
from torch.nn import functional as F
from spora.research_dialogue_folds import make_folds
from spora.research_question_context_pair import question_rows, pair_features
from spora.research_evidence_calibration import calibrate, confusion, rank_auc

SOURCE = ROOT/'runs/oasst-context-transfer-20261118'
REVIEW = ROOT/'runs/oasst-evidence-review-20261118'
OUT = ROOT/'runs/oasst-context-pair-ablation-20261118'
SEEDS = (4171, 4172, 4173)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(rows, predictions):
    result = {}
    for variant in ('current_only', 'original_history', 'wrong_history'):
        selected = [(r, p) for r, p in zip(rows, predictions) if r['variant'] == variant]
        result[variant] = dict(tp=sum(r['need']==1 and p==1 for r, p in selected),
                               fn=sum(r['need']==1 and p==0 for r, p in selected),
                               fp=sum(r['need']==0 and p==1 for r, p in selected),
                               tn=sum(r['need']==0 and p==0 for r, p in selected))
    case_ids = {r['case'] for r in rows if r['history_dependent']}
    result['dependent_all_three_correct'] = sum(
        all(r['need']==p for r, p in zip(rows, predictions) if r['case']==c) for c in case_ids)
    result['dependent_cases'] = len(case_ids)
    return result


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch.set_num_threads(2)
    paths = {'inputs': SOURCE/'inputs.json', 'features': SOURCE/'features.pt', 'result': SOURCE/'result.json',
             'queue': REVIEW/'queue.json', 'review': REVIEW/'second-review.json',
             'script': Path(__file__), 'pair_code': ROOT/'spora/research_question_context_pair.py', 'fold_code': ROOT/'spora/research_dialogue_folds.py',
             'calibration_code': ROOT/'spora/research_evidence_calibration.py'}
    hashes = {k: sha(p) for k, p in paths.items()}
    parent = json.loads(paths['result'].read_text(encoding='utf8'))
    for name in ('inputs', 'features'):
        if hashes[name] != parent['artifacts'][paths[name].name]:
            raise ValueError('Parent artifact mismatch')
    if hashes['queue'] != parent['source_sha256']['queue']:
        raise ValueError('Queue mismatch')
    original = json.loads(paths['inputs'].read_text(encoding='utf8'))
    queue = json.loads(paths['queue'].read_text(encoding='utf8'))
    review = json.loads(paths['review'].read_text(encoding='utf8'))
    agreed = {r['case'] for r in review['reviewed_cases'] if r['agrees_with_original'] and r['need'] is not None}
    indices = [i for i, r in enumerate(original) if r['case'] in agreed]
    rows = [original[i] for i in indices]
    mapping = {mid: r['group'] for r in queue for mid in r['message_ids']}
    folds = make_folds(rows, mapping)
    features = torch.load(paths['features'], map_location='cpu', weights_only=True)
    OUT.mkdir()
    manifest = {'source_sha256': hashes, 'rows': rows, 'folds': folds, 'seeds': SEEDS,
                'modes': ['linear_pair', 'interaction_pair'], 'pooling': 'last', 'steps': 800, 'learning_rate': .02, 'weight_decay': .001,
                'selection': 'Agreement only, already observed development corpus; no independent accuracy claim.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    runs, states = [], {}
    question_indices = question_rows(rows)
    context_features = features['last'][indices]
    for kind in ('linear_pair', 'interaction_pair'):
        x = pair_features(context_features, context_features[question_indices], kind)
        for seed in SEEDS:
            predictions = [None]*len(rows)
            fold_records = []
            for fold in folds:
                ti, ci, ei = (fold[k] for k in ('train', 'calibration', 'test'))
                torch.manual_seed(seed)
                head = nn.Linear(x.shape[-1], 1)
                opt = torch.optim.AdamW(head.parameters(), lr=.02, weight_decay=.001)
                y = torch.tensor([rows[i]['need'] for i in ti], dtype=torch.float32)
                for _ in range(800):
                    loss = F.binary_cross_entropy_with_logits(head(x[ti]).squeeze(-1), y)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                with torch.no_grad():
                    scores = head(x).squeeze(-1).tolist()
                thresholds, _ = calibrate([scores[i] for i in ci], [rows[i]['need'] for i in ci])
                threshold = thresholds['balanced']
                for i in ei:
                    predictions[i] = int(scores[i]>=threshold)
                record = {'fold': fold['fold'], 'threshold': threshold, 'training_loss': float(loss.detach()),
                          'scores': {}, 'metrics': {}}
                for name, ids in (('train', ti), ('calibration', ci), ('test', ei)):
                    ss, yy = [scores[i] for i in ids], [rows[i]['need'] for i in ids]
                    record['scores'][name] = ss
                    record['metrics'][name] = dict(confusion(ss, yy, threshold), auc=rank_auc(ss, yy))
                fold_records.append(record)
                states[f'{kind}-{seed}-{fold["fold"]}'] = {'state': head.state_dict(), 'threshold': threshold}
            if any(p is None for p in predictions):
                raise ValueError('Missing out-of-fold prediction')
            baseline = next(r for r in parent['results'] if r['pooling']=='last' and r['seed']==seed)
            baseline_predictions = [int(baseline['scores'][i]>=baseline['thresholds']['balanced']) for i in indices]
            run = {'pooling': kind, 'seed': seed, 'folds': fold_records, 'predictions': predictions,
                   'metrics': metrics(rows, predictions), 'frozen_baseline': metrics(rows, baseline_predictions)}
            runs.append(run)
            print(json.dumps({'pooling': kind, 'seed': seed, 'metrics': run['metrics']}), flush=True)
    if any(sha(p)!=hashes[k] for k, p in paths.items()):
        raise ValueError('Protected source changed')
    torch.save(states, OUT/'heads.pt')
    result = {'runs': runs, 'seconds': time.perf_counter()-started,
              'artifacts': {name: sha(OUT/name) for name in ('manifest.json', 'heads.pt')},
              'limits': ['Already observed development corpus; no new blind data. Equal classifier size and steps, not identical compute.',
                         'Folds purge donor trees but cannot remove semantic or pretraining overlap.',
                         'Classification only, no generation or neural memory validation.',
                         'Adaptation changes both training data and thresholds relative to frozen baseline.']}
    (OUT/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
