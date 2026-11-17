"""Frozen-head transfer audit on locally reviewed conversational referents.

Labels concern omitted conversational context, not truth or neural persistence.
Source text and model outputs stay in runs/. No training or threshold fitting.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_evidence_pooling_ablation_20261118 import pooled

SOURCE = ROOT / 'runs/oasst-evidence-review-20261118'
PARENT = ROOT / 'runs/deepseek-evidence-pooling-ablation-20261118'
OUT = ROOT / 'runs/oasst-context-transfer-20261118'


def build_rows(queue, annotations):
    lookup = {r['case']: r for r in queue}
    if len(lookup) != len(queue) or len({r['group'] for r in queue}) != len(queue):
        raise ValueError('Duplicate case or source tree')
    if {r['case'] for r in annotations} != set(lookup) or len(annotations) != len(queue):
        raise ValueError('Incomplete annotations')
    rows = []
    for review in annotations:
        need = review['needs_previous_context']
        if need is None:
            continue
        if need not in (0, 1):
            raise ValueError('Invalid label')
        row, other = lookup[review['case']], lookup[review['wrong_history_case']]
        if row['group'] == other['group']:
            raise ValueError('Wrong-history source must differ')
        if review['group'] != row['group'] or review['current_message_id'] != row['message_ids'][-1]:
            raise ValueError('Review source mismatch')
        question = row['messages'][-1]['content']
        for variant, context, target in (
            ('current_only', [], need), ('original_history', row['messages'][:-1], 0),
            ('wrong_history', other['messages'][:-1], need)):
            # Concatenate in temporal order, matching the plain-text probe input.
            text = '\n'.join([m['content'] for m in context] + [question])
            rows.append({'case': row['case'], 'group': row['group'], 'variant': variant,
                         'history_dependent': need, 'need': target, 'text': text,
                         'current_message_id': row['message_ids'][-1],
                         'history_message_ids': (other if variant == 'wrong_history' else row)['message_ids'][:-1]
                         if context else []})
    return rows


def summarize(rows, scores, threshold):
    if len(rows) != len(scores) or len(rows) % 3:
        raise ValueError('Expected aligned triples')
    result = {}
    for variant in ('current_only', 'original_history', 'wrong_history'):
        chosen = [(r, s) for r, s in zip(rows, scores) if r['variant'] == variant]
        result[variant] = {'tp': sum(r['need'] == 1 and s >= threshold for r, s in chosen),
                           'fn': sum(r['need'] == 1 and s < threshold for r, s in chosen),
                           'fp': sum(r['need'] == 0 and s >= threshold for r, s in chosen),
                           'tn': sum(r['need'] == 0 and s < threshold for r, s in chosen)}
    result['dependent_all_three_correct'] = sum(
        rows[i]['history_dependent'] == 1 and all((scores[i+j] >= threshold) == bool(rows[i+j]['need'])
                                                 for j in range(3)) for i in range(0, len(rows), 3))
    return result


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch.set_num_threads(2)
    paths = {'queue': SOURCE/'queue.json', 'review': SOURCE/'annotations.json', 'source_manifest': SOURCE/'manifest.json',
             'parent_result': PARENT/'result.json', 'parent_manifest': PARENT/'manifest.json', 'heads': PARENT/'heads.pt',
             'base': MODEL/'model.safetensors', 'config': MODEL/'config.json', 'tokenizer': MODEL/'tokenizer.json',
             'tokenizer_config': MODEL/'tokenizer_config.json', 'script': Path(__file__)}
    hashes = {k: sha(p) for k, p in paths.items()}
    parent = json.loads(paths['parent_result'].read_text(encoding='utf8'))
    manifest = json.loads(paths['parent_manifest'].read_text(encoding='utf8'))
    review = json.loads(paths['review'].read_text(encoding='utf8'))
    if (hashes['base'] != WEIGHT_SHA or hashes['queue'] != review['queue_sha256']
            or hashes['heads'] != parent['artifacts']['heads.pt']
            or hashes['parent_manifest'] != parent['artifacts']['manifest.json']):
        raise ValueError('Protected source mismatch')
    for key in ('base', 'config', 'tokenizer', 'tokenizer_config'):
        if hashes[key] != manifest['source_sha256'][str(paths[key])]:
            raise ValueError('Feature source mismatch: ' + key)
    queue = json.loads(paths['queue'].read_text(encoding='utf8'))
    source_manifest = json.loads(paths['source_manifest'].read_text(encoding='utf8'))
    if source_manifest['queue_sha256'] != hashes['queue']:
        raise ValueError('Queue manifest mismatch')
    rows = build_rows(queue, review['reviews'])
    old = {r['text'] for split in manifest['splits'].values() for r in split}
    if old & {r['text'] for r in rows}:
        raise ValueError('Exact prior-probe text overlap')
    OUT.mkdir()
    (OUT/'inputs.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf8')
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    features = {'mean': [], 'last': []}
    lengths = []
    with torch.inference_mode():
        for i, row in enumerate(rows):
            batch = tok(row['text'], return_tensors='pt', add_special_tokens=False).to('cuda')
            length = batch.input_ids.shape[1]
            if length > 2048:
                raise ValueError('Long input requires explicit handling; no silent truncation')
            lengths.append(length)
            values = pooled(model.model(**batch, use_cache=False).last_hidden_state, batch['attention_mask'])
            for key in features:
                features[key].append(values[key].cpu())
            if i % 15 == 0:
                print(json.dumps({'feature_rows': i+1, 'total': len(rows)}), flush=True)
    x = {k: F.normalize(torch.cat(v), dim=-1) for k, v in features.items()}
    torch.save(x, OUT/'features.pt')
    heads = torch.load(paths['heads'], map_location='cpu', weights_only=True)
    results = []
    for run in parent['results']:
        kind, seed = run['pooling'], run['seed']
        saved = heads[f'{kind}-{seed}']
        if saved['thresholds'] != run['thresholds']:
            raise ValueError('Parent threshold mismatch')
        head = nn.Linear(x[kind].shape[-1], 1)
        head.load_state_dict(saved['state'], strict=True)
        with torch.no_grad():
            scores = head(x[kind]).squeeze(-1).tolist()
        results.append({'pooling': kind, 'seed': seed, 'thresholds': saved['thresholds'], 'scores': scores,
                        'metrics': {k: summarize(rows, scores, t) for k, t in saved['thresholds'].items()}})
    if any(sha(p) != hashes[k] for k, p in paths.items()):
        raise ValueError('Protected input changed')
    result = {'source_sha256': hashes, 'rows': len(rows), 'lengths': lengths, 'results': results,
              'seconds': time.perf_counter()-started, 'artifacts': {f: sha(OUT/f) for f in ('inputs.json', 'features.pt')},
              'limits': ['Single-agent labels, no independent human adjudication.',
                         'Conversation reference task differs from personal fact evidence training.',
                         'No training, generation, or persistence test; source statements may be false.',
                         'Prior-tree exclusions do not establish absence from all earlier work or pretraining.']}
    (OUT/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps([{'pooling': r['pooling'], 'seed': r['seed'], 'balanced': r['metrics']['balanced']} for r in results]), flush=True)


if __name__ == '__main__':
    main()
