"""Isolated natural QA span training with synthetic abstention replay."""
import json
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch.nn import functional as F
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.audit_cmrc2018_dev_20261117 import SOURCE as DEV, BLOB_SHA1 as DEV_BLOB, blob_sha1
from scripts.cmrc_who_development_selection_20261117 import OUT as DEV_MANIFEST, select as select_dev
from scripts.deepseek_fact_competing_train_20261117 import build_groups
from scripts.deepseek_fact_head_natural_regression_20261117 import CANDIDATE
from scripts.deepseek_fact_query_training_pilot_20261117 import evaluate
from scripts.deepseek_spora_general_subject_locator import extract, pack
from scripts.deepseek_spora_interleaved_natural_transaction import HEADS, SOURCE, source_paths
from scripts.prepare_cmrc_train_pilot_20261117 import TRAIN, TRAIN_BLOB, OUT as TRAIN_MANIFEST, select_train
from spora.research_context_span import ContextSpanSelector

OUT = ROOT / 'runs/deepseek-cmrc-reading-train-pilot-20261117'


def choose_replay(rows):
    """A deterministic, balanced artificial replay; not natural no-answer data."""
    known = [r for r in rows if r['kind'] == 'known']
    unknown = [r for r in rows if r['kind'] == 'unknown']
    rng = random.Random(61532)
    rng.shuffle(known)
    rng.shuffle(unknown)
    return known[:16] + unknown[:16]


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch.manual_seed(61532)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    paths = {**source_paths(), 'train_data': TRAIN, 'dev_data': DEV,
             'train_manifest': TRAIN_MANIFEST, 'dev_manifest': DEV_MANIFEST,
             'prior': CANDIDATE / 'head.pt', 'prior_report': CANDIDATE / 'result.json'}
    hashes = {k: runtime.sha(p) for k, p in paths.items()}
    train_raw, dev_raw = TRAIN.read_bytes(), DEV.read_bytes()
    if blob_sha1(train_raw) != TRAIN_BLOB or blob_sha1(dev_raw) != DEV_BLOB:
        raise ValueError('Dataset mismatch')
    train, _ = select_train(json.loads(train_raw.decode('utf8')), json.loads(dev_raw.decode('utf8')))
    dev, _ = select_dev(json.loads(dev_raw.decode('utf8')))
    train_manifest = json.loads(TRAIN_MANIFEST.read_text(encoding='utf8'))
    dev_manifest = json.loads(DEV_MANIFEST.read_text(encoding='utf8'))
    if ([r['digest'] for r in train] != [r['digest'] for r in train_manifest['rows']]
            or [r['digest'] for r in dev] != [r['digest'] for r in dev_manifest['rows']]
            or len(train) != 64 or len(dev) != 32):
        raise ValueError('Manifest mismatch')
    saved = torch.load(paths['prior'], map_location='cpu', weights_only=True)
    prior_report = json.loads(paths['prior_report'].read_text(encoding='utf8'))
    if (hashes['base'] != runtime.WEIGHT_SHA or hashes['prior'] != prior_report['head_sha256']
            or any(saved['source_sha256'][k] != hashes[k] for k in source_paths())):
        raise ValueError('Head mismatch')
    artificial = build_groups()
    replay = choose_replay(artificial['train'])
    checks = {'synthetic_same': artificial['stress_same_type'], 'synthetic_homonym': artificial['stress_homonym']}
    if {r['text'] for r in replay} & {r['text'] for rows in checks.values() for r in rows}:
        raise ValueError('Replay/control text overlap')
    runtime.CHECKPOINT = SOURCE
    with tempfile.TemporaryDirectory(prefix='spora-cmrc-train-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'not-created.pt'
        chat = runtime.load_chat(natural_write_heads_path=HEADS)
        groups = {'train': train, 'dev': dev, 'replay': replay, **checks}
        data = {}
        for name, rows in groups.items():
            data[name] = []
            for row in rows:
                data[name].extend(extract(chat, [row]))
        if runtime.WORKING_CHECKPOINT.exists():
            raise ValueError('Unexpected individual file')
    usable_train = [item for item in data['train'] if bool(pack([item])[3][0])]
    if len(usable_train) < 40 or not all(bool(pack([item])[3][0]) for item in data['dev'] + data['replay']):
        raise ValueError('Insufficient representable spans')
    head = ContextSpanSelector(usable_train[0]['features'].shape[-1], rank=saved['rank'])
    head.load_state_dict(saved['state'], strict=True)
    before = {key: evaluate(head, data[key]) for key in ('dev', *checks)}
    rng = random.Random(61532)
    rng.shuffle(usable_train)
    rng.shuffle(data['replay'])
    # Full-context examples use batch 2; synthetic replay uses batch 8.
    natural_batches = [pack(usable_train[i:i + 2]) for i in range(0, len(usable_train), 2)]
    replay_batches = [pack(data['replay'][i:i + 8]) for i in range(0, len(data['replay']), 8)]
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.00001, weight_decay=.01)
    for step in range(400):
        batches = natural_batches if step % 2 == 0 else replay_batches
        x, mask, target, usable = batches[(step // 2) % len(batches)]
        loss = F.cross_entropy(head(x[usable], mask[usable]), target[usable])
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    after = {key: evaluate(head, data[key]) for key in ('dev', *checks)}
    train_result = evaluate(head, usable_train)
    if any(runtime.sha(p) != hashes[k] for k, p in paths.items()):
        raise ValueError('Protected source changed')
    OUT.mkdir()
    torch.save({'state': head.state_dict(), 'rank': head.rank, 'source_sha256': hashes}, OUT / 'head.pt')
    result = {'before': before, 'after': after, 'train': train_result,
              'selected_train': len(train), 'representable_train': len(usable_train), 'replay': len(replay),
              'steps': 400, 'source_sha256': hashes, 'head_sha256': runtime.sha(OUT / 'head.pt'),
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Only positive natural QA labels; synthetic negatives are not natural dialogue.',
                  'Development examples previously inspected, not a sealed independent final evaluation.',
                  'Exact answer/title exclusions cannot prove absence of semantic or pretraining overlap.',
                  'No autonomous extraction, identity authentication, memory state, or end-to-end chat.']}
    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'before': before, 'after': after, 'train': train_result,
                      'representable_train': len(usable_train)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
