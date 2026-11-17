"""Train an isolated span head on translated answerable/unanswerable QA plus replay."""
import hashlib
import json
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import pyarrow.parquet as parquet
import torch
from torch.nn import functional as F
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.audit_chinese_squadv2_20261117 import SOURCE as SQUAD_DEV, SHA as SQUAD_DEV_SHA
from scripts.audit_cmrc2018_dev_20261117 import SOURCE as CMRC_DEV, BLOB_SHA1 as CMRC_DEV_BLOB, blob_sha1
from scripts.cmrc_who_development_selection_20261117 import OUT as CMRC_DEV_MANIFEST, select as select_cmrc_dev
from scripts.deepseek_cmrc_reading_train_pilot_20261117 import choose_replay
from scripts.deepseek_fact_competing_train_20261117 import build_groups
from scripts.deepseek_fact_query_training_pilot_20261117 import evaluate
from scripts.deepseek_spora_general_subject_locator import extract, pack
from scripts.deepseek_spora_interleaved_natural_transaction import HEADS, SOURCE, source_paths
from scripts.prepare_cmrc_train_pilot_20261117 import TRAIN as CMRC_TRAIN, TRAIN_BLOB as CMRC_TRAIN_BLOB, OUT as CMRC_MANIFEST, select_train as select_cmrc_train
from scripts.prepare_squadv2_train_pilot_20261117 import TRAIN as SQUAD_TRAIN, TRAIN_SHA as SQUAD_TRAIN_SHA, OUT as SQUAD_MANIFEST, select_train as select_squad_train
from scripts.select_chinese_squadv2_who_20261117 import OUT as SQUAD_DEV_MANIFEST, select as select_squad_dev
from spora.research_context_span import ContextSpanSelector

OUT = ROOT / 'runs/deepseek-squadv2-mixed-training-20261117'
PRIOR = ROOT / 'runs/deepseek-cmrc-reading-train-pilot-20261117'


def check_manifest(rows, path):
    manifest = json.loads(path.read_text(encoding='utf8'))
    if [r['digest'] for r in rows] != [r['digest'] for r in manifest['rows']]:
        raise ValueError('Manifest mismatch')


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch.manual_seed(61533)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    paths = {**source_paths(), 'squad_train': SQUAD_TRAIN, 'squad_dev': SQUAD_DEV,
             'cmrc_train': CMRC_TRAIN, 'cmrc_dev': CMRC_DEV, 'squad_manifest': SQUAD_MANIFEST,
             'squad_dev_manifest': SQUAD_DEV_MANIFEST, 'cmrc_manifest': CMRC_MANIFEST,
             'cmrc_dev_manifest': CMRC_DEV_MANIFEST, 'prior': PRIOR / 'head.pt', 'prior_report': PRIOR / 'result.json'}
    hashes = {k: runtime.sha(p) for k, p in paths.items()}
    if (hashes['base'] != runtime.WEIGHT_SHA or hashes['squad_train'] != SQUAD_TRAIN_SHA
            or hashes['squad_dev'] != SQUAD_DEV_SHA or blob_sha1(CMRC_TRAIN.read_bytes()) != CMRC_TRAIN_BLOB
            or blob_sha1(CMRC_DEV.read_bytes()) != CMRC_DEV_BLOB):
        raise ValueError('Dataset mismatch')
    squad_train_rows = parquet.read_table(SQUAD_TRAIN).to_pylist()
    squad_dev_rows = parquet.read_table(SQUAD_DEV).to_pylist()
    cmrc_train_raw, cmrc_dev_raw = json.loads(CMRC_TRAIN.read_text(encoding='utf8')), json.loads(CMRC_DEV.read_text(encoding='utf8'))
    squad_train, _ = select_squad_train(squad_train_rows, squad_dev_rows)
    squad_dev, _ = select_squad_dev(squad_dev_rows)
    cmrc_train, _ = select_cmrc_train(cmrc_train_raw, cmrc_dev_raw)
    cmrc_dev, _ = select_cmrc_dev(cmrc_dev_raw)
    for rows, manifest in ((squad_train, SQUAD_MANIFEST), (squad_dev, SQUAD_DEV_MANIFEST),
                           (cmrc_train, CMRC_MANIFEST), (cmrc_dev, CMRC_DEV_MANIFEST)):
        check_manifest(rows, manifest)
    saved = torch.load(paths['prior'], map_location='cpu', weights_only=True)
    report = json.loads(paths['prior_report'].read_text(encoding='utf8'))
    if (hashes['prior'] != report['head_sha256']
            or any(saved['source_sha256'][k] != hashes[k] for k in source_paths())):
        raise ValueError('Prior mismatch')
    artificial = build_groups()
    groups = {'squad_train': squad_train, 'squad_dev': squad_dev, 'cmrc_train': cmrc_train,
              'cmrc_dev': cmrc_dev, 'replay': choose_replay(artificial['train']),
              'synthetic_same': artificial['stress_same_type'], 'synthetic_homonym': artificial['stress_homonym']}
    runtime.CHECKPOINT = SOURCE
    with tempfile.TemporaryDirectory(prefix='spora-mixed-qa-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'not-created.pt'
        chat = runtime.load_chat(natural_write_heads_path=HEADS)
        data = {}
        for name, rows in groups.items():
            data[name] = []
            for row in rows:
                data[name].extend(extract(chat, [row]))
        if runtime.WORKING_CHECKPOINT.exists():
            raise ValueError('Unexpected individual file')
    squad_usable = [item for item in data['squad_train'] if bool(pack([item])[3][0])]
    cmrc_usable = [item for item in data['cmrc_train'] if bool(pack([item])[3][0])]
    if len(squad_usable) < 80 or len(cmrc_usable) < 40:
        raise ValueError('Insufficient representable samples')
    head = ContextSpanSelector(cmrc_usable[0]['features'].shape[-1], rank=saved['rank'])
    head.load_state_dict(saved['state'], strict=True)
    checks = ('squad_dev', 'cmrc_dev', 'synthetic_same', 'synthetic_homonym')
    before = {k: evaluate(head, data[k]) for k in checks}
    rng = random.Random(61533)
    rng.shuffle(squad_usable)
    rng.shuffle(cmrc_usable)
    rng.shuffle(data['replay'])
    batch_groups = [[pack(items[i:i + size]) for i in range(0, len(items), size)]
                    for items, size in ((squad_usable, 2), (cmrc_usable, 2), (data['replay'], 8))]
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.00001, weight_decay=.01)
    for step in range(600):
        batches = batch_groups[step % len(batch_groups)]
        x, mask, target, usable = batches[(step // len(batch_groups)) % len(batches)]
        loss = F.cross_entropy(head(x[usable], mask[usable]), target[usable])
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    after = {k: evaluate(head, data[k]) for k in checks}
    train_scores = {'squad': evaluate(head, squad_usable), 'cmrc': evaluate(head, cmrc_usable)}
    if any(runtime.sha(p) != hashes[k] for k, p in paths.items()):
        raise ValueError('Protected source changed')
    OUT.mkdir()
    torch.save({'state': head.state_dict(), 'rank': head.rank, 'source_sha256': hashes}, OUT / 'head.pt')
    result = {'before': before, 'after': after, 'train': train_scores,
              'representable': {'squad': len(squad_usable), 'cmrc': len(cmrc_usable)},
              'steps': 600, 'source_sha256': hashes, 'head_sha256': runtime.sha(OUT / 'head.pt'),
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Machine-translated no-answer labels not independently verified.',
                  'Exact article title and context exclusion does not prove semantic non-overlap.',
                  'Development datasets already inspected and tuned; no sealed final test.',
                  'No real identities, user memories, persistence, or full answer generation.']}
    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'before': before, 'after': after, 'train': train_scores}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
