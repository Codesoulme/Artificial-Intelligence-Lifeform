"""Isolated training with same-relation competing values and old-task replay."""
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
from scripts.deepseek_fact_crossed_pilot_20261117 import cases, FACTS_TRAIN, FACTS_HELD
from scripts.deepseek_fact_crossed_names8_20261117 import NAMES
from scripts.deepseek_fact_crossed_stress_20261117 import cases as stress_cases
from scripts.deepseek_fact_query_training_pilot_20261117 import evaluate
from scripts.deepseek_spora_general_subject_locator import extract, pack
from scripts.deepseek_spora_interleaved_natural_transaction import HEADS, SOURCE, source_paths
from spora.research_context_span import ContextSpanSelector

OUT = ROOT / 'runs/deepseek-fact-competing-train-20261117'
BASE = ROOT / 'runs/deepseek-fact-crossed-names8-1200-20261117/head.pt'
TRAIN_NAMES = NAMES[:4]
TEST_NAMES = (('小庄', '阿齐'), ('小霜', '阿岳'))
TRAIN_SAME = ('喜欢喝茶', '喜欢喝咖啡', '喜欢喝果汁')
HELD_SAME = ('喜欢喝牛奶', '喜欢喝豆浆', '喜欢喝可乐')


def build_groups():
    groups = {'train': cases(TRAIN_NAMES, FACTS_TRAIN) + cases(TRAIN_NAMES, TRAIN_SAME),
              'old_new_names': cases(TEST_NAMES, FACTS_TRAIN),
              'old_new_values': cases(TEST_NAMES, FACTS_HELD),
              'same_new_names': cases(TEST_NAMES, TRAIN_SAME),
              'same_new_values': cases(TEST_NAMES, HELD_SAME)}
    groups.update({'stress_' + name: rows for name, rows in stress_cases().items()})
    training_texts = {r['text'] for r in groups['train']}
    if len(training_texts) != len(groups['train']) or any(
            training_texts & {r['text'] for r in rows} for k, rows in groups.items() if k != 'train'):
        raise ValueError('Training overlap')
    return groups


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch.manual_seed(7743)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    paths = {**source_paths(), 'prior': BASE}
    hashes = {k: runtime.sha(p) for k, p in paths.items()}
    weights = torch.load(paths['prior'], map_location='cpu', weights_only=True)
    if (hashes['base'] != runtime.WEIGHT_SHA or
            any(weights['source_sha256'][k] != hashes[k] for k in source_paths())):
        raise ValueError('Source mismatch')
    groups = build_groups()
    runtime.CHECKPOINT = SOURCE
    with tempfile.TemporaryDirectory(prefix='spora-competing-facts-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'not-created.pt'
        chat = runtime.load_chat(natural_write_heads_path=HEADS)
        data = {name: extract(chat, rows) for name, rows in groups.items()}
        if runtime.WORKING_CHECKPOINT.exists():
            raise ValueError('Unexpected individual file')
    width = data['train'][0]['features'].shape[-1]
    head = ContextSpanSelector(width, rank=weights['rank'])
    head.load_state_dict(weights['state'], strict=True)
    before = {name: evaluate(head, rows) for name, rows in data.items() if name != 'train'}
    order = list(range(len(data['train'])))
    random.Random(7743).shuffle(order)
    batches = [pack([data['train'][i] for i in order[start:start + 8]])
               for start in range(0, len(order), 8)]
    if any(not bool(batch[3].all()) for batch in batches):
        raise ValueError('Unrepresentable training span')
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.00002, weight_decay=.01)
    for step in range(1200):
        x, mask, target, usable = batches[step % len(batches)]
        loss = F.cross_entropy(head(x[usable], mask[usable]), target[usable])
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    after = {name: evaluate(head, rows) for name, rows in data.items()}
    if any(runtime.sha(p) != hashes[k] for k, p in paths.items()):
        raise ValueError('Protected source changed')
    OUT.mkdir()
    torch.save({'state': head.state_dict(), 'rank': head.rank, 'source_sha256': hashes}, OUT / 'head.pt')
    result = {'before': before, 'after': after, 'counts': {k: len(v) for k, v in groups.items()},
              'source_sha256': hashes, 'head_sha256': runtime.sha(OUT / 'head.pt'),
              'steps': 1200, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Artificial fixed syntax; repeated templates are not independent cases.',
                  'Stress examples and some test names were previously examined, not sealed tests.',
                  'Pretraining overlap unknown; no natural conversation or memory transaction.',
                  'Weights started from a prior trained pilot, not the original base head.']}
    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'before': before, 'after': after}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
