"""Counterbalance present/absent facts and distinguish name vs fact generalization."""
import itertools
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
from scripts.deepseek_fact_query_training_pilot_20261117 import evaluate
from scripts.deepseek_spora_consensus_reviewed_dialogue_audit import CONTEXT
from scripts.deepseek_spora_general_subject_locator import extract, pack
from scripts.deepseek_spora_interleaved_natural_transaction import HEADS, SOURCE, source_paths
from spora.research_context_span import ContextSpanSelector

OUT = ROOT / 'runs/deepseek-fact-crossed-pilot-20261117'
NAMES_TRAIN = (('小榆', '阿澄'), ('小楠', '阿芮'))
NAMES_HELD = (('小庄', '阿齐'), ('小芮', '阿谭'))
FACTS_TRAIN = ('喜欢喝茶', '喜欢蓝色', '喜欢跑步')
FACTS_HELD = ('喜欢喝果汁', '喜欢绿色', '喜欢爬山')


def cases(names, facts):
    rows = []
    for a, b in names:
        for first, second in itertools.permutations(facts, 2):
            absent = next(f for f in facts if f not in (first, second))
            for reverse in (False, True):
                clauses = [(a, first), (b, second)]
                if reverse:
                    clauses.reverse()
                evidence = '，'.join(person + fact for person, fact in clauses) + '。'
                for fact, person in ((first, a), (second, b), (absent, None)):
                    text = evidence + '谁' + fact + '？'
                    left = text.index(person) if person else None
                    rows.append({'text': text, 'label': 1 if person else None,
                                 'left': left, 'right': left + len(person) if person else None,
                                 'kind': 'known' if person else 'unknown', 'fact': fact})
    return rows


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch.manual_seed(7742)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    paths = {**source_paths(), 'context': CONTEXT / 'head.pt', 'report': CONTEXT / 'result.json'}
    hashes = {k: runtime.sha(p) for k, p in paths.items()}
    weights = torch.load(paths['context'], map_location='cpu', weights_only=True)
    report = json.loads(paths['report'].read_text(encoding='utf8'))
    if (hashes['base'] != runtime.WEIGHT_SHA or hashes['context'] != report['head_sha256']
            or any(weights['source_sha256'][k] != hashes[k] for k in source_paths())):
        raise ValueError('Source mismatch')
    groups = {'train': cases(NAMES_TRAIN, FACTS_TRAIN),
              'new_names': cases(NAMES_HELD, FACTS_TRAIN),
              'new_facts': cases(NAMES_TRAIN, FACTS_HELD)}
    seen = set()
    for rows in groups.values():
        texts = {r['text'] for r in rows}
        if len(texts) != len(rows) or seen & texts:
            raise ValueError('Duplicate or overlapping text')
        seen |= texts
    runtime.CHECKPOINT = SOURCE
    with tempfile.TemporaryDirectory(prefix='spora-fact-crossed-') as folder:
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
    random.Random(7742).shuffle(order)
    batches = [pack([data['train'][i] for i in order[start:start + 8]])
               for start in range(0, len(order), 8)]
    if any(not bool(batch[3].all()) for batch in batches):
        raise ValueError('Unrepresentable training span')
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.00002, weight_decay=.01)
    for step in range(300):
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
              'source_sha256': hashes, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Synthetic fixed syntax; development comparison only.',
                              'Names and base features may have appeared during prior development or pretraining.',
                              'No independent evaluation, identity verification, memory writes, or regression test.',
                              'Training recipe and data both changed from preceding pilot.']}
    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'before': before, 'after': after}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
