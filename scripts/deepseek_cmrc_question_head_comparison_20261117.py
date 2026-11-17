"""Matched-budget comparison of pooled and explicit-question span interaction."""
import copy
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
from scripts.deepseek_cmrc_reading_train_pilot_20261117 import choose_replay
from scripts.deepseek_fact_competing_train_20261117 import build_groups
from scripts.deepseek_fact_head_natural_regression_20261117 import CANDIDATE
from scripts.deepseek_spora_general_subject_locator import extract, pack
from scripts.deepseek_spora_interleaved_natural_transaction import HEADS, SOURCE, source_paths
from scripts.prepare_cmrc_train_pilot_20261117 import TRAIN, TRAIN_BLOB, OUT as TRAIN_MANIFEST, select_train
from spora.research_context_span import ContextSpanSelector
from spora.research_question_conditioned_span import QuestionConditionedSpan

OUT = ROOT / 'runs/deepseek-cmrc-question-head-comparison-20261117'


def question_start(text):
    marker = '\n问题：'
    if marker in text:
        return text.rfind(marker) + len(marker)
    last = text.rfind('。')
    if last < 0 or last + 1 >= len(text):
        raise ValueError('Missing explicit question boundary')
    return last + 1


def question_tokens(item):
    start = item['start'] + question_start(item['row']['text'])
    end = item['start'] + len(item['row']['text'])
    offsets = item['offsets']
    mask = (item['subject_mask'] & (offsets[:, 0] >= start)
            & (offsets[:, 1] <= end) & (offsets[:, 1] > offsets[:, 0]))
    if not bool(mask.any()):
        raise ValueError('Question has no valid tokens')
    return mask


def batch(items):
    x, mask, target, usable = pack(items)
    q = torch.zeros_like(mask)
    for i, item in enumerate(items):
        q[i, :len(item['mask'])] = question_tokens(item)
    return x, mask, q, target, usable


def assess(head, rows):
    head.eval()
    result = {'known': {'total': 0, 'exact': 0, 'rejected': 0, 'wrong': 0},
              'unknown': {'total': 0, 'rejected': 0, 'selected': 0}}
    with torch.inference_mode():
        for i in range(0, len(rows), 2):
            items = rows[i:i + 2]
            x, mask, q, target, usable = batch(items)
            if not bool(usable.all()):
                raise ValueError('Unrepresentable evaluation span')
            choices = head(x, mask, q).argmax(-1).tolist()
            for item, choice, gold in zip(items, choices, target.tolist()):
                entry = result[item['row']['kind']]
                entry['total'] += 1
                if item['row']['kind'] == 'known':
                    entry['exact' if choice == gold else 'rejected' if choice == 0 else 'wrong'] += 1
                else:
                    entry['rejected' if choice == 0 else 'selected'] += 1
    return result


class Baseline(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, x, mask, question_mask):
        return self.head(x, mask)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    start = time.perf_counter()
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
    for rows, path in ((train, TRAIN_MANIFEST), (dev, DEV_MANIFEST)):
        if [r['digest'] for r in rows] != [r['digest'] for r in json.loads(path.read_text(encoding='utf8'))['rows']]:
            raise ValueError('Manifest mismatch')
    saved = torch.load(paths['prior'], map_location='cpu', weights_only=True)
    report = json.loads(paths['prior_report'].read_text(encoding='utf8'))
    if (hashes['base'] != runtime.WEIGHT_SHA or hashes['prior'] != report['head_sha256']
            or any(saved['source_sha256'][k] != hashes[k] for k in source_paths())):
        raise ValueError('Head mismatch')
    artificial = build_groups()
    groups = {'train': train, 'dev': dev, 'replay': choose_replay(artificial['train']),
              'synthetic_same': artificial['stress_same_type'], 'synthetic_homonym': artificial['stress_homonym']}
    runtime.CHECKPOINT = SOURCE
    with tempfile.TemporaryDirectory(prefix='spora-question-head-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'not-created.pt'
        chat = runtime.load_chat(natural_write_heads_path=HEADS)
        data = {}
        for name, rows in groups.items():
            data[name] = []
            for row in rows:
                data[name].extend(extract(chat, [row]))
        if runtime.WORKING_CHECKPOINT.exists():
            raise ValueError('Unexpected individual file')
    train_items = [item for item in data['train'] if bool(pack([item])[3][0])]
    if len(train_items) < 40:
        raise ValueError('Insufficient representable examples')
    width = train_items[0]['features'].shape[-1]
    origin = ContextSpanSelector(width, rank=saved['rank'])
    origin.load_state_dict(saved['state'], strict=True)
    heads = {'baseline': Baseline(copy.deepcopy(origin)),
             'question': QuestionConditionedSpan(copy.deepcopy(origin), width, rank=32)}
    x, mask, q, _, _ = batch(data['dev'][:2])
    with torch.inference_mode():
        if not torch.equal(heads['baseline'](x, mask, q), heads['question'](x, mask, q)):
            raise ValueError('Initial output mismatch')
    check_names = ('dev', 'synthetic_same', 'synthetic_homonym')
    before = {name: {k: assess(head, data[k]) for k in check_names} for name, head in heads.items()}
    rng = random.Random(61532)
    rng.shuffle(train_items)
    rng.shuffle(data['replay'])
    natural_batches = [batch(train_items[i:i + 2]) for i in range(0, len(train_items), 2)]
    replay_batches = [batch(data['replay'][i:i + 8]) for i in range(0, len(data['replay']), 8)]
    for name, head in heads.items():
        head.train()
        optimizer = torch.optim.AdamW(head.parameters(), lr=.00001, weight_decay=.01)
        for step in range(400):
            batches = natural_batches if step % 2 == 0 else replay_batches
            x, mask, q, target, usable = batches[(step // 2) % len(batches)]
            loss = F.cross_entropy(head(x[usable], mask[usable], q[usable]), target[usable])
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(json.dumps({'trained': name}), flush=True)
    after = {name: {k: assess(head, data[k]) for k in check_names} for name, head in heads.items()}
    train_score = {name: assess(head, train_items) for name, head in heads.items()}
    if any(runtime.sha(p) != hashes[k] for k, p in paths.items()):
        raise ValueError('Protected source changed')
    OUT.mkdir()
    for name, head in heads.items():
        torch.save({'state': head.state_dict(), 'source_sha256': hashes}, OUT / (name + '.pt'))
    result = {'before': before, 'after': after, 'train': train_score, 'representable_train': len(train_items),
              'steps_per_head': 400, 'source_sha256': hashes,
              'candidate_sha256': {name: runtime.sha(OUT / (name + '.pt')) for name in heads},
              'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Candidate has more parameters and per-step compute, not a FLOP-matched comparison.',
                  'Question masks use known input formatting, not automatic task detection.',
                  'Small previously examined development set; no natural unanswerable labels.',
                  'No actual memory read, natural write, identity authentication, or end-to-end chat.']}
    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'after': after, 'train': train_score}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
