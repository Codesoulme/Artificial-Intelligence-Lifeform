"""P21 compositional wording benchmark; external labels never enter inference."""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
import torch
from scripts.reasoning_task import ROOT, label, changed_query, reversed_events
from scripts.ladder_task import tasks, sample
from spora.language_bridge import encode, PAD
from spora.short_reasoning import generate_equal_prefixes

# Exact questions are disjoint. Vocabulary and grammatical parts deliberately
# overlap: this tests composition, NOT arbitrary natural-language generalization.
PREFIXES = ('问:', '请问:', '问题:', '查询:')
SUFFIXES = ('现在哪?', '最后在哪?', '的位置?', '目前在哪?')
DEVELOPMENT = ((1, 2), (3, 1))
TEST = ((2, 3), (3, 2))
TRAIN = tuple((p, s) for p in range(4) for s in range(4)
              if (p, s) not in DEVELOPMENT + TEST)

def prompt(row, wording):
    records = ''.join(f'{i+1}:{obj}到{place};' for i, (obj, place) in enumerate(row['events']))
    question = (PREFIXES[wording[0]] + row['target'] + SUFFIXES[wording[1]]
                if isinstance(wording, (tuple, list)) else wording.format(target=row['target']))
    return '记录:' + records + question + '答:'

def batch(rows, wordings):
    sequences, starts = [], []
    if len(rows) != len(wordings) or not rows:
        raise ValueError('One wording per nonempty example required')
    for row, wording in zip(rows, wordings):
        prefix = encode(prompt(row, wording))[:-1]
        sequences.append(prefix + encode(row['answer'] + '。')[1:])
        starts.append(len(prefix))
    width = max(map(len, sequences))
    if width > 256: raise ValueError('Context exceeded')
    return torch.tensor([s + [PAD] * (width-len(s)) for s in sequences]), torch.tensor(starts)

@torch.no_grad()
def audit(model, state, rows, wording):
    groups = defaultdict(list); outputs = [None] * len(rows)
    for i, row in enumerate(rows):
        prefix = encode(prompt(row, wording))[:-1]
        groups[len(prefix)].append((i, prefix))
    for items in groups.values():
        for start in range(0, len(items), 16):
            part = items[start:start+16]
            generated = generate_equal_prefixes(model, state, [p for _, p in part], budget=16)
            for (i, _), output in zip(part, generated): outputs[i] = output
    details = []
    for row, output in zip(rows, outputs):
        # Independently reverse-scan events; do not trust precomputed labels.
        expected = next((p for o, p in reversed(row['events']) if o == row['target']), '未知')
        if expected != row['answer']: raise AssertionError('Invalid label')
        correct = output['utf8_valid'] and output['ended_with_eos'] and output['text'] == expected + '。'
        details.append(dict(prompt=prompt(row, wording), expected=expected, output=output, correct=correct))
    return dict(count=len(rows), accuracy=sum(d['correct'] for d in details)/len(rows), details=details)

def check(model, state, rows, wordings, details=False):
    results = {}
    for wording in wordings:
        base = audit(model, state, rows, wording)
        changed = audit(model, state, [changed_query(r) for r in rows], wording)
        reverse = audit(model, state, [reversed_events(r) for r in rows], wording)
        item = {'answer':base['accuracy'],
                'paired_query':sum(a['correct'] and b['correct'] for a,b in zip(base['details'],changed['details']))/len(rows)}
        for n in (1,2):
            indices = [i for i,r in enumerate(rows) if len(r['indices']) == n]
            if indices:
                item[f'paired_reversal_{n}'] = sum(base['details'][i]['correct'] and reverse['details'][i]['correct'] for i in indices)/len(indices)
        if details: item['records'] = dict(base=base, changed=changed, reverse=reverse)
        results[str(wording)] = item
    return results

def stress_rows():
    """Fixed development diagnostics, never a held-out test or training set."""
    rng = random.Random(310001); results = {}
    for n in (4, 6):
        rows = []
        for _ in range(64):
            pair = rng.sample(('杯','球','书','笔','包','盒'),2)
            events = [(rng.choice(pair),rng.choice(('桌','柜','箱','架'))) for _ in range(n)]
            events[0] = (pair[0],events[0][1]); events[1] = (pair[1],events[1][1])
            target = rng.choice(pair)
            rows.append(dict(events=events,target=target,pair=pair,**label(events,target)))
        results[f'records_{n}_repeated_locations'] = rows
    base = sample(tasks()['update']['development'],48,310002)
    results['unknown_object'] = [dict(r,target='灯',answer='未知',indices=[],latest=0) for r in base]
    results['long_question'] = base
    return results

def diagnostics(model, state):
    return {name:audit(model,state,rows,
        '请问:{target}经过这些移动之后现在位于什么地方?' if name=='long_question' else (0,0))
        for name,rows in stress_rows().items()}
