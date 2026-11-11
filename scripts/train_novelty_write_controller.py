"""Train and compare a novelty-aware write controller on synthetic policy traces."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from spora.internal_memory import ACTIONS, NoveltyWriteControl, WriteControl
from spora.observation_surprise import observation_surprise

OUTPUT = ROOT / 'runs/novelty-write-controller-train-20261111'


def make_rows(count, generator):
    rows = []
    for _ in range(count):
        source = int(torch.randint(0, 4, (), generator=generator))
        known = bool(torch.randint(0, 2, (), generator=generator))
        old_value = int(torch.randint(0, 8, (), generator=generator))
        new_value = old_value if bool(torch.randint(0, 2, (), generator=generator)) else int(torch.randint(0, 8, (), generator=generator))
        if not known:
            old_value, new_value = 16, int(torch.randint(0, 8, (), generator=generator))
        old_quality = float(torch.rand((), generator=generator)) if known else 0.0
        new_quality = float(torch.rand((), generator=generator))
        old_time = int(torch.randint(-1, 8, (), generator=generator)) if known else -1
        new_time = int(torch.randint(0, 9, (), generator=generator))
        freshness = float((new_time > old_time) - (new_time < old_time))
        same = float(known and old_value == new_value)
        surprise = observation_surprise(known=known, old_value=old_value, new_value=new_value,
                                        old_quality=old_quality, new_quality=new_quality,
                                        old_time=old_time, new_time=new_time)
        feature = [float(source == index) for index in range(4)] + [new_quality, float(known), same,
                  old_quality, freshness, float(known and old_time >= 0), surprise['score']]
        if source >= 2:
            action = 4
        elif not known:
            action = 0
        elif old_value != new_value:
            action = 1
        elif new_quality > old_quality:
            action = 2
        else:
            action = 3
        rows.append((feature, action))
    features = torch.tensor([row[0] for row in rows], dtype=torch.float32)
    labels = torch.tensor([row[1] for row in rows], dtype=torch.long)
    return features, labels


def train(model, features, labels, steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(features), labels)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite controller loss')
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(features).argmax(-1)
    return {'accuracy': float((prediction == labels).float().mean()),
            'correct': int((prediction == labels).sum()), 'total': len(labels), 'steps': steps}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    generator = torch.Generator().manual_seed(90120)
    train_features, train_labels = make_rows(12000, generator)
    held_features, held_labels = make_rows(4000, generator)
    novelty = NoveltyWriteControl(hidden=64, confidence=.8)
    legacy = WriteControl(hidden=64, confidence=.8)
    novelty_train = train(novelty, train_features, train_labels, 500)
    legacy_train = train(legacy, train_features[:, :10], train_labels, 500)
    with torch.no_grad():
        novelty_pred = novelty(held_features).argmax(-1)
        legacy_pred = legacy(held_features[:, :10]).argmax(-1)
    novelty_held = {'accuracy': float((novelty_pred == held_labels).float().mean()),
                    'correct': int((novelty_pred == held_labels).sum()), 'total': len(held_labels)}
    legacy_held = {'accuracy': float((legacy_pred == held_labels).float().mean()),
                   'correct': int((legacy_pred == held_labels).sum()), 'total': len(held_labels)}
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT / 'novelty-control.pt'
    torch.save({'format': 'spora-novelty-write-control-development-v1',
                'state': novelty.state_dict(), 'hidden': novelty.hidden, 'confidence': novelty.confidence,
                'input_width': 11, 'actions': ACTIONS}, checkpoint)
    report = {'status': 'research_only_controller_candidate', 'controller_sha256': __import__('hashlib').sha256(checkpoint.read_bytes()).hexdigest(),
              'train': {'novelty': novelty_train, 'legacy': legacy_train},
              'held': {'novelty': novelty_held, 'legacy': legacy_held},
              'dataset': {'train': len(train_labels), 'held': len(held_labels), 'seed': 90120},
              'seconds': time.perf_counter() - started,
              'limitations': ['Synthetic policy traces with labels defined by the intended write protocol.',
                              'The held split is generated from the same rule distribution.',
                              'No language parsing, source verification or real individual was used.',
                              'Candidate weights are not connected to a checkpoint.']}
    (OUTPUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['held'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    started = time.perf_counter()
    main()
