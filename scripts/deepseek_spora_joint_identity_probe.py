"""Train a research-only joint identity/address classifier with rejection.

Eight pre-enrolled identities are supervised. No dynamic enrollment, memory
read/write, or default-runtime change occurs in this experiment.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch import nn
from torch.nn import functional as F
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_named_binding_probe import NAMES, KEYS
from scripts.deepseek_observation_value_probe import pool

OUTPUT = ROOT / 'runs/deepseek-spora-joint-identity-probe-20261111'

# Text templates and negative transformations are disjoint across splits;
# enrolled identities are necessarily shared, so this is not unseen-person
# enrollment or a fully independent linguistic sample.
SPLITS = {
    'train': (('{n}现在喜欢喝什么？', '请查询{n}的饮料偏好。'),
              (('suffix', '哥'), ('suffix', '姐'), ('suffix', '仔'), ('prefix', '老'))),
    'calibration': (('我想查{n}爱喝哪种饮品。',),
                    (('suffix', '子'), ('prefix', '大'))),
    'held': (('{n}目前喜欢什么饮料？', '请说说{n}爱喝的饮品。'),
             (('suffix', '宁'), ('suffix', '明'), ('prefix', '陈'))),
}


def build(split):
    templates, transforms = SPLITS[split]
    rows = []
    for name, key in zip(NAMES, KEYS):
        for template in templates:
            rows.append({'text': template.format(n=name), 'expected_key': key, 'kind': 'known'})
            for side, affix in transforms:
                other = name + affix if side == 'suffix' else affix + name
                if other in NAMES:
                    raise ValueError('Negative happens to be an enrolled name')
                rows.append({'text': template.format(n=other), 'expected_key': None, 'kind': 'unknown'})
    return rows


def classify(logits, rows, threshold, unknown_index):
    with torch.no_grad():
        probabilities = logits.softmax(-1)
        scores, predictions = probabilities.max(-1)
    result = []
    for row, score, prediction in zip(rows, scores, predictions):
        predicted = int(prediction)
        accepted = predicted != unknown_index and float(score) >= threshold
        result.append({**row, 'predicted_class': predicted, 'score': float(score),
                       'accepted': accepted, 'correct_key': bool(accepted and row['kind'] == 'known'
                                                                   and predicted == row['expected_key'])})
    known = [r for r in result if r['kind'] == 'known']
    unknown = [r for r in result if r['kind'] == 'unknown']
    return {'known_total': len(known), 'known_correct': sum(r['correct_key'] for r in known),
            'known_wrong_accepted': sum(r['accepted'] and not r['correct_key'] for r in known),
            'known_rejected': sum(not r['accepted'] for r in known), 'unknown_total': len(unknown),
            'unknown_false_accept': sum(r['accepted'] for r in unknown), 'rows': result}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = {split: build(split) for split in SPLITS}
    for left in data:
        for right in data:
            if left != right and {r['text'] for r in data[left]} & {r['text'] for r in data[right]}:
                raise ValueError('Overlapping evaluation text')
    checkpoint_sha = runtime.sha(runtime.CHECKPOINT)
    torch.manual_seed(90122)
    torch.set_num_threads(2)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = runtime.load_chat()
        print('loaded runtime and isolated checkpoint', flush=True)
        unknown_index = chat.organism.config.key_count
        vectors = {}
        for split, rows in data.items():
            chunks = []
            for offset in range(0, len(rows), 12):
                questions = [r['text'] for r in rows[offset:offset + 12]]
                features, masks = runtime.extract(chat.base, chat.tokenizer, questions, batch_size=2)
                chunks.append(pool(features, masks).float())
                del features, masks
                print(split, min(offset + 12, len(rows)), '/', len(rows), flush=True)
            vectors[split] = torch.cat(chunks)
        model = nn.Linear(vectors['train'].shape[-1], unknown_index + 1)
        labels = torch.tensor([r['expected_key'] if r['expected_key'] is not None else unknown_index
                               for r in data['train']], dtype=torch.long)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
        for _ in range(500):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(vectors['train']), labels)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = {split: model(v) for split, v in vectors.items()}
            confidence = logits['calibration'].softmax(-1).max(-1).values
        thresholds = sorted({float(v) for v in confidence}) + [1.01]
        candidates = [(classify(logits['calibration'], data['calibration'], t, unknown_index), t)
                      for t in thresholds]
        feasible = [(res, t) for res, t in candidates
                    if res['unknown_false_accept'] == 0 and res['known_wrong_accepted'] == 0]
        _, threshold = max(feasible, key=lambda item: (item[0]['known_correct'], -item[1]))
        results = {split: classify(logits[split], rows, threshold, unknown_index)
                   for split, rows in data.items()}
    if runtime.sha(runtime.CHECKPOINT) != checkpoint_sha:
        raise ValueError('Protected bootstrap changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    path = OUTPUT / 'joint-identity.pt'
    torch.save({'format': 'spora-joint-identity-development-v1', 'state': model.state_dict(),
                'unknown_index': unknown_index, 'threshold': threshold,
                'source_checkpoint_sha256': checkpoint_sha}, path)
    report = {'status': 'research_only_no_writes', 'checkpoint_sha256': runtime.sha(path),
              'source_checkpoint_sha256': checkpoint_sha, 'threshold': threshold,
              'unknown_index': unknown_index, 'results': results, 'seconds': time.perf_counter() - started,
              'limitations': ['Eight supervised enrolled names; fixed address classes, no dynamic enrollment.',
                              'Synthetic wording and negative transformations; same names across splits.',
                              'Small calibration sample; thresholds do not certify identity or reliability.',
                              'A correct class does not prove source trust, fact accuracy or write authorization.',
                              'Experimental model is not connected to default runtime or user state.']}
    (OUTPUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for split, result in results.items():
        print(split, {key: value for key, value in result.items() if key != 'rows'}, flush=True)


if __name__ == '__main__':
    main()
