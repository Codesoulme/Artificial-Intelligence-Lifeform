"""Matched isolated head training on fact-conditioned attribution; synthetic pilot."""
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch.nn import functional as F
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_spora_consensus_reviewed_dialogue_audit import CONTEXT
from scripts.deepseek_spora_general_subject_locator import extract, pack
from scripts.deepseek_spora_interleaved_natural_transaction import HEADS, SOURCE, source_paths
from scripts.deepseek_spora_joint_span_probe import SpanSelector
from spora.research_context_span import ContextSpanSelector

OUT = ROOT / 'runs/deepseek-fact-query-training-pilot-20261117'
TRAIN_NAMES = (('小榆', '阿澄'), ('小楠', '阿芮'), ('小遥', '阿卓'), ('小夏', '阿冬'))
HELD_NAMES = (('小岚', '阿祺'), ('小岑', '阿淳'))
TRAIN_FACTS = (('喜欢喝茶', '喜欢蓝色', '喜欢跑步'), ('喜欢喝咖啡', '喜欢红色', '喜欢游泳'))
HELD_FACTS = (('喜欢喝果汁', '喜欢绿色', '喜欢爬山'),)


def cases(names, facts):
    result = []
    for a, b in names:
        for first, second, unknown in facts:
            for swapped in (False, True):
                p, q = (b, a) if swapped else (a, b)
                for reverse in (False, True):
                    clauses = [(p, first), (q, second)]
                    if reverse:
                        clauses.reverse()
                    evidence = '，'.join(person + fact for person, fact in clauses) + '。'
                    for queried, target, kind in ((first, p, 'known'), (second, q, 'known'),
                                                  (unknown, None, 'unknown')):
                        text = evidence + '谁' + queried + '？'
                        left = text.index(target) if target else None
                        result.append({'text': text, 'label': 1 if target else None,
                                       'left': left, 'right': left + len(target) if target else None,
                                       'kind': kind})
    return result


def evaluate(head, items):
    counts = {'known': {'total': 0, 'exact': 0, 'rejected': 0, 'wrong': 0},
              'unknown': {'total': 0, 'rejected': 0, 'selected': 0}}
    head.eval()
    with torch.inference_mode():
        for offset in range(0, len(items), 8):
            subset = items[offset:offset + 8]
            x, mask, target, usable = pack(subset)
            if not bool(usable.all()):
                raise ValueError('Unrepresentable gold span')
            choices = head(x, mask).argmax(dim=-1)
            for item, choice, gold in zip(subset, choices.tolist(), target.tolist()):
                group = counts[item['row']['kind']]
                group['total'] += 1
                if item['row']['kind'] == 'known':
                    group['exact' if choice == gold else 'rejected' if choice == 0 else 'wrong'] += 1
                else:
                    group['rejected' if choice == 0 else 'selected'] += 1
    return counts


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    start = time.perf_counter()
    torch.manual_seed(7741)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    paths = {**source_paths(), 'linear': CONTEXT / 'linear-head.pt',
             'context': CONTEXT / 'head.pt', 'report': CONTEXT / 'result.json'}
    hashes = {key: runtime.sha(path) for key, path in paths.items()}
    old_linear = torch.load(paths['linear'], map_location='cpu', weights_only=True)
    old_context = torch.load(paths['context'], map_location='cpu', weights_only=True)
    report = json.loads(paths['report'].read_text(encoding='utf8'))
    if (hashes['base'] != runtime.WEIGHT_SHA or hashes['context'] != report['head_sha256']
            or any(old_linear['source_sha256'][k] != hashes[k] for k in source_paths())
            or any(old_context['source_sha256'][k] != hashes[k] for k in source_paths())):
        raise ValueError('Source mismatch')
    train = cases(TRAIN_NAMES, TRAIN_FACTS)
    held = cases(HELD_NAMES, HELD_FACTS)
    if set(r['text'] for r in train) & set(r['text'] for r in held):
        raise ValueError('Text overlap')
    runtime.CHECKPOINT = SOURCE
    with tempfile.TemporaryDirectory(prefix='spora-fact-training-') as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'not-created.pt'
        chat = runtime.load_chat(natural_write_heads_path=HEADS)
        train_items = extract(chat, train)
        held_items = extract(chat, held)
        if runtime.WORKING_CHECKPOINT.exists():
            raise ValueError('Unexpected individual file')
    width = train_items[0]['features'].shape[-1]
    heads = {'linear': SpanSelector(width), 'context': ContextSpanSelector(width, rank=old_context['rank'])}
    heads['linear'].load_state_dict(old_linear['state'], strict=True)
    heads['context'].load_state_dict(old_context['state'], strict=True)
    before = {name: evaluate(head, held_items) for name, head in heads.items()}
    batches = [pack(train_items[i:i + 8]) for i in range(0, len(train_items), 8)]
    if any(not bool(batch[3].all()) for batch in batches):
        raise ValueError('Unrepresentable training span')
    initial = {name: copy.deepcopy(head.state_dict()) for name, head in heads.items()}
    for head in heads.values():
        head.train()
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.00002, weight_decay=0.01)
        for step in range(150):
            x, mask, target, usable = batches[step % len(batches)]
            loss = F.cross_entropy(head(x[usable], mask[usable]), target[usable])
            if not bool(torch.isfinite(loss)):
                raise ValueError('Non-finite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    after = {name: evaluate(head, held_items) for name, head in heads.items()}
    train_score = {name: evaluate(head, train_items) for name, head in heads.items()}
    if any(runtime.sha(path) != hashes[key] for key, path in paths.items()):
        raise ValueError('Protected source changed')
    OUT.mkdir()
    for name, head in heads.items():
        torch.save({'state': head.state_dict(), 'initial': initial[name], 'source_sha256': hashes},
                   OUT / (name + '.pt'))
    result = {'before': before, 'after': after, 'train': train_score,
              'counts': {'train': len(train), 'held': len(held)}, 'source_sha256': hashes,
              'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Synthetic and small: same question syntax, no natural dialogue.',
                  'Prior weights were trained and selected on previously examined examples.',
                  'Not a matched random initialization comparison; same fine-tuning budget only.',
                  'No independent calibration, prior ability regression, writes, or deployment.']}
    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result['after'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
