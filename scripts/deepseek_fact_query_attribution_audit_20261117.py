"""Isolated counterfactual audit of fact-conditioned subject selection."""
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_spora_consensus_reviewed_dialogue_audit import CONTEXT, chosen
from scripts.deepseek_spora_general_subject_locator import extract, pack
from scripts.deepseek_spora_interleaved_natural_transaction import HEADS, SOURCE, source_paths
from scripts.deepseek_spora_joint_span_probe import SpanSelector
from spora.research_context_span import ContextSpanSelector

OUT = ROOT / 'runs/deepseek-fact-query-attribution-audit-20261117.json'
NAMES = (('小岚', '阿祺'), ('小岑', '阿淳'), ('小霖', '阿彤'),
         ('小岚岚', '阿祺祺'), ('小岑岑', '阿淳淳'), ('小霖霖', '阿彤彤'))


def make_cases():
    rows = []
    for pair, (a, b) in enumerate(NAMES):
        for swapped in (False, True):
            drink, color = (b, a) if swapped else (a, b)
            text = f'{drink}喜欢喝茶，{color}喜欢蓝色。'
            for query, target in (('谁喜欢喝茶？', drink), ('谁喜欢蓝色？', color), ('谁喜欢跑步？', None)):
                full = text + query
                left = full.index(target) if target else None
                rows.append({'text': full, 'label': 1 if target else None,
                             'left': left, 'right': left + len(target) if target else None,
                             'case': f'{pair}-{int(swapped)}-{query}', 'pair': pair,
                             'swapped': swapped, 'kind': 'unknown' if target is None else
                             'drink' if target == drink else 'color'})
    return rows


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    paths = {**source_paths(), 'linear': CONTEXT / 'linear-head.pt',
             'context': CONTEXT / 'head.pt', 'report': CONTEXT / 'result.json'}
    hashes = {name: runtime.sha(path) for name, path in paths.items()}
    context = torch.load(paths['context'], map_location='cpu', weights_only=True)
    linear = torch.load(paths['linear'], map_location='cpu', weights_only=True)
    report = json.loads(paths['report'].read_text(encoding='utf8'))
    if (hashes['base'] != runtime.WEIGHT_SHA or hashes['context'] != report['head_sha256']
            or any(context['source_sha256'][key] != hashes[key] for key in source_paths())
            or any(linear['source_sha256'][key] != hashes[key] for key in source_paths())):
        raise ValueError('Source mismatch')
    runtime.CHECKPOINT = SOURCE
    with tempfile.TemporaryDirectory(prefix='spora-fact-query-') as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'not-created.pt'
        chat = runtime.load_chat(natural_write_heads_path=HEADS)
        items = extract(chat, make_cases())
        if runtime.WORKING_CHECKPOINT.exists():
            raise ValueError('Unexpected individual file')
    width = items[0]['features'].shape[-1]
    heads = {'linear': SpanSelector(width), 'context': ContextSpanSelector(width, rank=context['rank'])}
    heads['linear'].load_state_dict(linear['state'], strict=True)
    heads['context'].load_state_dict(context['state'], strict=True)
    for head in heads.values():
        head.eval()
    results = []
    for item in items:
        x, _, _, usable = pack([item])
        if item['row']['label'] is not None and not bool(usable[0]):
            raise ValueError('Authored gold span cannot be represented')
        result = {k: item['row'][k] for k in ('case', 'pair', 'swapped', 'kind')}
        result['digest'] = hashlib.sha256(item['row']['text'].encode('utf8')).hexdigest()
        result['heads'] = {}
        for name, head in heads.items():
            selection = chosen(head, item, x.shape[1])
            any_selected = bool(selection.any())
            result['heads'][name] = {'selected': any_selected,
                'exact': bool(torch.equal(selection, item['gold'])) if item['row']['label'] is not None else None}
        results.append(result)
    summary = {}
    for kind in ('drink', 'color', 'unknown'):
        subset = [r for r in results if r['kind'] == kind]
        summary[kind] = {'cases': len(subset), **{name: {
            'selected': sum(r['heads'][name]['selected'] for r in subset),
            'exact': sum(r['heads'][name]['exact'] is True for r in subset) if kind != 'unknown' else None}
            for name in heads}}
    if any(runtime.sha(path) != hashes[name] for name, path in paths.items()):
        raise ValueError('Protected source changed')
    result = {'summary': summary, 'rows': results, 'source_sha256': hashes,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Synthetic small controlled development cases; not natural dialogue.',
                  'Outputs are mention spans only; no actual identity binding or memory writes.',
                  'Names and base pretraining overlap not exhaustively checked.',
                  'No natural false-write rate or general chat conclusion.']}
    with OUT.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
