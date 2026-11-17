"""Diagnose no-answer margins using a separate article-disjoint calibration subset."""
import json
import math
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import pyarrow.parquet as parquet
import torch
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.audit_chinese_squadv2_20261117 import SOURCE as DATA, SHA
from scripts.deepseek_spora_general_subject_locator import extract, pack
from scripts.deepseek_spora_interleaved_natural_transaction import HEADS, SOURCE, source_paths
from scripts.deepseek_spora_natural_write_span_diagnostic import selected_text
from scripts.select_chinese_squadv2_who_20261117 import OUT as DEV_MANIFEST, select as select_dev
from scripts.squadv2_calibration_development_20261117 import OUT as CAL_MANIFEST, select_calibration
from spora.research_context_span import ContextSpanSelector

OUT = ROOT / 'runs/deepseek-squadv2-margin-calibration-20261117.json'
CANDIDATE = ROOT / 'runs/deepseek-squadv2-mixed-training-20261117'


def measure(rows, threshold):
    result = {'known': {'total': 0, 'exact': 0, 'wrong': 0, 'rejected': 0},
              'unknown': {'total': 0, 'selected': 0, 'rejected': 0}}
    for row in rows:
        entry = result[row['kind']]
        entry['total'] += 1
        selected = row['margin'] > threshold
        if row['kind'] == 'known':
            entry['exact' if selected and row['best_exact'] else 'wrong' if selected else 'rejected'] += 1
        else:
            entry['selected' if selected else 'rejected'] += 1
    return result


def choose_threshold(rows, max_unknown_selected):
    if not rows or not any(r['kind'] == 'known' for r in rows) or not any(r['kind'] == 'unknown' for r in rows):
        raise ValueError('Calibration needs both groups')
    thresholds = sorted({r['margin'] for r in rows} | {0.0})
    best = None
    for threshold in thresholds:
        score = measure(rows, threshold)
        if score['unknown']['selected'] > max_unknown_selected:
            continue
        key = (score['known']['exact'], -score['known']['wrong'], -score['unknown']['selected'], threshold)
        if best is None or key > best[0]:
            best = (key, threshold)
    if best is None:
        raise ValueError('No feasible threshold')
    return best[1]


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    paths = {**source_paths(), 'data': DATA, 'cal_manifest': CAL_MANIFEST, 'dev_manifest': DEV_MANIFEST,
             'candidate': CANDIDATE / 'head.pt', 'candidate_report': CANDIDATE / 'result.json'}
    hashes = {k: runtime.sha(p) for k, p in paths.items()}
    source_rows = parquet.read_table(DATA).to_pylist()
    cal, dev = select_calibration(source_rows), select_dev(source_rows)[0]
    for rows, name in ((cal, 'cal_manifest'), (dev, 'dev_manifest')):
        manifest = json.loads(paths[name].read_text(encoding='utf8'))
        if manifest['source_sha256'] != SHA or [r['digest'] for r in rows] != [r['digest'] for r in manifest['rows']]:
            raise ValueError('Selection mismatch')
    saved = torch.load(paths['candidate'], map_location='cpu', weights_only=True)
    report = json.loads(paths['candidate_report'].read_text(encoding='utf8'))
    if (hashes['data'] != SHA or hashes['base'] != runtime.WEIGHT_SHA or hashes['candidate'] != report['head_sha256']
            or any(saved['source_sha256'][k] != hashes[k] for k in source_paths())):
        raise ValueError('Source mismatch')
    runtime.CHECKPOINT = SOURCE
    with tempfile.TemporaryDirectory(prefix='spora-margin-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'not-created.pt'
        chat = runtime.load_chat(natural_write_heads_path=HEADS)
        groups = {}
        for name, rows in (('calibration', cal), ('development', dev)):
            groups[name] = []
            for row in rows:
                groups[name].extend(extract(chat, [row]))
        if runtime.WORKING_CHECKPOINT.exists():
            raise ValueError('Unexpected individual file')
    head = ContextSpanSelector(groups['calibration'][0]['features'].shape[-1], rank=saved['rank']).eval()
    head.load_state_dict(saved['state'], strict=True)
    scored = {}
    for name, items in groups.items():
        scored[name] = []
        for item in items:
            x, mask, _, usable = pack([item])
            if item['row']['kind'] == 'known' and not bool(usable[0]):
                raise ValueError('Unrepresentable answer')
            with torch.inference_mode():
                logits = head(x, mask)[0]
                best = int(logits[1:].argmax()) + 1
                margin = float(logits[best] - logits[0])
            if not math.isfinite(margin):
                raise ValueError('Nonfinite margin')
            left, right = divmod(best - 1, x.shape[1])
            selected = torch.zeros(len(item['mask']), dtype=torch.bool)
            selected[left:right + 1] = True
            text = selected_text(item['row']['text'], selected, item['offsets'], item['start'])
            row = item['row']
            expected = row['text'][row['left']:row['right']] if row['kind'] == 'known' else None
            scored[name].append({'digest': row['digest'], 'kind': row['kind'], 'margin': margin,
                                 'best_exact': text == expected if expected is not None else None})
    threshold = choose_threshold(scored['calibration'], max_unknown_selected=2)
    summary = {name: {'default': measure(rows, 0.0), 'calibrated': measure(rows, threshold)}
               for name, rows in scored.items()}
    if any(runtime.sha(p) != hashes[k] for k, p in paths.items()):
        raise ValueError('Protected source changed')
    result = {'threshold': threshold, 'max_calibration_unknown_selected': 2, 'summary': summary,
              'rows': scored, 'source_sha256': hashes, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Calibration tolerance is descriptive, not statistical risk guarantee.',
                  'Calibration labels machine-translated and not independently verified.',
                  'Development evaluation already seen; same public validation distribution.',
                  'No deployment, memory, identity or natural dialogue evaluation.']}
    with OUT.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'threshold': threshold, 'summary': summary}), flush=True)


if __name__ == '__main__':
    main()
