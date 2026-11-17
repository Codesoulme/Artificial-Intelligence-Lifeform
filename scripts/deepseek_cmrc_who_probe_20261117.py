"""Read-only CMRC development check; QA answers are not identity truth."""
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
from scripts.audit_cmrc2018_dev_20261117 import SOURCE as DATA, BLOB_SHA1, blob_sha1
from scripts.cmrc_who_development_selection_20261117 import OUT as MANIFEST, select
from scripts.deepseek_fact_head_natural_regression_20261117 import CANDIDATE
from scripts.deepseek_spora_consensus_reviewed_dialogue_audit import CONTEXT, chosen
from scripts.deepseek_spora_general_subject_locator import extract, pack
from scripts.deepseek_spora_interleaved_natural_transaction import HEADS, SOURCE, source_paths
from scripts.deepseek_spora_natural_write_span_diagnostic import selected_text
from spora.research_context_span import ContextSpanSelector

OUT = ROOT / 'runs/deepseek-cmrc-who-probe-20261117.json'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    paths = {**source_paths(), 'data': DATA, 'manifest': MANIFEST,
             'observation': CONTEXT / 'head.pt', 'observation_report': CONTEXT / 'result.json',
             'query': CANDIDATE / 'head.pt', 'query_report': CANDIDATE / 'result.json'}
    hashes = {k: runtime.sha(p) for k, p in paths.items()}
    raw = DATA.read_bytes()
    if blob_sha1(raw) != BLOB_SHA1:
        raise ValueError('Source mismatch')
    rows, eligible = select(json.loads(raw.decode('utf8')))
    manifest = json.loads(MANIFEST.read_text(encoding='utf8'))
    if (hashes['base'] != runtime.WEIGHT_SHA or manifest['source_sha256'] != hashes['data']
            or [r['digest'] for r in rows] != [r['digest'] for r in manifest['rows']] or len(rows) != 32):
        raise ValueError('Source or selection mismatch')
    states = {name: torch.load(paths[name], map_location='cpu', weights_only=True)
              for name in ('observation', 'query')}
    for name in states:
        report = json.loads(paths[name + '_report'].read_text(encoding='utf8'))
        if (hashes[name] != report['head_sha256']
                or any(states[name]['source_sha256'][k] != hashes[k] for k in source_paths())):
            raise ValueError('Head mismatch')
    runtime.CHECKPOINT = SOURCE
    with tempfile.TemporaryDirectory(prefix='spora-cmrc-who-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'not-created.pt'
        chat = runtime.load_chat(natural_write_heads_path=HEADS)
        items = []
        for row in rows:
            items.extend(extract(chat, [row]))
        if runtime.WORKING_CHECKPOINT.exists():
            raise ValueError('Unexpected individual file')
    width = items[0]['features'].shape[-1]
    heads = {}
    for name, state in states.items():
        heads[name] = ContextSpanSelector(width, rank=state['rank']).eval()
        heads[name].load_state_dict(state['state'], strict=True)
    details = []
    for item in items:
        x, _, _, usable = pack([item])
        row = item['row']
        expected = row['text'][row['left']:row['right']]
        record = {'digest': row['digest'], 'representable': bool(usable[0]), 'heads': {}}
        for name, head in heads.items():
            mask = chosen(head, item, x.shape[1])
            text = selected_text(row['text'], mask, item['offsets'], item['start']) if bool(mask.any()) else None
            record['heads'][name] = {'selected': text is not None, 'exact': text == expected}
        details.append(record)
    summary = {'cases': len(details), 'eligible': eligible, 'representable': sum(r['representable'] for r in details),
               **{name: {'selected': sum(r['heads'][name]['selected'] for r in details),
                          'exact': sum(r['heads'][name]['exact'] for r in details)} for name in heads}}
    if any(runtime.sha(p) != hashes[k] for k, p in paths.items()):
        raise ValueError('Source changed')
    result = {'summary': summary, 'rows': details, 'source_sha256': hashes,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['CMRC development questions, not a sealed independent evaluation.',
                  'Some who-answer labels may refer to non-persons, organizations or collective groups.',
                  'Labels are dataset answer spans, not confirmed identity or evidence sufficiency.',
                  'Full context is supplied, no memory reading, writing, answer generation or unknown cases.',
                  'Only one gold text and location retained after restrictive deterministic filtering.']}
    with OUT.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
