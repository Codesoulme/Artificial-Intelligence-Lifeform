"""Compare prior full-context predictions with oracle answer-sentence predictions."""
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
from scripts.deepseek_cmrc_who_probe_20261117 import OUT as PRIOR
from scripts.deepseek_fact_head_natural_regression_20261117 import CANDIDATE
from scripts.deepseek_spora_consensus_reviewed_dialogue_audit import CONTEXT, chosen
from scripts.deepseek_spora_general_subject_locator import extract, pack
from scripts.deepseek_spora_interleaved_natural_transaction import HEADS, SOURCE, source_paths
from scripts.deepseek_spora_natural_write_span_diagnostic import selected_text
from spora.research_context_span import ContextSpanSelector

OUT = ROOT / 'runs/deepseek-cmrc-oracle-sentence-probe-20261117.json'
DELIMITERS = '。！？；\n'


def oracle_sentence(row):
    context, sep, question = row['text'].partition('\n问题：')
    left, right = row['left'], row['right']
    if not sep or not 0 <= left < right <= len(context):
        raise ValueError('Invalid source offsets')
    start = max((context.rfind(ch, 0, left) for ch in DELIMITERS), default=-1) + 1
    next_ends = [i for ch in DELIMITERS if (i := context.find(ch, right)) >= 0]
    end = min(next_ends) + 1 if next_ends else len(context)
    if any(ch in context[left:right] for ch in DELIMITERS):
        raise ValueError('Answer crosses sentence boundary')
    clipped = context[start:end] + '\n问题：' + question
    if clipped[left - start:right - start] != context[left:right]:
        raise ValueError('Answer offset changed')
    return {**row, 'text': clipped, 'left': left - start, 'right': right - start}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    paths = {**source_paths(), 'data': DATA, 'manifest': MANIFEST, 'prior': PRIOR,
             'observation': CONTEXT / 'head.pt', 'observation_report': CONTEXT / 'result.json',
             'query': CANDIDATE / 'head.pt', 'query_report': CANDIDATE / 'result.json'}
    hashes = {k: runtime.sha(p) for k, p in paths.items()}
    raw = DATA.read_bytes()
    if blob_sha1(raw) != BLOB_SHA1 or hashes['base'] != runtime.WEIGHT_SHA:
        raise ValueError('Source mismatch')
    rows, _ = select(json.loads(raw.decode('utf8')))
    manifest = json.loads(MANIFEST.read_text(encoding='utf8'))
    prior = json.loads(PRIOR.read_text(encoding='utf8'))
    if ([r['digest'] for r in rows] != [r['digest'] for r in manifest['rows']]
            or [r['digest'] for r in rows] != [r['digest'] for r in prior['rows']]
            or any(prior['source_sha256'][key] != hashes[key] for key in ('base', 'data', 'manifest', 'query', 'observation'))):
        raise ValueError('Prior sample or source mismatch')
    clipped = [oracle_sentence(row) for row in rows]
    states = {name: torch.load(paths[name], map_location='cpu', weights_only=True)
              for name in ('observation', 'query')}
    for name in states:
        report = json.loads(paths[name + '_report'].read_text(encoding='utf8'))
        if hashes[name] != report['head_sha256'] or any(
                states[name]['source_sha256'][k] != hashes[k] for k in source_paths()):
            raise ValueError('Head mismatch')
    runtime.CHECKPOINT = SOURCE
    with tempfile.TemporaryDirectory(prefix='spora-cmrc-oracle-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'not-created.pt'
        chat = runtime.load_chat(natural_write_heads_path=HEADS)
        items = []
        for row in clipped:
            items.extend(extract(chat, [row]))
        if runtime.WORKING_CHECKPOINT.exists():
            raise ValueError('Unexpected individual file')
    width = items[0]['features'].shape[-1]
    heads = {}
    for name, state in states.items():
        heads[name] = ContextSpanSelector(width, rank=state['rank']).eval()
        heads[name].load_state_dict(state['state'], strict=True)
    records = []
    for item, previous in zip(items, prior['rows']):
        x, _, _, usable = pack([item])
        row = item['row']
        expected = row['text'][row['left']:row['right']]
        record = {'digest': row['digest'], 'representable': bool(usable[0]), 'heads': {}}
        for name, head in heads.items():
            mask = chosen(head, item, x.shape[1])
            text = selected_text(row['text'], mask, item['offsets'], item['start']) if bool(mask.any()) else None
            record['heads'][name] = {'selected': text is not None, 'exact': text == expected,
                                     'full_exact': previous['heads'][name]['exact']}
        records.append(record)
    summary = {'cases': len(records), 'representable': sum(r['representable'] for r in records),
               **{name: {'oracle_selected': sum(r['heads'][name]['selected'] for r in records),
                          'oracle_exact': sum(r['heads'][name]['exact'] for r in records),
                          'full_exact': sum(r['heads'][name]['full_exact'] for r in records)} for name in heads}}
    if any(runtime.sha(p) != hashes[k] for k, p in paths.items()):
        raise ValueError('Source changed')
    result = {'summary': summary, 'rows': records, 'source_sha256': hashes,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Oracle uses the gold answer location, not autonomous retrieval.',
                  'Sentence may omit required context; factual answerability is not re-annotated.',
                  'Previously inspected development data; no negative or independent evaluation.',
                  'No actual memory state, writes, or generated answers.']}
    with OUT.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
