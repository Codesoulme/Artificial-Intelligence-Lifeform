"""Small reviewed natural-text attribution probe, local source only."""
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import pyarrow.parquet as parquet
import torch
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.audit_duie_subject_relations import SOURCE as DATA, SOURCE_SHA
from scripts.duie_targeted_relation_candidates_20261117 import candidates
from scripts.duie_targeted_relation_review_20261117 import REVIEW, compare
from scripts.deepseek_fact_head_natural_regression_20261117 import CANDIDATE
from scripts.deepseek_spora_consensus_reviewed_dialogue_audit import CONTEXT, chosen
from scripts.deepseek_spora_general_subject_locator import extract, pack
from scripts.deepseek_spora_interleaved_natural_transaction import HEADS, SOURCE, source_paths
from scripts.deepseek_spora_natural_write_span_diagnostic import selected_text
from spora.research_context_span import ContextSpanSelector

OUT = ROOT / 'runs/deepseek-duie-relation-query-probe-20261117.json'


def make_rows(selected):
    compare(selected, REVIEW)
    result = []
    for row in selected:
        if REVIEW[row['digest']] != 'two_distinct_supported_pairs':
            continue
        facts = sorted({fact for pair in row['pairs'] for fact in pair})
        for subject, relation, obj in facts:
            if relation not in ('妻子', '丈夫'):
                raise ValueError('Unexpected relation')
            left = row['text'].index(subject)
            text = row['text'] + '。谁的' + relation + '是' + obj + '？'
            result.append({'text': text, 'label': 1, 'left': left, 'right': left + len(subject),
                           'source_digest': row['digest']})
    if len(result) != 8:
        raise ValueError('Unexpected reviewed fact count')
    return result


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    paths = {**source_paths(), 'data': DATA, 'observation': CONTEXT / 'head.pt',
             'observation_report': CONTEXT / 'result.json', 'query': CANDIDATE / 'head.pt',
             'query_report': CANDIDATE / 'result.json'}
    hashes = {k: runtime.sha(p) for k, p in paths.items()}
    observation = torch.load(paths['observation'], map_location='cpu', weights_only=True)
    query = torch.load(paths['query'], map_location='cpu', weights_only=True)
    report_o = json.loads(paths['observation_report'].read_text(encoding='utf8'))
    report_q = json.loads(paths['query_report'].read_text(encoding='utf8'))
    if (hashes['data'] != SOURCE_SHA or hashes['base'] != runtime.WEIGHT_SHA
            or hashes['observation'] != report_o['head_sha256'] or hashes['query'] != report_q['head_sha256']
            or any(observation['source_sha256'][k] != hashes[k] for k in source_paths())
            or any(query['source_sha256'][k] != hashes[k] for k in source_paths())):
        raise ValueError('Source mismatch')
    rows = make_rows(candidates(parquet.read_table(DATA).to_pylist()))
    runtime.CHECKPOINT = SOURCE
    with tempfile.TemporaryDirectory(prefix='spora-duie-query-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'not-created.pt'
        chat = runtime.load_chat(natural_write_heads_path=HEADS)
        items = extract(chat, rows)
        if runtime.WORKING_CHECKPOINT.exists():
            raise ValueError('Unexpected individual file')
    width = items[0]['features'].shape[-1]
    heads = {'observation': ContextSpanSelector(width, rank=observation['rank']),
             'fact_query': ContextSpanSelector(width, rank=query['rank'])}
    for name, state in (('observation', observation), ('fact_query', query)):
        heads[name].load_state_dict(state['state'], strict=True)
        heads[name].eval()
    details = []
    for item in items:
        x, _, _, usable = pack([item])
        row = item['row']
        expected = row['text'][row['left']:row['right']]
        detail = {'source_digest': row['source_digest'], 'representable': bool(usable[0]), 'heads': {}}
        for name, head in heads.items():
            mask = chosen(head, item, x.shape[1])
            text = selected_text(row['text'], mask, item['offsets'], item['start']) if bool(mask.any()) else None
            detail['heads'][name] = {'selected': text is not None, 'exact': text == expected}
        details.append(detail)
    summary = {'cases': len(details), 'documents': len({r['source_digest'] for r in details}),
               'representable': sum(r['representable'] for r in details), **{
                   name: {'selected': sum(r['heads'][name]['selected'] for r in details),
                          'exact': sum(r['heads'][name]['exact'] for r in details)} for name in heads}}
    if any(runtime.sha(p) != hashes[k] for k, p in paths.items()):
        raise ValueError('Protected source changed')
    result = {'summary': summary, 'rows': details, 'source_sha256': hashes,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Two development source texts, selected after manual review; not independent.',
                  'Questions authored from dataset relations; no absent fact or identity truth verification.',
                  'Source rights unresolved; no raw or derived text exported.',
                  'No memory read, write, persistence, or output generation.']}
    with OUT.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
