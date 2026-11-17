"""Compare three isolated heads on translated answerable/unanswerable development items."""
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
from scripts.audit_chinese_squadv2_20261117 import SOURCE as DATA, SHA
from scripts.select_chinese_squadv2_who_20261117 import OUT as MANIFEST, select
from scripts.deepseek_cmrc_question_head_comparison_20261117 import batch
from scripts.deepseek_fact_head_natural_regression_20261117 import CANDIDATE as PRIOR
from scripts.deepseek_spora_general_subject_locator import extract
from scripts.deepseek_spora_interleaved_natural_transaction import HEADS, SOURCE, source_paths
from scripts.deepseek_spora_natural_write_span_diagnostic import selected_text
from spora.research_context_span import ContextSpanSelector
from spora.research_question_conditioned_span import QuestionConditionedSpan

OUT = ROOT / 'runs/deepseek-squadv2-who-probe-20261117.json'
NATURAL = ROOT / 'runs/deepseek-cmrc-reading-train-pilot-20261117'
COMPARISON = ROOT / 'runs/deepseek-cmrc-question-head-comparison-20261117'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    paths = {**source_paths(), 'data': DATA, 'manifest': MANIFEST,
             'prior': PRIOR / 'head.pt', 'prior_report': PRIOR / 'result.json',
             'natural': NATURAL / 'head.pt', 'natural_report': NATURAL / 'result.json',
             'question': COMPARISON / 'question.pt', 'question_report': COMPARISON / 'result.json'}
    hashes = {k: runtime.sha(p) for k, p in paths.items()}
    manifest = json.loads(MANIFEST.read_text(encoding='utf8'))
    rows, _ = select(parquet.read_table(DATA).to_pylist())
    if (hashes['data'] != SHA or hashes['base'] != runtime.WEIGHT_SHA or manifest['source_sha256'] != SHA
            or [r['digest'] for r in rows] != [r['digest'] for r in manifest['rows']]):
        raise ValueError('Source or selection mismatch')
    reports = {name: json.loads(paths[name + '_report'].read_text(encoding='utf8'))
               for name in ('prior', 'natural', 'question')}
    if (hashes['prior'] != reports['prior']['head_sha256'] or hashes['natural'] != reports['natural']['head_sha256']
            or hashes['question'] != reports['question']['candidate_sha256']['question']):
        raise ValueError('Candidate mismatch')
    states = {name: torch.load(paths[name], map_location='cpu', weights_only=True)
              for name in ('prior', 'natural', 'question')}
    if any(states[name]['source_sha256'][key] != hashes[key] for name in states for key in source_paths()):
        raise ValueError('Feature provenance mismatch')
    runtime.CHECKPOINT = SOURCE
    with tempfile.TemporaryDirectory(prefix='spora-squadv2-who-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'not-created.pt'
        chat = runtime.load_chat(natural_write_heads_path=HEADS)
        items = []
        for row in rows:
            items.extend(extract(chat, [row]))
        if runtime.WORKING_CHECKPOINT.exists():
            raise ValueError('Unexpected individual file')
    width = items[0]['features'].shape[-1]
    prior = ContextSpanSelector(width, rank=states['prior']['rank']).eval()
    natural = ContextSpanSelector(width, rank=states['natural']['rank']).eval()
    question = QuestionConditionedSpan(ContextSpanSelector(width, rank=states['prior']['rank']), width, rank=32).eval()
    for name, head in (('prior', prior), ('natural', natural), ('question', question)):
        head.load_state_dict(states[name]['state'], strict=True)
    heads = {'prior': prior, 'natural': natural, 'question': question}
    details = []
    for item in items:
        x, mask, q, _, usable = batch([item])
        row = item['row']
        expected = row['text'][row['left']:row['right']] if row['label'] is not None else None
        record = {'digest': row['digest'], 'group': row['kind'], 'representable': bool(usable[0]), 'heads': {}}
        for name, head in heads.items():
            with torch.inference_mode():
                logits = head(x, mask, q) if name == 'question' else head(x, mask)
                choice = int(logits.argmax())
            chosen = torch.zeros(len(item['mask']), dtype=torch.bool)
            if choice:
                left, right = divmod(choice - 1, x.shape[1])
                if right >= len(chosen):
                    raise ValueError('Padding selected')
                chosen[left:right + 1] = True
            text = selected_text(row['text'], chosen, item['offsets'], item['start']) if bool(chosen.any()) else None
            record['heads'][name] = {'selected': text is not None,
                                      'exact': text == expected if expected is not None else None}
        details.append(record)
    summary = {}
    for group in ('known', 'unknown'):
        subset = [r for r in details if r['group'] == group]
        summary[group] = {'total': len(subset), 'representable': sum(r['representable'] for r in subset),
                          **{name: {'selected': sum(r['heads'][name]['selected'] for r in subset),
                                    'exact': sum(r['heads'][name]['exact'] is True for r in subset) if group == 'known' else None}
                             for name in heads}}
    if any(runtime.sha(p) != hashes[k] for k, p in paths.items()):
        raise ValueError('Protected source changed')
    result = {'summary': summary, 'rows': details, 'source_sha256': hashes,
              'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Machine-translated and model-revised data; no independent label verification.',
                  'SQuAD impossibility is paragraph-relative and may be altered by translation.',
                  '32 development cases; no natural conversations, identity or memory evaluation.',
                  'Question ranges supplied by formatting; no autonomous task detection.']}
    with OUT.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
