"""Audit fact-conditioned candidate against prior head on reviewed dialogue."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.audit_p236_duconv import ARCHIVE, ARCHIVE_SHA
from scripts.deepseek_spora_consensus_reviewed_dialogue_audit import CONTEXT, rows_from_review, chosen
from scripts.deepseek_spora_duconv_dev_identity_audit import reviewed_sample, select_sample
from scripts.deepseek_spora_general_subject_locator import extract, pack
from scripts.deepseek_spora_interleaved_natural_transaction import HEADS, SOURCE, source_paths
from scripts.deepseek_spora_natural_write_span_diagnostic import selected_text
from spora.research_context_span import ContextSpanSelector

OUT = ROOT / 'runs/deepseek-fact-head-natural-regression-20261117.json'
CANDIDATE = ROOT / 'runs/deepseek-fact-paraphrase-train-20261117'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    paths = {**source_paths(), 'archive': ARCHIVE, 'before': CONTEXT / 'head.pt',
             'before_report': CONTEXT / 'result.json', 'candidate': CANDIDATE / 'head.pt',
             'candidate_report': CANDIDATE / 'result.json'}
    hashes = {k: runtime.sha(p) for k, p in paths.items()}
    before = torch.load(paths['before'], map_location='cpu', weights_only=True)
    candidate = torch.load(paths['candidate'], map_location='cpu', weights_only=True)
    report_before = json.loads(paths['before_report'].read_text(encoding='utf8'))
    report_candidate = json.loads(paths['candidate_report'].read_text(encoding='utf8'))
    if (hashes['base'] != runtime.WEIGHT_SHA or hashes['archive'] != ARCHIVE_SHA
            or hashes['before'] != report_before['head_sha256']
            or hashes['candidate'] != report_candidate['head_sha256']
            or any(before['source_sha256'][k] != hashes[k] for k in source_paths())
            or any(candidate['source_sha256'][k] != hashes[k] for k in source_paths())):
        raise ValueError('Source mismatch')
    rows = rows_from_review(reviewed_sample(select_sample()))
    runtime.CHECKPOINT = SOURCE
    with tempfile.TemporaryDirectory(prefix='spora-fact-natural-regression-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'not-created.pt'
        chat = runtime.load_chat(natural_write_heads_path=HEADS)
        items = extract(chat, rows)
        if runtime.WORKING_CHECKPOINT.exists():
            raise ValueError('Unexpected individual file')
    width = items[0]['features'].shape[-1]
    heads = {'before': ContextSpanSelector(width, rank=before['rank']),
             'candidate': ContextSpanSelector(width, rank=candidate['rank'])}
    for name, payload in (('before', before), ('candidate', candidate)):
        heads[name].load_state_dict(payload['state'], strict=True)
        heads[name].eval()
    details = []
    for item in items:
        x, _, _, usable = pack([item])
        row = item['row']
        if row['label'] is not None and not bool(usable[0]):
            raise ValueError('Unrepresentable gold')
        expected = row['text'][row['left']:row['right']] if row['group'] == 'single' else None
        detail = {'digest': row['digest'], 'group': row['group'], 'heads': {}}
        for name, head in heads.items():
            mask = chosen(head, item, x.shape[1])
            text = selected_text(row['text'], mask, item['offsets'], item['start']) if bool(mask.any()) else None
            detail['heads'][name] = {'selected': text is not None,
                                     'exact': text == expected if expected is not None else None}
        details.append(detail)
    summary = {}
    for group in ('single', 'none', 'multiple'):
        subset = [row for row in details if row['group'] == group]
        summary[group] = {'total': len(subset), **{name: {
            'selected': sum(r['heads'][name]['selected'] for r in subset),
            'exact': sum(r['heads'][name]['exact'] is True for r in subset) if group == 'single' else None}
            for name in heads}}
    if any(runtime.sha(p) != hashes[k] for k, p in paths.items()):
        raise ValueError('Protected source changed')
    result = {'summary': summary, 'rows': details, 'source_sha256': hashes,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Previously reviewed single-turn development data, one annotator.',
                  'Natural sentences lack an explicit fact query, unlike candidate training.',
                  'Multiple subjects lack unique target labels; count selection only.',
                  'No identity, read, write, answer or natural-language memory evaluation.']}
    with OUT.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
