"""Compare explicit two-head dispatch on reviewed text and a controlled query group."""
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
from scripts.deepseek_fact_competing_train_20261117 import build_groups
from scripts.deepseek_fact_head_natural_regression_20261117 import CANDIDATE
from scripts.deepseek_fact_paraphrase_train_20261117 import rewrite
from scripts.deepseek_fact_query_training_pilot_20261117 import evaluate
from scripts.deepseek_spora_consensus_reviewed_dialogue_audit import CONTEXT, chosen, rows_from_review
from scripts.deepseek_spora_duconv_dev_identity_audit import reviewed_sample, select_sample
from scripts.deepseek_spora_general_subject_locator import extract, pack
from scripts.deepseek_spora_interleaved_natural_transaction import HEADS, SOURCE, source_paths
from scripts.deepseek_spora_natural_write_span_diagnostic import selected_text
from spora.research_context_span import ContextSpanSelector
from spora.research_subject_task_heads import SubjectTask, SubjectTaskHeads

OUT = ROOT / 'runs/deepseek-task-heads-probe-20261117.json'


class BoundHead(torch.nn.Module):
    def __init__(self, wrapper, task):
        super().__init__()
        self.wrapper, self.task = wrapper, task

    def forward(self, features, mask):
        return self.wrapper(features, mask, task=self.task)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    paths = {**source_paths(), 'archive': ARCHIVE, 'observation': CONTEXT / 'head.pt',
             'observation_report': CONTEXT / 'result.json', 'query': CANDIDATE / 'head.pt',
             'query_report': CANDIDATE / 'result.json'}
    hashes = {k: runtime.sha(p) for k, p in paths.items()}
    observation = torch.load(paths['observation'], map_location='cpu', weights_only=True)
    query = torch.load(paths['query'], map_location='cpu', weights_only=True)
    report_o = json.loads(paths['observation_report'].read_text(encoding='utf8'))
    report_q = json.loads(paths['query_report'].read_text(encoding='utf8'))
    if (hashes['base'] != runtime.WEIGHT_SHA or hashes['archive'] != ARCHIVE_SHA
            or hashes['observation'] != report_o['head_sha256'] or hashes['query'] != report_q['head_sha256']
            or any(observation['source_sha256'][k] != hashes[k] for k in source_paths())
            or any(query['source_sha256'][k] != hashes[k] for k in source_paths())):
        raise ValueError('Source mismatch')
    natural = rows_from_review(reviewed_sample(select_sample()))
    query_rows = [rewrite(row, 'unseen') for row in build_groups()['same_new_values']]
    runtime.CHECKPOINT = SOURCE
    with tempfile.TemporaryDirectory(prefix='spora-two-task-heads-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'not-created.pt'
        chat = runtime.load_chat(natural_write_heads_path=HEADS)
        natural_items = extract(chat, natural)
        query_items = extract(chat, query_rows)
        if runtime.WORKING_CHECKPOINT.exists():
            raise ValueError('Unexpected individual file')
    width = natural_items[0]['features'].shape[-1]
    head_o = ContextSpanSelector(width, rank=observation['rank'])
    head_q = ContextSpanSelector(width, rank=query['rank'])
    head_o.load_state_dict(observation['state'], strict=True)
    head_q.load_state_dict(query['state'], strict=True)
    wrapper = SubjectTaskHeads(head_o, head_q).eval()
    bound = {task.value: BoundHead(wrapper, task).eval() for task in SubjectTask}
    rows = []
    for item in natural_items:
        x, _, _, usable = pack([item])
        row = item['row']
        if row['label'] is not None and not bool(usable[0]):
            raise ValueError('Unrepresentable gold span')
        expected = row['text'][row['left']:row['right']] if row['group'] == 'single' else None
        outputs = {}
        for name, head in bound.items():
            mask = chosen(head, item, x.shape[1])
            text = selected_text(row['text'], mask, item['offsets'], item['start']) if bool(mask.any()) else None
            outputs[name] = {'selected': text is not None,
                             'exact': text == expected if expected is not None else None}
        rows.append({'group': row['group'], 'heads': outputs})
    natural_summary = {}
    for group in ('single', 'none', 'multiple'):
        subset = [r for r in rows if r['group'] == group]
        natural_summary[group] = {'total': len(subset), **{name: {
            'selected': sum(r['heads'][name]['selected'] for r in subset),
            'exact': sum(r['heads'][name]['exact'] is True for r in subset) if group == 'single' else None}
            for name in bound}}
    query_summary = {name: evaluate(head, query_items) for name, head in bound.items()}
    if any(runtime.sha(p) != hashes[k] for k, p in paths.items()):
        raise ValueError('Protected source changed')
    result = {'natural': natural_summary, 'query': query_summary, 'source_sha256': hashes,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Caller manually supplies task; no automatic intent inference.',
                 'Both groups are already examined development data with different label semantics.',
                 'No actual memory reads, identity verification, writes or production integration.',
                 'No global accuracy or end-to-end chat claim.']}
    with OUT.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({'natural': natural_summary, 'query': query_summary}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
