"""Measure C-Eval and short-dialogue retention after checkpoint changes."""

import argparse
import json
from pathlib import Path

from scripts.audit_local_named_chat import FRESH_CASES, run
from scripts.audit_spora_dialogue_migration import CEVAL_GROUPS, ceval_questions, score_ceval
from scripts.deepseek_local_baseline import MODEL, sha, torch
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_qwen_counterfactual_reply_20261122 import HELD, evaluate, records
from spora.integrated_checkpoint import load_integrated
from spora.language_training import frozen_digest
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('report', type=Path)
    parser.add_argument('--ceval', type=Path, default=Path('runs'))
    parser.add_argument('--short', action='store_true')
    parser.add_argument('--memory', action='store_true')
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    digest = sha(args.checkpoint)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model, organism, state = load_integrated(args.checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    individual = frozen_digest(organism, state)
    questions = [row for folder, subjects in CEVAL_GROUPS.items()
                 for row in ceval_questions(args.ceval / folder, tokenizer, subjects)]
    rows = score_ceval(model, questions, model.position_table_max_position or model.base.config.max_position_embeddings)
    report = {'checkpoint_sha256': digest, 'ceval': rows}
    if args.short:
        report['original'] = run(model, tokenizer)
        report['fresh'] = run(model, tokenizer, cases=FRESH_CASES)
        print('Short dialogue', {name: (report[name]['cases_passed'], report[name]['turns_passed'])
                                 for name in ('original', 'fresh')}, flush=True)
    if args.memory:
        report['memory'] = evaluate(model, tokenizer, organism, records(organism, HELD))
        print('Memory', report['memory']['summary'], flush=True)
    if sha(args.checkpoint) != digest or frozen_digest(organism, state) != individual:
        raise ValueError('Checkpoint or individual changed during evaluation')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('C-Eval', sum(row['choice'] == row['expected'] for row in rows), '/', len(rows), flush=True)


if __name__ == '__main__':
    main()
