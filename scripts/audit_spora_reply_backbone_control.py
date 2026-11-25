"""Compare frozen instruction weights and their SPORA port on the same reply tasks."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

from scripts.audit_spora_chat_backbones import reply
from scripts.audit_spora_dialogue_backbone_pivot import compare_qwen
from scripts.audit_spora_reply_preservation import (
    BUDGET, PUBLIC_BUDGET, REVISIONS, assess, history_for, public_checks, public_rows,
)
from scripts.deepseek_local_baseline import sha
from scripts.train_spora_recurrent_counterfactual import store_json


def held_cases(data, reference):
    cases = [
        {'split': 'public', 'group': group, 'level': level, 'question': question}
        for group, level, question in public_rows(data)
    ]
    cases.extend(
        {'split': 'revision', 'group': name, 'previous': previous, 'question': question}
        for name, previous, question, _, _ in REVISIONS
    )
    for case, row in zip(cases, reference['rows'], strict=True):
        if any(row.get(key) != value for key, value in case.items()):
            raise ValueError('Stored DeepSeek output does not match the shared evaluation cases')
    return cases


def evaluate(model, tokenizer, eos, cases):
    revision_patterns = {name: (required, forbidden) for name, _, _, required, forbidden in REVISIONS}
    outputs = []
    for case in cases:
        split, group = case['split'], case['group']
        history = history_for(case['previous']) if split == 'revision' else ()
        budget = BUDGET if split == 'revision' else PUBLIC_BUDGET
        output = reply(model, tokenizer, history, case['question'], device='cuda', deepseek=False, limit=budget, eos=eos)
        if split == 'public':
            checks = public_checks(group, case['level'], output['answer'])
            passed = output['eos'] and all(checks.values())
        else:
            required, forbidden = revision_patterns[group]
            checks = assess(output['answer'], output['eos'], required, (*forbidden, r'\n|以下'), 65)
            passed = checks['pass']
        outputs.append({**output, 'checks': checks, 'passed': passed})
        print(type(model).__name__, split, group, passed, output['eos'], flush=True)
    return outputs


def summarize(rows, model):
    summary = {}
    for split in ('public', 'revision'):
        group = [row['outputs'][model] for row in rows if row['split'] == split]
        counts = Counter(key for output in group for key, value in output['checks'].items() if value)
        summary[split] = {
            'cases': len(group), 'passed': sum(output['passed'] for output in group),
            'eos': sum(output['eos'] for output in group), 'checks_passed': dict(counts),
            'median_generated_tokens': median(output.get('generated_tokens', len(output.get('token_ids', ())))
                                              for output in group),
        }
    return summary


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = {path: sha(path) for path in (args.data, args.reference, Path(__file__))}
    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    cases = held_cases(args.data, reference)
    comparison = compare_qwen(args.checkpoint, args.donor, args.calibration,
                              lambda model, tokenizer, eos: evaluate(model, tokenizer, eos, cases))
    rows = [
        {**case, 'outputs': {'deepseek': prior['outputs']['base'], 'qwen': native, 'qwen_spora': integrated}}
        for case, prior, native, integrated in zip(cases, reference['rows'], comparison.pop('native'),
                                                  comparison.pop('integrated'), strict=True)
    ]
    if any(sha(path) != digest for path, digest in sources.items()):
        raise ValueError('An evaluation input changed during inference')
    summary = {model: summarize(rows, model) for model in rows[0]['outputs']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    store_json(args.output, {**comparison, 'evaluation_sha256': {str(path): digest for path, digest in sources.items()},
                            'public_budget': PUBLIC_BUDGET, 'revision_budget': BUDGET,
                            'rows': rows, 'summary': summary})
    print(summary, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'reference', 'checkpoint', 'donor', 'calibration', 'output'):
        cli.add_argument(name, type=Path)
    main(cli.parse_args())
