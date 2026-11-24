"""Compare native language backbones with the same early and late dialogue evidence."""

import argparse
from collections import Counter
from pathlib import Path
import re

from scripts.audit_spora_chat_backbones import FILES, reply
from scripts.audit_spora_recurrent_field_composition import SCENARIOS, read
from scripts.benchmark_spora_recurrent_dialogue import (
    MODEL, SOURCE, AutoTokenizer, grade, load_full_streamed, recall_prompts, sha, torch,
)
from scripts.deepseek_spora_full_stream_loader import load_qwen_streamed
from scripts.train_spora_recurrent_counterfactual import store_json


QWEN_SHA = 'dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee'


def histories(row):
    evidence = row['history'][:4]
    filler = row['history'][4:]
    return {'compact': evidence, 'early': evidence + filler, 'late': filler + evidence}


def main(args):
    model_dir = args.qwen.resolve() if args.qwen else MODEL
    protected = {path: sha(path) for path in (SOURCE, SCENARIOS, *(model_dir / name for name in FILES))}
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    source = {**read(SOURCE), 'recall': read(SCENARIOS)['recall']}
    reference = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False) if args.qwen else tokenizer
    prompts = [row for row in recall_prompts(reference, source, extended=True) if row['length'] == 'extended']
    model = load_qwen_streamed(model_dir, QWEN_SHA) if args.qwen else load_full_streamed()
    eos = model.generation_config.eos_token_id if args.qwen else tokenizer.eos_token_id
    settings = {'device': str(model.device), 'deepseek': args.qwen is None, 'limit': 48,
                'eos': eos if isinstance(eos, list) else [eos]}
    summaries, rows = {}, {}
    for mode in ('compact', 'early', 'late'):
        counts = Counter()
        answers = []
        for row in prompts:
            values = row['values']
            history = histories(row)[mode]
            result = reply(model, tokenizer, history, row['question'], **settings)
            if mode == 'early' and args.qwen is None and result['prompt_tokens'] != row['input_tokens']:
                raise ValueError('The native DeepSeek prompt differs from the original experiment')
            if mode == 'late' and result['prompt_tokens'] != rows['early'][len(answers)]['prompt_tokens']:
                raise ValueError('Moving the fact turns changed prompt length')
            passed = grade(result, [re.escape(values['new']), re.escape(values['keep'])], [re.escape(values['old'])])
            counts.update({'passed': passed, 'new': values['new'] in result['answer'],
                           'keep': values['keep'] in result['answer'], 'old': values['old'] in result['answer'],
                           'eos': result['eos']})
            answers.append({'case': row['case'], 'variant': row['variant'], 'values': values, **result,
                            'passed': passed})
        rows[mode] = answers
        summaries[mode] = dict(counts)
        print(mode, summaries[mode], flush=True)
    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError('A benchmark source changed during the run')
    store_json(args.output, {'sources': {str(path): digest for path, digest in protected.items()},
                             'settings': settings, 'summaries': summaries, 'rows': rows})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--qwen', type=Path, help='Local Qwen2.5-1.5B-Instruct snapshot')
    options = parser.parse_args()
    if options.output.exists() or not options.output.parent.is_dir():
        parser.error('Choose a new result path in an existing directory')
    main(options)
