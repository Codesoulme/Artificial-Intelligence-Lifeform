"""Factor distance and requested field count in revision recall."""

import argparse
import json
from pathlib import Path
import re
from time import perf_counter

from scripts.audit_spora_chat_backbones import reply
from scripts.benchmark_spora_recurrent_dialogue import (
    MODEL, SOURCE, AutoTokenizer, encode, extend_positions, frozen_digest, grade, load_full_streamed,
    load_integrated_runtime, recall_prompts, sha, torch, using,
)
from scripts.train_spora_dialogue_composition import settings_for
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.research_recurrent_decode import load_recurrent_replacements


TRANSFER = SOURCE.with_name('recurrent-layout-recall-transfer-20261126.json')
SINGLE_QUESTIONS = {
    'market-sign': ('导览牌最后摆在哪儿？', '最后谁来收导览牌？'),
    'radio-rehearsal': ('广播最新的录音时间是什么？', '剪片头的是哪位？'),
    'library-storage': ('值班纸上写用什么做标签？', '未登记的书去哪儿找？'),
    'printing-handoff': ('最后要交几份样张？', '样张最后交给谁？'),
    'seed-workshop': ('明天布置时新牌子上写什么？', '明天布置时记录工具拿什么？'),
}


def paired_prompts(tokenizer, source):
    rows = []
    for row in recall_prompts(tokenizer, source, extended=True):
        if row['length'] != 'extended':
            continue
        facts, filler = row['history'][:4], row['history'][4:]
        questions = (row['question'], *SINGLE_QUESTIONS[row['case']])
        for field, question in zip(('both', 'new', 'keep'), questions, strict=True):
            lengths = []
            for distance, history in (('far', facts + filler), ('near', filler + facts)):
                length = len(encode(tokenizer, [*history, {'role': 'user', 'content': question}]))
                lengths.append(length)
                rows.append(dict(case=row['case'], variant=row['variant'], field=field, distance=distance,
                                 history=history, question=question, values=row['values'], input_tokens=length))
            if len(set(lengths)) != 1:
                raise ValueError(f'Moving the fact turns changed the input length for {row["case"]}: {lengths}')
    if len(rows) != 6 * sum(len(case['values']) for case in source['recall']):
        raise ValueError('Incomplete factorial prompts')
    return rows


def measure(decoder, tokenizer, prompts):
    settings = settings_for(decoder, tokenizer, 48)
    rows = []
    for prompt in prompts:
        if prompt['input_tokens'] + settings['limit'] > decoder.position_table_max_position:
            raise ValueError('Prompt plus response exceeds the position table')
        answer = reply(decoder, tokenizer, prompt['history'], prompt['question'], **settings)
        if answer['prompt_tokens'] != prompt['input_tokens']:
            raise ValueError('Prompt used at generation differs from the measurement')
        required = ('new', 'keep') if prompt['field'] == 'both' else (prompt['field'],)
        values = prompt['values']
        passed = grade(answer, [re.escape(values[field]) for field in required], [re.escape(values['old'])])
        rows.append({name: value for name, value in prompt.items() if name != 'history'} | answer | {
            'passed': passed, 'mentions': {field: value in answer['answer'] for field, value in values.items()}})
    metrics = {}
    for distance in ('far', 'near'):
        metrics[distance] = {}
        for field in ('both', 'new', 'keep'):
            group = [row for row in rows if (row['distance'], row['field']) == (distance, field)]
            metrics[distance][field] = dict(count=len(group), passed=sum(row['passed'] for row in group),
                eos=sum(row['eos'] for row in group),
                mentions={key: sum(row['mentions'][key] for row in group) for key in ('new', 'keep', 'old')})
    return dict(rows=rows, metrics=metrics)


def main(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {path: sha(path) for path in (args.checkpoint, SOURCE, TRANSFER,
        *(directory / name for directory in args.layers for name in ('attention.pt', 'comparison.json')))}
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    source['recall'] = json.loads(TRANSFER.read_text(encoding='utf-8'))
    prompts = paired_prompts(tokenizer, source)
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    base.lm_head.to(base.device)
    identity = frozen_digest(runtime.organism, runtime.state)
    replacements = load_recurrent_replacements(decoder, args.layers, sources[args.checkpoint])
    if set(replacements) != {19, 22}:
        raise ValueError('Expected the two-layer migration baseline')
    results = {}
    for label, mode in (('parent', 'parent'), ('19-22', 'recurrent')):
        with using(replacements, mode):
            arm = measure(decoder, tokenizer, prompts)
        results[label] = arm
        store_json(args.output / f'{label}.json', arm)
        print(label, arm['metrics'], flush=True)
    if identity != frozen_digest(runtime.organism, runtime.state) or any(
            sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('A source or the individual changed during the comparison')
    store_json(args.output / 'result.json', dict(results=results, individual_sha256=identity,
               sources={str(path): digest for path, digest in sources.items()}, seconds=perf_counter() - started))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('layers', type=Path, nargs=2)
    main(cli.parse_args())
