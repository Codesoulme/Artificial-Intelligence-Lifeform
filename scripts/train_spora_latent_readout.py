"""Test separate hidden reads and one natural joint answer on the same decoder."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import random
import re
from time import perf_counter

from torch.nn import functional as F

from scripts.audit_spora_recurrent_field_composition import QUESTIONS, SCENARIOS, read
from scripts.benchmark_spora_recurrent_dialogue import grade, reply, settings_for
from scripts.train_spora_recurrent_counterfactual import (
    DATA, CORPUS, HELD, MANIFEST, MODEL, SOURCE, AutoTokenizer, alternate, dialogue_data, encode,
    evaluate, extend_positions, frozen_digest, install, load_full_streamed, load_integrated_runtime,
    load_recurrent_replacements, material, recall_prompts, records, sha, store_json, torch,
)
from scripts.train_spora_recurrent_long_alignment import training_history
from spora.language_training import atomic_save
from spora.research_latent_readout import LatentReadout


SEED = 25124
FIELD_NAMES = {'when': '日期和时间', 'place': '地点', 'gear': '物品', 'person': '收件人',
               'quantity': '数量', 'direction': '方向'}


def prompt_ids(tokenizer, history, question):
    return encode(tokenizer, [*history, {'role': 'user', 'content': question}])


def training_examples(tokenizer):
    payload = read(DATA)
    fillers = material(tokenizer, CORPUS, 40)
    randomizer = random.Random(SEED)
    examples = []
    for index, first in enumerate(dialogue_data(payload, held=False, count=8)):
        for case in (first, alternate(first, payload, randomizer)):
            values = dict(case['current'])
            changes = set(case['changes'])
            if 'day' in values:
                values['when'] = values.pop('day') + values.pop('time')
                if {'day', 'time'} & changes:
                    changes = (changes - {'day', 'time'}) | {'when'}
            updated, = changes
            unchanged = sorted(values.keys() - changes)
            fields = (updated, unchanged[index % len(unchanged)])
            names = tuple(FIELD_NAMES[field] for field in fields)
            a, b = (values[field] for field in fields)
            question = (f'现在的{names[0]}和{names[1]}各是什么？', f'把最新的{names[0]}和{names[1]}一起告诉我。',
                        f'我需要这次的{names[0]}及{names[1]}，请发完整。')[index % 3]
            target_text = (f'{a}，{b}。', f'{names[0]}为{a}，{names[1]}为{b}。')[index % 2]
            history = training_history(case, fillers, index)
            queries = [prompt_ids(tokenizer, history, f'请只说现在的{name}。') for name in names]
            labels = tokenizer.encode(target_text, add_special_tokens=False) + [tokenizer.eos_token_id]
            ids = prompt_ids(tokenizer, history, question) + labels[:-1]
            examples.append((queries, ids, labels))
    return examples


class _Captured(Exception):
    pass


@torch.no_grad()
def capture(decoder, layer, requests):
    results = []
    width = 1

    def save_hidden(_module, _args, output):
        results.append(output[0, -width:].detach().cpu())
        raise _Captured

    with layer.register_forward_hook(save_hidden):
        for ids, width in requests:
            if len(ids) > decoder.position_table_max_position:
                raise ValueError('The capture exceeds the decoder position table')
            try:
                decoder(input_ids=torch.tensor([ids], device=decoder.base.device), use_cache=True, logits_to_keep=1)
            except _Captured:
                continue
            raise RuntimeError('The final layer did not emit features')
    return results


def fit(base, bridge, samples, steps):
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=2e-4, weight_decay=.001)
    randomizer = random.Random(SEED)
    order = list(range(len(samples)))
    trace, window = [], []
    for step in range(steps):
        if step % len(samples) == 0:
            randomizer.shuffle(order)
        slots, hidden, labels = samples[order[step % len(samples)]]
        transformed = bridge(hidden[None].to(base.device), slots[None].to(base.device))
        logits = base.lm_head(base.model.norm(transformed))[0].float()
        loss = F.cross_entropy(logits, torch.tensor(labels, device=base.device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        window.append(float(loss.detach()))
        if (step + 1) % 32 == 0 or step + 1 == steps:
            trace.append({'step': step + 1, 'loss': sum(window) / len(window)})
            print(trace[-1], flush=True)
            window.clear()
    return trace


def score_answer(decoder, tokenizer, layer, bridge, row, slots, new):
    values = row['values']

    def inject(_module, _args, output):
        return bridge(output, slots)

    with layer.register_forward_hook(inject):
        answer = reply(decoder, tokenizer, row['history'], row['question'], **settings_for(decoder, tokenizer, 48))
    answer.pop('seconds')
    if answer['prompt_tokens'] != row['input_tokens']:
        raise ValueError('The joint answer prompt changed')
    forbidden = [values['old']]
    if new != values['new']:
        forbidden.append(values['new'])
    return {**answer, 'passed': grade(answer, [re.escape(new), re.escape(values['keep'])],
                                     [re.escape(value) for value in forbidden])}


def ablate(decoder, tokenizer, layer, bridge, held, questions):
    rows = []
    for pair in zip(held[::2], held[1::2], strict=True):
        for row, other in (pair, pair[::-1]):
            values, other_value = row['values'], other['values']['new']
            field = questions[row['case']]
            changed = [turn.copy() for turn in row['history']]
            update = changed[2]['content']
            if update.count(values['new']) != 1:
                raise ValueError('The state swap needs one occurrence of the corrected value')
            changed[2]['content'] = update.replace(values['new'], other_value)
            queries = [(prompt_ids(tokenizer, row['history'], field[name]), 1) for name in ('new', 'keep')]
            queries.append((prompt_ids(tokenizer, changed, field['new']), 1))
            new, keep, swap = (hidden[0] for hidden in capture(decoder, layer, queries))
            normal = torch.stack((new, keep))[None].to(decoder.base.device)
            swapped = torch.stack((swap, keep))[None].to(decoder.base.device)
            rows.append({'case': row['case'], 'variant': row['variant'], 'values': values, 'swap_value': other_value,
                         'matched': score_answer(decoder, tokenizer, layer, bridge, row, normal, values['new']),
                         'swap_new': score_answer(decoder, tokenizer, layer, bridge, row, swapped, other_value)})
    return rows


def main(args):
    started = perf_counter()
    model_inputs = (args.checkpoint, SOURCE, SCENARIOS,
                    *(folder / name for folder in args.candidates for name in ('attention.pt', 'comparison.json')))
    inputs = {path: sha(path) for path in (*model_inputs, QUESTIONS, DATA, CORPUS, MANIFEST, args.baseline)}
    if read(MANIFEST)['files'][CORPUS.name] != inputs[CORPUS]:
        raise ValueError('The filler corpus changed')
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    scenarios = {**read(SOURCE), 'recall': read(SCENARIOS)['recall']}
    held = [row for row in recall_prompts(tokenizer, scenarios, extended=True) if row['length'] == 'extended']
    archive = read(args.baseline)
    previous = [row for row in archive['conditions']['recurrent']['recall'] if row['length'] == 'extended']
    sources = {Path(name).resolve(): digest for name, digest in archive['sources'].items()}
    if (sources != {path.resolve(): inputs[path] for path in model_inputs}
            or any(any(saved[key] != value for key, value in row.items())
                   for row, saved in zip(held, previous, strict=True))):
        raise ValueError('The baseline uses different model inputs or dialogues')
    plans = training_examples(tokenizer)
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    torch.cuda.empty_cache()
    base.lm_head.to(base.device)
    individual = frozen_digest(runtime.organism, runtime.state)
    if archive['individual_sha256'] != individual:
        raise ValueError('The baseline uses another individual')
    bridge = LatentReadout(base.config.hidden_size).to(base.device)
    layer = base.model.layers[-1]
    candidates = load_recurrent_replacements(decoder, args.candidates, inputs[args.checkpoint])
    args.output.mkdir(parents=True, exist_ok=False)
    bound = {str(path): digest for path, digest in inputs.items()}
    with ExitStack() as state:
        install(state, candidates)
        print(f'Capturing two reads and joint answer features for {len(plans)} training records', flush=True)
        requests = [request for queries, ids, labels in plans for request in
                    [(queries[0], 1), (queries[1], 1), (ids, len(labels))]]
        captured = capture(decoder, layer, requests)
        samples = [(torch.cat((first, second)), hidden, plan[2]) for first, second, hidden, plan in
                   zip(captured[::3], captured[1::3], captured[2::3], plans, strict=True)]
        torch.cuda.empty_cache()
        trace = fit(base, bridge, samples, args.steps)
        bridge.eval().requires_grad_(False)
        weights = args.output / 'bridge.pt'
        atomic_save({'weights': {key: value.cpu() for key, value in bridge.state_dict().items()},
                     'sources': bound, 'individual_sha256': individual, 'steps': args.steps}, weights)
        print('Saved latent readout; evaluating original and counterfactual reads', flush=True)
        with torch.no_grad():
            after = ablate(decoder, tokenizer, layer, bridge, held, read(QUESTIONS))
        memory = evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))
    if (frozen_digest(runtime.organism, runtime.state) != individual
            or any(sha(path) != digest for path, digest in inputs.items())):
        raise ValueError('The individual or an input changed')
    summary = {'base': sum(row['passed'] for row in previous),
               'matched': sum(row['matched']['passed'] for row in after),
               'swap_new': sum(row['swap_new']['passed'] for row in after),
               'swap_mentions': sum(row['swap_value'] in row['swap_new']['answer'] for row in after),
               'count': len(after)}
    store_json(args.output / 'result.json', {'sources': bound, 'individual_sha256': individual,
                                             'weight_sha256': sha(weights), 'summary': summary, 'rows': after,
                                             'training': len(plans), 'trace': trace, 'memory': memory,
                                             'seconds': perf_counter() - started})
    print(summary, memory['summary'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('candidates', nargs=2, type=Path)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=192)
    arguments = parser.parse_args()
    if arguments.steps < 1 or arguments.output.exists():
        parser.error('Choose positive updates and a new output directory')
    main(arguments)
