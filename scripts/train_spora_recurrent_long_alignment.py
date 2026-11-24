"""Align the recurrent token mixer on long dialogue and measure answer transfer."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import random

from scripts.audit_spora_recurrent_attention import capture, optimize
from scripts.benchmark_spora_recurrent_dialogue import (
    MODEL, SOURCE, AutoTokenizer, encode, exchange, extend_positions, frozen_digest, load_full_streamed,
    load_integrated_runtime, recall_prompts, run_dialogue, sha, torch,
)
from scripts.train_qwen_counterfactual_reply_20261122 import HELD, evaluate, records
from scripts.train_spora_dialogue_working_state import DATA, dialogue_data
from spora.language_training import atomic_save
from spora.research_recurrent_decode import load_recurrent_replacements, replace


ROOT = SOURCE.parents[1]
CORPUS = ROOT / 'runs/clean-zh-pretrain-v1/pretrain-train.jsonl'
MANIFEST = CORPUS.with_name('manifest.json')
SEED = 112429


def material(tokenizer, path, count):
    available = []
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            text = json.loads(line)['text']
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) < 125:
                continue
            snippet = tokenizer.decode(tokens[:100], skip_special_tokens=True).strip()
            if snippet and snippet not in available:
                available.append(snippet)
            if len(available) == count:
                return available
    raise ValueError('The existing Chinese corpus has too few independent passages')


def training_sequences(tokenizer, examples, fillers):
    rows = []
    for index, example in enumerate(examples):
        count = (0, 5, 11, 14)[index % 4]
        rng = random.Random(SEED + index)
        selected = rng.sample(fillers, count)
        initial_offset = index % (count + 1)
        update_offset = initial_offset + (index // 3) % (count - initial_offset + 1)
        history = []
        for offset in range(count + 1):
            if offset == initial_offset:
                exchange(history, example['first'][0], '好，我知道了。')
            if offset == update_offset:
                exchange(history, example['update'][0], '好，已经更改。')
            if offset < count:
                exchange(history, selected[offset], '这段也看到了。')
        history.append({'role': 'user', 'content': example['question']})
        prompt = encode(tokenizer, history)
        answer = tokenizer.encode(example['answer'], add_special_tokens=False)
        if len(prompt) + len(answer) >= 2048:
            raise ValueError('Alignment example exceeds the position table')
        rows.append((prompt + answer, len(prompt) - 1))
    return rows


def train(core, train_rows, device, steps, lr):
    core.train().requires_grad_(True)
    optimizer = torch.optim.AdamW(core.parameters(), lr=lr, weight_decay=0.)
    rng = random.Random(SEED)
    trace = []
    order = []
    for step in range(steps):
        if not order:
            order = list(range(len(train_rows)))
            rng.shuffle(order)
        item, answer_start = train_rows[order.pop()]
        qkv = tuple(tensor.to(device) for tensor in item['qkv'])
        target = item['mix'].to(device).float()
        output, _ = core(*qkv, batched_prefill=True)
        error = (output - target).square()
        # Preserve the general mixer and give extra weight to the positions that generate the answer.
        global_error = error.mean() / target.square().mean().clamp_min(1e-8)
        answer_error = error[:, :, answer_start:].mean() / target[:, :, answer_start:].square().mean().clamp_min(1e-8)
        loss = .35 * global_error + .65 * answer_error
        optimize(core, optimizer, loss)
        if (step + 1) % 16 == 0 or step + 1 == steps:
            trace.append({'step': step + 1, 'all_positions': float(global_error.detach()),
                          'answer_positions': float(answer_error.detach())})
    core.eval().requires_grad_(False)
    return trace


@torch.no_grad()
def alignment_score(core, rows, device):
    total = []
    for item in rows:
        projected, _ = core(*(value.to(device) for value in item['qkv']), batched_prefill=True)
        target = item['mix'].to(device).float()
        positions = (slice(None), slice(-16, None))
        total.append([float((projected[:, :, pos] - target[:, :, pos]).square().mean()
                            / target[:, :, pos].square().mean().clamp_min(1e-8)) for pos in positions])
    return {'all_positions': sum(row[0] for row in total) / len(total),
            'last_16_positions': sum(row[1] for row in total) / len(total)}


def run_candidate(runtime, tokenizer, candidates, prompts, open_cases):
    decoder = runtime.decoder
    memory = records(runtime.organism, HELD)
    with ExitStack() as installed:
        for block, candidate in candidates.values():
            installed.enter_context(replace(block, candidate))
        dialogue = run_dialogue(decoder, tokenizer, prompts, open_cases)
        recalled = evaluate(decoder, tokenizer, runtime.organism, memory)
    return dialogue, recalled


def main(args):
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    sources = [args.checkpoint, SOURCE, CORPUS, MANIFEST, DATA]
    sources.extend(directory / name for directory in args.candidates for name in ('attention.pt', 'comparison.json'))
    protected = {path: sha(path) for path in sources}
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if manifest['files'][CORPUS.name] != protected[CORPUS]:
        raise ValueError('Chinese corpus differs from its manifest')
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    examples = dialogue_data(json.loads(DATA.read_text(encoding='utf-8')), held=False, count=16)
    prompts = recall_prompts(tokenizer, source, extended=True)
    training = training_sequences(tokenizer, examples, material(tokenizer, CORPUS, 40))
    held = [encode(tokenizer, [*row['history'], {'role': 'user', 'content': row['question']}])
            for row in prompts if row['length'] == 'extended']
    if {tuple(ids[:start + 1]) for ids, start in training} & {tuple(ids) for ids in held}:
        raise ValueError('An extended recall sequence overlaps training')
    base = load_full_streamed()
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    device = decoder.base.device
    individual = frozen_digest(runtime.organism, runtime.state)
    extend_positions(decoder, rotary, 2048)
    bindings = {str(path): digest for path, digest in protected.items()}
    if args.verify:
        result = json.loads((args.output / 'result.json').read_text(encoding='utf-8'))
        if result['sources'] != bindings or result['individual_sha256'] != individual:
            raise ValueError('Alignment inputs differ from the original experiment')
        directories = [args.output / f'layer-{index}' for index in result['metrics']]
        aligned = load_recurrent_replacements(decoder, directories, protected[args.checkpoint])
        dialogue, recalled = run_candidate(runtime, tokenizer, aligned, prompts, source['open'])
        if (dialogue != result['dialogue'] or recalled != result['memory']['aligned']
                or frozen_digest(runtime.organism, runtime.state) != individual
                or any(sha(path) != digest for path, digest in protected.items())):
            raise ValueError('Reload changed an answer, input or individual state')
        store_json(args.output / 'reload.json', {'identical': True, 'recall': len(dialogue['recall']),
                                                'open_turns': sum(len(case['turns']) for case in dialogue['open']),
                                                'memory': recalled['summary']})
        print('Reloaded', len(dialogue['recall']), 'recall answers and', recalled['summary']['count'], 'persistent answers',
              flush=True)
        return
    candidates = load_recurrent_replacements(decoder, args.candidates, protected[args.checkpoint])
    metrics = {}
    for index, (block, candidate) in candidates.items():
        teacher_train, _ = capture(decoder, block.self_attn, [ids for ids, _ in training])
        teacher_held, _ = capture(decoder, block.self_attn, held)
        before = alignment_score(candidate.core, teacher_held, device)
        examples = list(zip(teacher_train, (start for _, start in training), strict=True))
        trace = train(candidate.core, examples, device, args.steps, args.learning_rate)
        after = alignment_score(candidate.core, teacher_held, device)
        metrics[index] = {'before': before, 'after': after, 'training': trace}
        print(index, 'before', before, 'after', after, flush=True)
        del examples, teacher_train, teacher_held
    memory = records(runtime.organism, HELD)
    original_memory = evaluate(decoder, tokenizer, runtime.organism, memory)
    dialogue, aligned_memory = run_candidate(runtime, tokenizer, candidates, prompts, source['open'])
    if (frozen_digest(runtime.organism, runtime.state) != individual
            or any(sha(path) != digest for path, digest in protected.items())):
        raise ValueError('An alignment input changed')
    for index, (_, candidate) in candidates.items():
        destination = args.output / f'layer-{index}'
        destination.mkdir()
        path = destination / 'attention.pt'
        atomic_save({'format': 'spora-recurrent-attention-v1', 'layer': index, 'block': candidate.core.block,
                     'checkpoint_sha256': protected[args.checkpoint], 'core': candidate.core.cpu().state_dict()}, path)
        store_json(destination / 'comparison.json', {'layer': index, 'block': candidate.core.block, 'sources': bindings,
                                                     'candidate_sha256': sha(path), 'individual_sha256': individual,
                                                     'calibration': metrics[index]})
    result = {'train_lengths': [len(ids) for ids, _ in training], 'training_steps': args.steps, 'metrics': metrics,
              'individual_sha256': individual, 'sources': bindings, 'dialogue': dialogue,
              'memory': {'original': original_memory, 'aligned': aligned_memory,
                         'identical': original_memory == aligned_memory}}
    store_json(args.output / 'result.json', result)
    print({key: value for key, value in dialogue.items() if key not in ('recall', 'open')}, flush=True)


def store_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('candidates', nargs=2, type=Path)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.steps < 1 or not 0 < args.learning_rate <= .01:
        parser.error('Choose positive steps and a learning rate through .01')
    main(args)
