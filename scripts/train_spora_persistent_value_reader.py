"""Train a small value projection inside native attention using withheld Chinese names."""

import argparse
from pathlib import Path
from time import perf_counter

from scripts import audit_spora_persistent_attention_slots as slots
from scripts.audit_spora_version_reply_actions import EVENTS, FACTS, QUESTIONS, TRAINING, TRAIN_NAMES
from spora.research_recurrent_decode import attention_block, load_recurrent_replacements


torch = slots.torch
LAYER = 23
RANK = 8
BIAS = 4
SEED = 40623
MAX_REPLY = 8


class ValueProjection(torch.nn.Module):
    """A shared low-rank residual on saved value vectors."""

    def __init__(self, width):
        super().__init__()
        self.down = torch.nn.Linear(width, RANK, bias=False)
        self.up = torch.nn.Linear(RANK, width, bias=False)
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, saved_value):
        return saved_value + self.up(torch.nn.functional.gelu(self.down(saved_value.float()))).to(saved_value)


def profiles():
    train = []
    for domain, (field, _, values) in TRAINING.items():
        for index, name in enumerate(TRAIN_NAMES[:4]):
            train.append((domain, name, f'{name}{field}是哪一个？只给名称。',
                          (values[index], values[(index + 1) % len(values)])))
    held = [(domain, name, QUESTIONS[domain][0].format(name=name), (old, current))
            for domain, ((name, old, current), _) in EVENTS.items()]
    if ({name for _, name, _, _ in train} & {name for _, name, _, _ in held}
            or {v for *_, values in train for v in values} & {v for *_, values in held for v in values}):
        raise ValueError('Training and holdout overlap')
    return train, held


@torch.no_grad()
def build_rows(decoder, tokenizer, attention, cases):
    rows = []
    for domain, name, question, values in cases:
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
                                               tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(prompt + '</think>\n\n', add_special_tokens=False)
        versions = []
        for value in values:
            if value in question:
                raise ValueError('A stored value appears in the question')
            record = {'role': 'user', 'content': FACTS[domain].format(name=name, value=value)}
            written = tokenizer.apply_chat_template([record], tokenize=False, add_generation_prompt=False)
            record_ids = tokenizer.encode(written, add_special_tokens=False)
            labels = tokenizer.encode(value, add_special_tokens=False)
            if not 1 <= len(record_ids) <= slots.SLOT_BUDGET or not 1 <= len(labels) < MAX_REPLY:
                raise ValueError('A record or answer exceeds the experiment budget')
            matrix = slots.capture_record(decoder, attention, record_ids)
            versions.append({'value': value, 'labels': [*labels, tokenizer.eos_token_id],
                             'slots': tuple(x.to(decoder.base.device) for x in matrix)})
        rows.append({'domain': domain, 'name': name, 'question': question, 'ids': ids, 'versions': versions})
    return rows


def scores(decoder, attention, ids, memory, head=None):
    if memory is None:
        return slots.read_attention(decoder, ids, ())
    key, stored_value = memory
    value = stored_value if head is None else head(stored_value)
    return slots.read_attention(decoder, ids, ((attention, (key, value)),), bias=BIAS)


def fit(decoder, attention, head, rows, steps):
    samples = [(row['ids'] + version['labels'][:position], version['slots'], token)
               for row in rows for version in row['versions']
               for position, token in enumerate(version['labels'])]
    optimizer = torch.optim.AdamW(head.parameters(), lr=.004)
    rng = torch.Generator().manual_seed(SEED)
    losses = []
    schedule = []
    while len(schedule) < steps:
        schedule.extend(torch.randperm(len(samples), generator=rng).tolist())
    for step, index in enumerate(schedule[:steps], 1):
        ids, memory, target = samples[index]
        loss = -scores(decoder, attention, ids, memory, head)[target]
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite value projection loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 1 or step % 32 == 0 or step == steps:
            print('fit', step, 'mean loss', sum(losses[-32:]) / len(losses[-32:]), flush=True)
    return {'losses': losses, 'supervised_positions': len(samples), 'positions_sampled': len(set(schedule[:steps]))}


@torch.no_grad()
def generate(decoder, attention, row, memory, eos, head=None):
    generated = []
    for _ in range(MAX_REPLY):
        token = int(scores(decoder, attention, row['ids'] + generated, memory, head).argmax())
        generated.append(token)
        if token == eos:
            break
    return generated


@torch.no_grad()
def evaluate(decoder, tokenizer, attention, head, rows):
    results = []
    eos = tokenizer.eos_token_id
    for row in rows:
        empty = generate(decoder, attention, row, None, eos)
        versions = []
        for version in row['versions']:
            memory = version['slots']
            samples = {'empty': empty, 'native': generate(decoder, attention, row, memory, eos),
                       'adapted': generate(decoder, attention, row, memory, eos, head)}
            versions.append({
                'target': version['value'], 'target_ids': version['labels'],
                'outputs': {mode: {'ids': ids, 'text': tokenizer.decode(ids, skip_special_tokens=True),
                                   'exact': ids == version['labels']}
                            for mode, ids in samples.items()},
            })
        results.append({**{key: row[key] for key in ('domain', 'name', 'question')}, 'versions': versions})
        print('held', row['domain'], [version['outputs']['adapted']['text'] for version in versions], flush=True)
    return results


def main(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    sources = {path: slots.sha(path) for path in (
        args.checkpoint, Path(__file__), Path(slots.__file__), slots.MODEL / 'model.safetensors',
        slots.MODEL / 'tokenizer.json', slots.MODEL / 'tokenizer_config.json',
        *(directory / name for directory in args.candidates for name in ('attention.pt', 'comparison.json')),
    )}
    tokenizer = slots.AutoTokenizer.from_pretrained(slots.MODEL, local_files_only=True, trust_remote_code=False)
    base = slots.load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = slots.load_integrated_runtime(args.checkpoint, base, slots.MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    slots.extend_positions(decoder, rotary, slots.query_audit.POSITIONS)
    torch.cuda.empty_cache()
    base.lm_head.to(base.device)
    individual = slots.frozen_digest(runtime.organism, runtime.state)
    replacements = load_recurrent_replacements(decoder, args.candidates, sources[args.checkpoint])
    if tuple(replacements) != (19, 22):
        raise ValueError('Expected the established recurrent layout')
    attention = attention_block(decoder, LAYER).self_attn
    head = ValueProjection(attention.head_dim).to(base.device)
    train, held = profiles()
    with slots.using(replacements, 'recurrent'):
        training = build_rows(decoder, tokenizer, attention, train)
        holdout = build_rows(decoder, tokenizer, attention, held)
        example = training[0]
        with torch.no_grad():
            memory = example['versions'][0]['slots']
            if not torch.equal(scores(decoder, attention, example['ids'], memory),
                               scores(decoder, attention, example['ids'], memory, head)):
                raise RuntimeError('Initialization changed the native reader')
            empty = [scores(decoder, attention, row['ids'], None).cpu() for row in holdout]
        trace = fit(decoder, attention, head, training, args.steps)
        with torch.no_grad():
            if any(not torch.equal(before, scores(decoder, attention, row['ids'], None).cpu())
                   for before, row in zip(empty, holdout, strict=True)):
                raise RuntimeError('Training changed no-memory outputs')
        slots.atomic_save({key: value.cpu() for key, value in head.state_dict().items()}, args.output / 'value.pt')
        results = evaluate(decoder, tokenizer, attention, head, holdout)
    if individual != slots.frozen_digest(runtime.organism, runtime.state) or any(
        slots.sha(path) != digest for path, digest in sources.items()
    ):
        raise RuntimeError('A research input or individual changed during training')
    slots.store_json(args.output / 'result.json', {
        'layer': LAYER, 'rank': RANK, 'bias': BIAS, 'steps': args.steps, 'training': trace, 'held': results,
        'sources': {str(path): digest for path, digest in sources.items()}, 'individual_sha256': individual,
        'value_sha256': slots.sha(args.output / 'value.pt'), 'seconds': perf_counter() - started,
        'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('candidates', nargs=2, type=Path)
    parser.add_argument('--steps', type=int, default=128)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('Training requires at least one step')
    main(args)
