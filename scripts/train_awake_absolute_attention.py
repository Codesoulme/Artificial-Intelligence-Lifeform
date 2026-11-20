"""Fit SPORA-style learned positions inside a pretrained attention transplant."""

import argparse
import random
import time
from pathlib import Path

from audit_awake_feedforward_memory import history_cases
from audit_awake_multiturn_layers import CASES, conversation, encode, score_memory
from audit_awake_language_layers import load_layer_pair
from deepseek_output_aligned_memory import MODEL, SOURCE, sha, torch, write
from scripts.deepseek_spora_memory_answer_distill import _memory_args
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import grow
from spora.internal_memory import observe_sourced
from spora.language_training import atomic_save
from spora.pretrained_attention import AbsoluteAttention
from train_awake_partial_layer import TRAIN_FILES, corpus


LAYER = 18
MAX_POSITION = 256
SEED = 316201


@torch.no_grad()
def capture(decoder, source, requests):
    rows = []

    def transfer(value):
        return value.detach().cpu() if value is not None else None

    def record(_module, _args, arguments, output):
        rows.append((transfer(arguments['hidden_states']),
                     tuple(map(transfer, arguments['position_embeddings'])),
                     transfer(arguments['attention_mask']), transfer(arguments['position_ids']),
                     transfer(output[0])))

    hook = source.register_forward_hook(record, with_kwargs=True)
    try:
        for indices, kwargs in requests:
            decoder(input_ids=torch.tensor([indices], device='cuda'), logits_to_keep=1, **dict(kwargs, use_cache=False))
    finally:
        hook.remove()
    return rows


def replay_requests(organism, tokenizer, *, value_offset=0, with_answers=False):
    state = fresh_controlled_state(organism)
    requests = []
    for index, name in enumerate(('明珂', '茹川', '洛源', '青洵', '江禾', '孟亭'), start=5):
        event_time = state.cursor
        earlier, current = (index + value_offset) % 8, (index + value_offset + 3) % 8
        for value in (earlier, current):
            if state.cursor >= len(state.pages) * organism.config.time_count:
                state = grow(organism, state)
            state, event = observe_sourced(organism, state, index, value, 'direct', 1.)
            if not event['current_write']:
                raise ValueError('Controlled memory write rejected')
        for intent, question in (
            ('current', f'{name}最近通常喝哪种饮料？'),
            ('current', f'现在登记的{name}的饮品偏好是什么？'),
            ('past', f'{name}更新之前最常喝什么？'),
            ('past', f'最早那条记录里{name}喜欢喝什么？'),
        ):
            prompt = encode(tokenizer, [dict(role='user', content=question)])
            row = dict(key=index, intent=intent, event_time=event_time)
            requests.append((prompt, row, earlier if intent == 'past' else current))
    if with_answers:
        return [(prompt, _memory_args(organism, state, row), value) for prompt, row, value in requests]
    return [(prompt, _memory_args(organism, state, row)) for prompt, row, _ in requests]


def prediction(model, row):
    hidden, rotary, mask, positions, expected = row
    values = model(hidden.cuda().float(), tuple(component.cuda() for component in rotary),
                   None if mask is None else mask.cuda(), position_ids=positions.cuda())[0]
    return values, expected.cuda().float()


@torch.no_grad()
def relative_error(model, rows):
    error, energy = 0., 0.
    for row in rows:
        actual, expected = prediction(model, row)
        error += (actual - expected).square().sum().item()
        energy += expected.square().sum().item()
    return error / energy


def main(output, steps, memory_replay=False):
    protected = {path: sha(path) for path in (SOURCE, MODEL / 'model.safetensors', *TRAIN_FILES)}
    started = time.perf_counter()
    decoder, tokenizer, _, _, organism, _ = load_layer_pair()
    layer = decoder.base.model.layers[LAYER]
    source = layer.self_attn
    requests = [(tokenizer.encode(text, add_special_tokens=False)[:192], {}) for text in corpus()]
    captured = capture(decoder, source, requests)
    held, training = captured[:8], captured[8:]
    if not training or len(held) != 8:
        raise ValueError('Expected separate training and held-out sequences')
    _, memory_cases = history_cases(organism)
    replay = capture(decoder, source, replay_requests(organism, tokenizer)) if memory_replay else []

    def score(attention):
        layer.self_attn = attention
        try:
            dialogues = [conversation(decoder, tokenizer, case, MAX_POSITION) for case in CASES]
            memory = score_memory(decoder, tokenizer, memory_cases)
        finally:
            layer.self_attn = source
        return dict(dialogues=dialogues, memory=memory,
                    strict_turns=sum(turn['correct'] for group in dialogues for turn in group),
                    eos=sum(turn['eos'] for group in dialogues for turn in group),
                    memory_correct=sum(row['correct'] for row in memory))

    candidate = AbsoluteAttention(source, MAX_POSITION).float().eval()
    candidate.requires_grad_(False)
    candidate.position.requires_grad_(True)
    candidate.out.requires_grad_(True)
    untrained_error = relative_error(candidate, held)
    before = score(source)
    optimizer = torch.optim.AdamW([*candidate.position.parameters(), *candidate.out.parameters()], lr=2e-4)
    rng = random.Random(SEED)
    for iteration in range(steps):
        candidate.rotary_fraction = .75 * max(0., 1. - 2. * iteration / steps)
        pool = replay if replay and iteration % 3 == 0 else training
        actual, expected = prediction(candidate, rng.choice(pool))
        loss = (actual - expected).square().mean() / expected.square().mean().clamp_min(1e-8)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*candidate.position.parameters(), *candidate.out.parameters()], 1.,
                                      error_if_nonfinite=True)
        optimizer.step()
        if (iteration + 1) % 16 == 0:
            print('Step', iteration + 1, 'rotary fraction', round(candidate.rotary_fraction, 3),
                  'relative error', round(loss.item(), 6), flush=True)
    candidate.rotary_fraction = 0.
    trained_error = relative_error(candidate, held)
    after = score(candidate.requires_grad_(False).half())
    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError('A protected source changed during the run')
    artifact = output / 'attention.pt'
    atomic_save(dict(layer=LAYER, format='spora-absolute-attention-v1', max_position=MAX_POSITION,
                     source=protected[SOURCE], state={name: value.cpu() for name, value in candidate.state_dict().items()}),
                artifact)
    result = dict(layer=LAYER, steps=steps, held_sequences=len(held), training_sequences=len(training),
                  memory_replay_sequences=len(replay),
                  untrained_error=untrained_error, trained_error=trained_error, before=before, after=after,
                  sources={str(path): digest for path, digest in protected.items()}, candidate_sha256=sha(artifact),
                  seconds=time.perf_counter()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated())
    write(output / 'result.json', result)
    print({name: result[name] for name in ('untrained_error', 'trained_error', 'seconds')},
          {name: {key: result[name][key] for key in ('strict_turns', 'eos', 'memory_correct')}
           for name in ('before', 'after')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--memory-replay', action='store_true', help='Mix memory-conditioned attention targets into training')
    args = parser.parse_args()
    if args.steps < 2:
        parser.error('Expected at least two alignment steps')
    args.output.mkdir()
    main(args.output, args.steps, args.memory_replay)
