"""Distill whole-decoder predictions into one new query/key projection."""

import json
import random

import torch
from torch.nn import functional as F

from audit_awake_multiturn_layers import encode
from scripts.deepseek_semantic_value_pilot import WORDS
from train_awake_absolute_attention import replay_requests
from train_awake_partial_layer import TRAIN_FILES
from train_awake_replacement_replay import divergence, generate


TRAIN_TOPICS = frozenset(('constraint', 'reference', 'correction', 'count', 'format', 'rewriting',
                          'preference_change', 'list', 'spatial', 'sequence', 'boundary', 'translation'))
VALIDATION_TOPICS = frozenset(('sorting', 'calendar', 'category', 'units'))
ROLLOUT_TOPICS = frozenset(('constraint', 'reference', 'correction', 'count', 'preference_change', 'list'))


def answer_request(tokenizer, prompt, answer, controls):
    target = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
    return prompt + target[:-1], target, controls


def prepare_requests(runtime, tokenizer, excluded_questions):
    source = json.loads(TRAIN_FILES[0].read_text(encoding='utf-8'))
    ordinary, held = [], []
    for scenario in source['dialogues']:
        topic = scenario['topic']
        if topic not in TRAIN_TOPICS | VALIDATION_TOPICS:
            continue
        target = ordinary if topic in TRAIN_TOPICS else held
        history = []
        for question, answer in scenario['turns']:
            if question in excluded_questions:
                raise ValueError('A training dialogue overlaps the migration evaluation')
            history.append({'role': 'user', 'content': question})
            target.append(answer_request(tokenizer, encode(tokenizer, history), answer, {}))
            history.append({'role': 'assistant', 'content': answer})
    prose = []
    for text in json.loads(TRAIN_FILES[1].read_text(encoding='utf-8'))['train'][:16]:
        tokens = tokenizer.encode(text, add_special_tokens=False)[:112]
        if len(tokens) > 16:
            prose.append((tokens[:-1], tokens[-8:], {}))
    memory = [answer_request(tokenizer, prompt, WORDS[value], controls) for prompt, controls, value in
              replay_requests(runtime.organism, tokenizer, value_offset=4, with_answers=True)]
    if not all((ordinary, prose, memory, held)) or max(len(ids) for ids, _, _ in ordinary + held) > 192:
        raise ValueError('Whole-decoder calibration requires four pools and short conversations')
    return ordinary, prose, memory[4:], held, memory[:4]


def prediction(decoder, request):
    ids, target, controls = request
    logits = decoder(input_ids=torch.tensor([ids], device='cuda'), logits_to_keep=len(target),
                     **dict(controls, use_cache=False)).logits[0].float()
    return logits, torch.tensor(target, device='cuda')


@torch.no_grad()
def teacher_targets(decoder, requests):
    return [prediction(decoder, request)[0].log_softmax(-1).half().cpu() for request in requests]


@torch.no_grad()
def validation(decoder, requests, targets):
    divergences, scores = [], []
    for request, target in zip(requests, targets, strict=True):
        logits, labels = prediction(decoder, request)
        divergences.append(float(divergence(logits, target.float().log_softmax(-1))))
        scores.append(float(F.cross_entropy(logits, labels)))
    return {'teacher_kl': sum(divergences) / len(divergences), 'answer_nll': sum(scores) / len(scores)}


@torch.no_grad()
def student_requests(decoder, tokenizer, excluded_questions):
    scenarios = json.loads(TRAIN_FILES[0].read_text(encoding='utf-8'))['dialogues']
    rows = []
    for scenario in scenarios:
        if scenario['topic'] not in ROLLOUT_TOPICS:
            continue
        history = []
        for question, _ in scenario['turns']:
            if question in excluded_questions:
                raise ValueError('A student rollout overlaps the migration evaluation')
            history.append({'role': 'user', 'content': question})
            prompt = encode(tokenizer, history)
            tail = generate(decoder, tokenizer, prompt, 20)
            rows.append((prompt + tail[:-1], tail, {}))
            history.append({'role': 'assistant', 'content': tokenizer.decode(tail, skip_special_tokens=True).strip()})
    return rows


def fit_dialogue_predictions(runtime, tokenizer, layer, native, candidate, steps, excluded_questions, *,
                             student_prefixes=False):
    pools = prepare_requests(runtime, tokenizer, excluded_questions)
    decoder = runtime.decoder
    layer.self_attn = native
    targets = [teacher_targets(decoder, requests) for requests in pools]
    parameters = []
    for name in ('weight', 'bias'):
        master = torch.nn.Parameter(getattr(candidate, name).detach().float())
        setattr(candidate, name, master)
        parameters.append(master)
    layer.self_attn = candidate
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=0.)
    rng = random.Random(221831)

    def held():
        return {name: validation(decoder, requests, references) for name, requests, references in
                zip(('dialogue', 'memory'), pools[3:], targets[3:], strict=True)}

    def refresh_rollouts():
        requests = student_requests(decoder, tokenizer, excluded_questions)
        layer.self_attn = native
        try:
            return requests, teacher_targets(decoder, requests)
        finally:
            layer.self_attn = candidate

    training = list(zip(pools[:3], targets[:3], strict=True))
    schedule = (0, 0, 1, 2)
    try:
        before = held()
        rollout_summary = None
        if student_prefixes:
            initial = refresh_rollouts()
            training.append(initial)
            schedule = (3, 0, 3, 2, 1, 3, 0, 2)
            rollout_summary = {'initial_requests': len(initial[0]), 'initial_tokens': sum(len(row[1]) for row in initial[0]),
                               'before': validation(decoder, *initial)}
            print('Collected student dialogues', rollout_summary, flush=True)
        trace = []
        for step in range(steps):
            if student_prefixes and step == steps // 2:
                training[3] = refresh_rollouts()
                rollout_summary['refreshed_tokens'] = sum(len(row[1]) for row in training[3][0])
                rollout_summary['changed_requests'] = sum(a[:2] != b[:2] for a, b in
                                                           zip(initial[0], training[3][0], strict=True))
                print('Refreshed student dialogues', rollout_summary['changed_requests'], flush=True)
            pool = schedule[step % len(schedule)]
            requests, references = training[pool]
            index = rng.randrange(len(requests))
            logits, labels = prediction(decoder, requests[index])
            kl = divergence(logits, references[index].float().log_softmax(-1))
            loss = kl + .10 * F.cross_entropy(logits, labels) if pool == 0 else kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            if (step + 1) % 16 == 0 or step + 1 == steps:
                row = {'step': step + 1, 'teacher_kl': float(kl.detach()), 'loss': float(loss.detach())}
                trace.append(row)
                print('Whole-decoder alignment', row, flush=True)
        after = held()
        if rollout_summary is not None:
            rollout_summary['after'] = validation(decoder, *initial)
    finally:
        for name in ('weight', 'bias'):
            parameter = getattr(candidate, name).detach().to(native.qkv.weight.dtype)
            setattr(candidate, name, torch.nn.Parameter(parameter, requires_grad=False))
        layer.self_attn = native
    return {'steps': steps, 'train_dialogue': len(pools[0]), 'train_prose': len(pools[1]),
            'train_memory': len(pools[2]), 'validation_dialogue': len(pools[3]), 'validation_memory': len(pools[4]),
            'before': before, 'after': after, 'student_prefixes': rollout_summary, 'trace': trace}
