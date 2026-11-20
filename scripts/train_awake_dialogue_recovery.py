"""Recover language after full decoder grafts with dialogue and neural-memory replay."""

import argparse
import math
import random
import time
from pathlib import Path

from audit_awake_feedforward_memory import history_cases
from audit_awake_multiturn_layers import CASES, CEVAL_SUBJECTS, ceval_questions, conversation, encode, score_ceval, score_memory
from audit_awake_replacement_memory import ALLOWED
from deepseek_output_aligned_memory import MODEL, ROOT, SOURCE, read, sha, torch, write
from spora.language_training import atomic_save
from train_awake_absolute_attention import replay_requests
from train_awake_replacement_replay import answer_logits, divergence, load_layer_pair, switch


CORPUS = ROOT / 'data/chinese-foundations.json'
SEED = 60327
LIMIT = 256


def dialogue_cases(tokenizer):
    examples = []
    for dialogue in read(CORPUS)['dialogues']:
        messages = []
        for question, answer in dialogue['turns']:
            messages.append(dict(role='user', content=question))
            prompt = encode(tokenizer, messages)
            target = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
            if len(prompt) + len(target) <= LIMIT:
                examples.append((question, prompt, target))
            messages.append(dict(role='assistant', content=answer))
    held_questions = {question for group in CASES for question in group[::2]}
    if held_questions & {question for question, _, _ in examples}:
        raise ValueError('Training and development prompts overlap')
    return examples


def memory_logits(decoder, prompt, target, memory):
    inputs = torch.tensor([prompt + target[:-1]], device='cuda')
    return decoder(input_ids=inputs, logits_to_keep=len(target), **memory).logits[0].float()


@torch.no_grad()
def memory_targets(decoder, tokenizer, organism):
    rows = []
    eos = tokenizer.eos_token_id
    for prompt, memory in replay_requests(organism, tokenizer):
        inputs = torch.tensor([prompt], device='cuda')
        generated = decoder.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                     do_sample=False, temperature=None, top_p=None, max_new_tokens=8,
                                     pad_token_id=tokenizer.pad_token_id, eos_token_id=eos,
                                     **dict(memory, use_cache=True))
        observed = generated[0, len(prompt):].tolist()
        if observed:
            target = memory_logits(decoder, prompt, observed, memory).log_softmax(-1).cpu()
            rows.append((prompt, observed, memory, target))
    return rows


@torch.no_grad()
def evaluate(decoder, tokenizer, memory, questions):
    turns = [conversation(decoder, tokenizer, case, LIMIT, decoding=dict(limit=48)) for case in CASES]
    recalled = score_memory(decoder, tokenizer, memory)
    choices = score_ceval(decoder, questions, min(layer.max_length for layer in decoder.base.model.layers
                                                 if hasattr(layer, 'max_length')))
    summary = dict(ceval=sum(row['choice'] == row['expected'] for row in choices),
                   ceval_total=sum(row['choice'] is not None for row in choices),
                   dialogue_correct=sum(row['correct'] for group in turns for row in group),
                   dialogue_eos=sum(row['eos'] for group in turns for row in group),
                   memory_correct=sum(row['correct'] for row in recalled))
    return dict(summary=summary, dialogue=turns, memory=recalled, ceval=choices)


def main(output, ceval, steps, teacher_weight):
    sources = (SOURCE, MODEL / 'model.safetensors', MODEL / 'tokenizer.json', MODEL / 'tokenizer_config.json',
               CORPUS, *(ceval / f'{subject}.parquet' for subject in CEVAL_SUBJECTS))
    protected = {source: sha(source) for source in sources}
    started = time.perf_counter()
    decoder, tokenizer, originals, replacements, organism, _ = load_layer_pair()
    train = dialogue_cases(tokenizer)
    _, held_memory = history_cases(organism)
    questions = list(ceval_questions(ceval, tokenizer))
    replay = memory_targets(decoder, tokenizer, organism)
    if not train or not replay:
        raise ValueError('Expected both dialogue and memory examples')
    before = evaluate(decoder, tokenizer, held_memory, questions)
    print('Before', before['summary'], flush=True)

    switch(decoder, originals)
    with torch.no_grad():
        teacher = [answer_logits(decoder, prompt, target).log_softmax(-1).cpu() for _, prompt, target in train]
    switch(decoder, replacements)
    editable = {index: {name: param.requires_grad_(True) for name, param in block.named_parameters() if name in ALLOWED}
                for index, block in replacements.items()}
    parameters = [param for group in editable.values() for param in group.values()]
    initial = {index: {name: param.detach().cpu().clone() for name, param in group.items()}
               for index, group in editable.items()}
    optimizer = torch.optim.AdamW(parameters, lr=.002, weight_decay=0.)
    rng = random.Random(SEED)
    sampling_order = list(range(len(train)))
    losses = []
    for step in range(steps):
        if step % len(train) == 0:
            rng.shuffle(sampling_order)
        index = sampling_order[step % len(train)]
        _, prompt, target = train[index]
        optimizer.zero_grad(set_to_none=True)
        logits = answer_logits(decoder, prompt, target)
        supervised = torch.nn.functional.cross_entropy(logits, torch.tensor(target, device='cuda'))
        teacher_loss = divergence(logits, teacher[index])
        (supervised + teacher_weight * teacher_loss).backward()
        preservation = None
        if step % 3 == 0:
            memory_prompt, memory_tail, kwargs, reference = replay[(step // 3) % len(replay)]
            preservation = divergence(memory_logits(decoder, memory_prompt, memory_tail, kwargs), reference)
            preservation.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(dict(reference=supervised.item(), teacher_kl=teacher_loss.item(),
                           memory_kl=None if preservation is None else preservation.item()))
        if (step + 1) % 12 == 0:
            window = losses[-12:]
            average = {name: sum(values) / len(values) for name in losses[-1]
                       if (values := [row[name] for row in window if row[name] is not None])}
            print('Step', step + 1, average, flush=True)
    after = evaluate(decoder, tokenizer, held_memory, questions)
    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError('A protected source changed during training')
    delta = {index: {name: param.detach().cpu() - initial[index][name] for name, param in group.items()}
             for index, group in editable.items()}
    path = output / 'delta.pt'
    atomic_save(dict(source_sha256=protected[SOURCE], delta=delta), path)
    write(output / 'result.json', dict(before=before, after=after, steps=steps, teacher_weight=teacher_weight, losses=losses,
                                      training_examples=len(train), memory_examples=len(replay),
                                      delta_sha256=sha(path), seconds=time.perf_counter()-started,
                                      peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                                      sources={str(source): digest for source, digest in protected.items()}))
    print('After', after['summary'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ceval', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=72)
    parser.add_argument('--teacher-weight', type=float, default=.5)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('Expected at least one step')
    if not math.isfinite(args.teacher_weight) or args.teacher_weight < 0:
        parser.error('Expected a finite nonnegative teacher weight')
    args.output.mkdir()
    main(args.output, args.ceval, args.steps, args.teacher_weight)
