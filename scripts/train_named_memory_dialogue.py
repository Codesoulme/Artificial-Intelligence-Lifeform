"""Train ordinary-only residuals on multi-turn replies and contrast stale histories."""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.audit_local_named_chat import CASES, run
from scripts.audit_qwen_packed_reply_20261122 import HELD, evaluate, records
from scripts.spora_named_memory_chat import MODEL, load_full_streamed, sha
from spora.integrated_checkpoint import load_integrated_runtime
from spora.language_training import frozen_digest
from spora.pretrained_attention import SPORAPretrainedBlock, SPORAResidualBlock
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
SEED = 99112
LAYERS = (22, 24, 27)
FOUNDATIONS = ROOT / 'data/chinese-foundations.json'
CORRECTIONS = ROOT / 'data/dialogue-correction-training-v1.json'
REVIEWED = ROOT / 'runs/teaching-curriculum-v5/teacher.jsonl'


def source_dialogues():
    foundations = json.loads(FOUNDATIONS.read_text(encoding='utf-8'))['dialogues']
    correction = json.loads(CORRECTIONS.read_text(encoding='utf-8'))['families']
    reviewed = [json.loads(line) for line in REVIEWED.read_text(encoding='utf-8').splitlines()]
    groups = {'foundations': [row['turns'] for row in foundations], 'correction': [],
              'reviewed': []}
    for rows in correction.values():
        for case in rows:
            groups['correction'].append(list(zip(case[::2], case[1::2], strict=True)))
    for row in reviewed:
        messages = row['messages']
        if len(messages) % 2 or [message['role'] for message in messages] != ['user', 'assistant'] * (len(messages) // 2):
            continue
        texts = [message['content'] for message in messages]
        groups['reviewed'].append(list(zip(texts[::2], texts[1::2], strict=True)))
    return groups


def tokens(tokenizer, history, question):
    messages = [{'role': role, 'content': text} for user, answer in history
                for role, text in (('user', user), ('assistant', answer))]
    messages.append({'role': 'user', 'content': question})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(prompt + '</think>\n\n', add_special_tokens=False)


def examples(tokenizer):
    held = {question for _, turns in CASES for question, _ in turns}
    groups = {}
    for group, dialogues in source_dialogues().items():
        rows = []
        for conversation in dialogues:
            if any(question in held for question, _ in conversation):
                raise ValueError('Training question occurs in independent evaluation')
            for index, (question, answer) in enumerate(conversation):
                history = list(conversation[:index])
                target = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
                if not 2 <= len(target) <= 80:
                    continue
                prompt = tokens(tokenizer, history, question)
                if len(prompt) + len(target) > 512:
                    continue
                echo = tokenizer.encode(question, add_special_tokens=False)
                stale = tokenizer.encode(history[-1][1], add_special_tokens=False) if history else []
                rows.append((prompt, target, echo, stale))
                if history and group == 'correction':
                    corrupted = [(user, user) for user, _ in history]
                    alternate = tokens(tokenizer, corrupted, question)
                    if len(alternate) + len(target) <= 512:
                        rows.append((alternate, target, echo, stale))
        if not rows:
            raise ValueError(f'No training requests for {group}')
        groups[group] = rows
    return groups


def forbidden_prefixes(target, phrases):
    negatives = set()
    for phrase in phrases:
        if len(phrase) < 3:
            continue
        for index, token in enumerate(phrase[:len(target)]):
            if token != target[index]:
                negatives.add((index, token))
                break
    return sorted(negatives)


def fit(decoder, blocks, groups, steps):
    parameters = [weight for block in blocks for weight in (block.down.weight, block.up.weight)]
    optimizer = torch.optim.AdamW(parameters, lr=2e-4, weight_decay=.01)
    rng = random.Random(SEED)
    selected = ('correction', 'foundations', 'reviewed', 'correction', 'foundations')
    trace = []
    running = []
    for step in range(steps):
        group = selected[step % len(selected)]
        prompt, target, echo, stale = rng.choice(groups[group])
        labels = torch.tensor(target, device='cuda')
        ids = torch.tensor([prompt + target[:-1]], device='cuda')
        logits = decoder(input_ids=ids, use_cache=False, logits_to_keep=len(target)).logits[0].float()
        loss = F.cross_entropy(logits, labels)
        negatives = forbidden_prefixes(target, (echo, stale) if group == 'correction' else (echo,))
        if negatives:
            positions, ids = zip(*negatives, strict=True)
            probabilities = logits.softmax(-1)[list(positions), list(ids)]
            loss = loss - .15 * torch.log1p(-probabilities.clamp(max=.9999)).mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite dialogue training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        running.append(float(loss.detach()))
        if (step + 1) % 32 == 0 or step + 1 == steps:
            trace.append({'step': step + 1, 'mean_loss': sum(running) / len(running)})
            print(trace[-1], flush=True)
            running.clear()
    return trace


def main(checkpoint, output, steps):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    protected = {path: sha(path) for path in (checkpoint, FOUNDATIONS, CORRECTIONS, REVIEWED)}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    groups = examples(tokenizer)
    print('Training requests', {name: len(rows) for name, rows in groups.items()}, flush=True)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL)
    decoder = runtime.decoder.eval().requires_grad_(False)
    state_digest = frozen_digest(runtime.organism, runtime.state)
    parents = decoder.base.model.layers
    blocks = []
    for index in LAYERS:
        if type(parents[index]) is not SPORAPretrainedBlock or index in decoder.layer_indices:
            raise ValueError('Expected an unmodified compatible layer outside the memory readers')
        block = SPORAResidualBlock(parents[index], rank=16, ordinary_only=True,
                                   memory_context=decoder._memory_context).eval()
        parents[index] = block
        blocks.append(block)
    trace = fit(decoder, blocks, groups, steps)
    decoder.requires_grad_(False)
    dialogue = run(decoder, tokenizer)
    print('Independent dialogue', {key: value for key, value in dialogue.items() if key != 'rows'}, flush=True)
    memory = evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))
    print('Memory control', memory['summary'], flush=True)
    if any(sha(path) != digest for path, digest in protected.items()) or frozen_digest(runtime.organism, runtime.state) != state_digest:
        raise ValueError('Experiment changed protected inputs')
    output.mkdir(parents=True, exist_ok=False)
    candidate = runtime.save(output / 'integrated.pt')
    report = {'candidate_sha256': sha(candidate), 'sources': {str(path): digest for path, digest in protected.items()},
              'steps': steps, 'layers': list(LAYERS), 'seed': SEED, 'trace': trace,
              'training_requests': {name: len(rows) for name, rows in groups.items()},
              'dialogue': dialogue, 'memory': memory}
    with (output / 'result.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=256)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    main(**vars(args))
