"""Distill a complete GELU feed-forward layer while preserving native recall."""

import argparse
import random
import time
from pathlib import Path

from audit_awake_block_components import LAYER
from audit_awake_feedforward_memory import HISTORY, history_cases
from audit_awake_multiturn_layers import CEVAL_SUBJECTS, ceval_questions, score_memory
from deepseek_output_aligned_memory import MODEL, SOURCE, read, sha, torch, write
from scripts.deepseek_semantic_value_pilot import WORDS
from spora.language_training import atomic_save
from train_awake_absolute_attention import replay_requests
from train_awake_dialogue_recovery import dialogue_cases, evaluate, memory_logits, memory_targets
from train_awake_partial_layer import SPORAFeedForward, TRAIN_FILES, collect, corpus, relative_mse
from train_awake_replacement_replay import answer_logits, divergence, load_layer_pair


SEED = 51381
OUTPUT_STEPS = 72


def load_student(path, layer, source_sha256, digest, reference):
    report = read(path.parent / 'result.json')
    expected_digest = report.get('checkpoints', {}).get(path.stem, report['candidate_sha256'])
    if digest != expected_digest:
        raise ValueError('Feed-forward checkpoint digest mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['layer'] != layer or payload['source_sha256'] != source_sha256:
        raise ValueError('Feed-forward checkpoint provenance mismatch')
    width = reference.gate_proj.in_features
    network = torch.nn.Sequential(torch.nn.Linear(width, width * 4), torch.nn.GELU(),
                                  torch.nn.Linear(width * 4, width))
    student = SPORAFeedForward(network)
    student.load_state_dict(payload['feedforward'], strict=True)
    return student.to(device=reference.gate_proj.weight.device)


@torch.no_grad()
def collect_replay(decoder, teacher, rows):
    features, labels = [], []

    def record(_module, args, output):
        features.append(args[0][0].float())
        labels.append(output[0].float())

    hook = teacher.register_forward_hook(record)
    try:
        for prompt, target, memory, _ in rows:
            memory_logits(decoder, prompt, target, memory)
    finally:
        hook.remove()
    return torch.cat(features), torch.cat(labels)


def fit(student, ordinary, replay, steps):
    generator = torch.Generator(device=ordinary[0].device).manual_seed(SEED)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=.01)
    for _ in range(steps):
        batch = []
        for (features, labels), count in zip((ordinary, replay), (64, 32), strict=True):
            indices = torch.randint(len(features), (count,), generator=generator, device=features.device)
            batch.append((features[indices], labels[indices]))
        x, y = map(torch.cat, zip(*batch, strict=True))
        loss = (student(x) - y).square().mean() / y.square().mean().clamp_min(1e-8)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    return student


def align_outputs(decoder, student, language, replay, *, paired_targets=False):
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-5, weight_decay=.01)
    rng = random.Random(SEED)
    stats = []
    for step in range(OUTPUT_STEPS):
        memory_step = step % 2 == 0
        prompt, tail, kwargs, teacher = rng.choice(replay if memory_step else language)
        logits = memory_logits(decoder, prompt, tail, kwargs)
        loss = divergence(logits, teacher)
        if paired_targets and memory_step:
            labels = torch.tensor(tail, device=logits.device)
            loss = .5 * loss + torch.nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        stats.append(loss.item())
        if (step + 1) % 24 == 0:
            print('Output alignment', step + 1, 'mean loss', round(sum(stats[-24:]) / 24, 5), flush=True)
    return stats


@torch.no_grad()
def paired_targets(decoder, tokenizer, organism, *, history_pairs=()):
    rows = []
    for side, offset in enumerate((0, 2)):
        requests = replay_requests(organism, tokenizer, value_offset=offset, with_answers=True)
        requests = [(prompt, kwargs, WORDS[value]) for prompt, kwargs, value in requests]
        if history_pairs:
            requests.extend(history_pairs[side])
        for prompt, kwargs, value in requests:
            tail = tokenizer.encode(value, add_special_tokens=False) + [tokenizer.eos_token_id]
            teacher = memory_logits(decoder, prompt, tail, kwargs).log_softmax(-1).cpu()
            rows.append((prompt, tail, kwargs, teacher))
    return rows


def save_student(output, name, student, source_digest):
    path = output / f'{name}.pt'
    atomic_save(dict(layer=LAYER, source_sha256=source_digest,
                     feedforward={name: value.detach().cpu() for name, value in student.state_dict().items()}), path)
    return sha(path)


def main(output, start, ceval, steps, paired, verify):
    sources = (SOURCE, MODEL / 'model.safetensors', MODEL / 'tokenizer.json', MODEL / 'tokenizer_config.json',
               start, *TRAIN_FILES, *(ceval / f'{subject}.parquet' for subject in CEVAL_SUBJECTS))
    protected = {path: sha(path) for path in sources}
    started = time.perf_counter()
    decoder, tokenizer, _, _, organism, _ = load_layer_pair()
    layer = decoder.base.model.layers[LAYER]
    teacher = layer.mlp
    _, memory = history_cases(organism)
    alt_history = tuple((key, name, (earlier + 2) % 8, (current + 2) % 8)
                        for key, name, earlier, current in HISTORY)
    _, alt_memory = history_cases(organism, alt_history)
    questions = list(ceval_questions(ceval, tokenizer))
    results = {}

    def score(name):
        result = evaluate(decoder, tokenizer, memory, questions)
        alternate = score_memory(decoder, tokenizer, alt_memory)
        pairs = zip(result['memory'], alternate, strict=True)
        result['summary']['memory_switches'] = sum(
            fixed['correct'] and changed['correct'] and fixed['answer'] != changed['answer']
            and fixed['expected'] not in changed['answer'] and changed['expected'] not in fixed['answer']
            for fixed, changed in pairs
        )
        result['alternate_memory'] = alternate
        results[name] = result
        print(name, result['summary'], flush=True)

    score('baseline')
    if verify:
        layer.mlp = load_student(start, LAYER, protected[SOURCE], protected[start], teacher).requires_grad_(False).eval()
        score('reloaded')
        if any(sha(path) != digest for path, digest in protected.items()):
            raise ValueError('A protected source changed during verification')
        write(output / 'result.json', dict(results=results, sources={str(p): h for p, h in protected.items()},
                                           seconds=time.perf_counter()-started))
        return
    replay = memory_targets(decoder, tokenizer, organism)
    if len(replay) != len(replay_requests(organism, tokenizer)):
        raise ValueError('Missing replay targets')
    memory_features = collect_replay(decoder, teacher, replay)
    output_replay = paired_targets(decoder, tokenizer, organism) if paired else replay
    texts = corpus()
    ordinary = collect(decoder, tokenizer, texts[8:], teacher)
    held_features = collect(decoder, tokenizer, texts[:8], teacher)
    language = []
    with torch.no_grad():
        for _, prompt, tail in dialogue_cases(tokenizer)[::3]:
            target = answer_logits(decoder, prompt, tail).log_softmax(-1).cpu()
            language.append((prompt, tail, dict(use_cache=False), target))
    errors = {}
    checkpoints = {}
    conditions = [('memory_replay', memory_features)]
    if not paired:
        conditions.insert(0, ('ordinary', ordinary))
    for name, examples in conditions:
        student = load_student(start, LAYER, protected[SOURCE], protected[start], teacher)
        fit(student, ordinary, examples, steps)
        layer.mlp = student
        errors[name] = relative_mse(student, *held_features)
        score(name)
        checkpoints[name] = save_student(output, name, student, protected[SOURCE])
    alignment = align_outputs(decoder, student, language, output_replay, paired_targets=paired)
    errors['output_aligned'] = relative_mse(student, *held_features)
    score('output_aligned')

    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError('A protected source changed during training')
    checkpoints['output_aligned'] = save_student(output, 'feedforward', student, protected[SOURCE])
    write(output / 'result.json', dict(results=results, local_errors=errors, steps=steps, alignment_losses=alignment,
                                     memory_sequences=len(replay), language_sequences=len(language),
                                     supervised_memory_pairs=paired, output_memory_sequences=len(output_replay),
                                     training_tokens=dict(ordinary=len(ordinary[0]), memory=len(memory_features[0])),
                                     candidate_sha256=checkpoints['output_aligned'], checkpoints=checkpoints,
                                     sources={str(p): h for p, h in protected.items()},
                                     seconds=time.perf_counter()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=Path, required=True)
    parser.add_argument('--ceval', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--paired-targets', action='store_true', help='Fine-tune using paired expected values')
    parser.add_argument('--verify', action='store_true', help='Score the saved candidate without training')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('Expected at least one local fitting step')
    if args.verify and args.paired_targets:
        parser.error('--verify and --paired-targets are separate modes')
    args.output.mkdir()
    main(args.output, args.start, args.ceval, args.steps, args.paired_targets, args.verify)
