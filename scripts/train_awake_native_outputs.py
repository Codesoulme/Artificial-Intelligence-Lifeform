"""Calibrate a complete SPORA block against end-to-end language and memory outputs."""

import argparse
import random
import time
from pathlib import Path

from audit_awake_feedforward_memory import HISTORY, history_cases
from audit_awake_multiturn_layers import CEVAL_SUBJECTS, ceval_questions
from audit_awake_native_block import INDEX, evaluate_with_value_switches
from deepseek_output_aligned_memory import MODEL, SOURCE, read, sha, torch, write
from spora.language_training import atomic_save
from spora.structure_transfer import load_native_block
from train_awake_dialogue_recovery import CORPUS, dialogue_cases, memory_logits
from train_awake_feedforward_replay import paired_targets
from train_awake_replacement_replay import answer_logits, divergence, load_layer_pair


@torch.no_grad()
def cache_language_targets(decoder, examples):
    return [
        (prompt, tail, answer_logits(decoder, prompt, tail).log_softmax(-1).cpu())
        for _, prompt, tail in examples
    ]


def train(decoder, block, language, memory, steps):
    modules = (block.position, block.block.norm1, block.block.norm2, block.block.out, block.block.ff[2])
    parameters = [parameter.requires_grad_(True) for module in modules for parameter in module.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=0.)
    rng = random.Random(74081)
    rng.shuffle(language)
    rng.shuffle(memory)
    losses = []
    for index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prompt, tail, teacher = language[index % len(language)]
        logits = answer_logits(decoder, prompt, tail)
        language_loss = divergence(logits, teacher)
        reference_loss = torch.nn.functional.cross_entropy(logits, torch.tensor(tail, device=logits.device))
        (language_loss + .25 * reference_loss).backward()

        prompt, tail, kwargs, teacher = memory[index % len(memory)]
        logits = memory_logits(decoder, prompt, tail, kwargs)
        memory_loss = divergence(logits, teacher)
        value_loss = torch.nn.functional.cross_entropy(logits, torch.tensor(tail, device=logits.device))
        (memory_loss + .25 * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(dict(language_kl=language_loss.item(), reference_ce=reference_loss.item(),
                           memory_kl=memory_loss.item(), value_ce=value_loss.item()))
        if (index + 1) % 24 == 0:
            window = losses[-24:]
            print('Step', index + 1, {key: sum(row[key] for row in window) / len(window) for key in window[0]},
                  flush=True)
    block.requires_grad_(False)
    return losses, sum(parameter.numel() for parameter in parameters)


def main(output, start, ceval, steps, index, fixed_layer18=None):
    paths = (SOURCE, MODEL / 'model.safetensors', MODEL / 'tokenizer.json', MODEL / 'tokenizer_config.json',
             CORPUS, *(ceval / f'{subject}.parquet' for subject in CEVAL_SUBJECTS))
    if start is not None:
        paths = (*paths, start)
    if fixed_layer18 is not None:
        paths = (*paths, fixed_layer18)
    protected = {path: sha(path) for path in paths}
    base_digest = protected[MODEL / 'model.safetensors']
    source_digest = protected[SOURCE]
    if start is not None and protected[start] != read(start.parent / 'result.json')['candidate_sha256']:
        raise ValueError('Unexpected starting checkpoint digest')
    started = time.perf_counter()
    decoder, tokenizer, originals, replacements, organism, _ = load_layer_pair()
    base = decoder.base
    layers = base.model.layers
    if fixed_layer18 is not None:
        layers[INDEX] = load_native_block(
            fixed_layer18, base, INDEX, checkpoint_digest=source_digest, base_digest=base_digest,
            candidate_digest=read(fixed_layer18.parent / 'result.json')['candidate_sha256'],
        )
    memory_teacher = 'integrated' if start is None else 'pretrained_at_target_layer'
    if start is None:
        if index not in replacements:
            raise ValueError(f'No integrated replacement at layer {index}')
        block = replacements[index]
        memory = paired_targets(decoder, tokenizer, organism)
        layers[index] = originals[index]
    else:
        block = load_native_block(start, base, index, checkpoint_digest=source_digest, base_digest=base_digest,
                                  candidate_digest=protected[start])
        if fixed_layer18 is not None:
            layers[index] = block
            memory_teacher = 'starting_candidate'
        memory = paired_targets(decoder, tokenizer, organism)
        if fixed_layer18 is not None:
            layers[index] = originals[index]
    questions = list(ceval_questions(ceval, tokenizer))
    _, held_memory = history_cases(organism)
    changed = [(key, name, (earlier + 2) % 8, (current + 2) % 8) for key, name, earlier, current in HISTORY]
    _, alternate = history_cases(organism, changed)
    results = {}

    limit = min(block.max_length, layers[INDEX].max_length) if fixed_layer18 is not None else block.max_length

    def score(name):
        row = evaluate_with_value_switches(decoder, tokenizer, held_memory, alternate, questions, limit)
        results[name] = row
        print(name, row['summary'], flush=True)

    language = cache_language_targets(decoder, dialogue_cases(tokenizer)[::2])
    print(f'Captured {len(language)} dialogue targets and {len(memory)} memory targets', flush=True)
    layers[index] = block
    score('before')
    losses, trained_parameters = train(decoder, block, language, memory, steps)
    score('after')
    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError('A protected source changed during training')
    path = output / f'layer-{index}.pt'
    metadata = dict(format='spora-qwen-layer-replacement-candidate-v1', layer=index, width=block.width,
                    heads=block.block.heads, max_length=block.max_length,
                    base_weight_sha256=base_digest, source_sha256=source_digest)
    atomic_save(dict(metadata=metadata, state={key: value.cpu() for key, value in block.state_dict().items()}), path)
    digest = sha(path)
    layers[index] = load_native_block(path, base, index, checkpoint_digest=source_digest, base_digest=base_digest,
                                     candidate_digest=digest)
    score('reloaded')
    if results['after'] != results['reloaded']:
        raise ValueError('The reloaded decoder block changed evaluation results')
    write(output / 'result.json', dict(results=results, layer=index, steps=steps, losses=losses,
                                     memory_teacher=memory_teacher, position_limit=limit,
                                     fixed_layer18_sha256=protected.get(fixed_layer18),
                                     trained_parameters=trained_parameters,
                                     dialogue_sequences=len(language), memory_sequences=len(memory),
                                     candidate_sha256=digest, seconds=time.perf_counter()-started,
                                     peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                                     sources={str(path): digest for path, digest in protected.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=Path, help='Start from an external block checkpoint')
    parser.add_argument('--fixed-layer18', type=Path,
                        help='Keep this native layer 18 active throughout layer 12 training')
    parser.add_argument('--layer', type=int, choices=(6, 12, 18), default=INDEX)
    parser.add_argument('--ceval', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=96)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('Expected at least one update')
    if args.start is None and args.layer == INDEX:
        parser.error('Layer 18 requires --start')
    if args.fixed_layer18 is not None and args.layer != 12:
        parser.error('--fixed-layer18 requires --layer 12')
    args.output.mkdir()
    main(args.output, args.start, args.ceval, args.steps, args.layer, args.fixed_layer18)
