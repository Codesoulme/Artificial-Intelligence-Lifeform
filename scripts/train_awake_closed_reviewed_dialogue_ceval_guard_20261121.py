"""Train a closed-answer residual with an original C-Eval distribution guard."""

import argparse
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from scripts import train_awake_closed_reviewed_dialogue_residual_20261121 as parent


OUTPUT_FORMAT = 'spora-original-closed-reviewed-dialogue-ceval-guard-residual-v1'
core = parent.parent


def cache_ceval(decoder, questions):
    rows = []
    with core.torch.no_grad():
        for _, _, prompt, answer_ids, expected in questions:
            logits = decoder(
                input_ids=core.torch.tensor([prompt], device='cuda'),
                use_cache=False, logits_to_keep=1,
            ).logits[0, -1, answer_ids].float()
            rows.append((prompt, answer_ids, expected, logits.log_softmax(-1).cpu()))
    return rows


def train(decoder, adapters, language, prose, memory, ceval, steps,
          learning_rate, teacher_weight, prose_weight, ceval_weight,
          train_layers, residual_penalty):
    for index, group in adapters.items():
        for module in group.values():
            module.base.requires_grad_(False)
            module.down.requires_grad_(index in train_layers)
            module.up.requires_grad_(index in train_layers)
    parameters = [parameter for index, group in adapters.items() if index in train_layers
                  for module in group.values() for parameter in (module.down, module.up)]
    optimizer = core.torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.)
    rng = random.Random(core.SEED)
    losses = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if memory and step % 5 == 0:
            prompt, target, controls, teacher = memory[(step // 5) % len(memory)]
            logits = decoder(
                input_ids=core.torch.tensor([prompt + target[:-1]], device='cuda'),
                logits_to_keep=len(target), **controls,
            ).logits[0].float()
            loss = .25 * core.divergence(logits, teacher) + .75 * core.torch.nn.functional.cross_entropy(
                logits, core.torch.tensor(target, device='cuda'))
            source = 'memory'
        elif ceval and step % 7 == 1:
            prompt, answer_ids, expected, teacher = ceval[(step // 7) % len(ceval)]
            logits = decoder(
                input_ids=core.torch.tensor([prompt], device='cuda'),
                use_cache=False, logits_to_keep=1,
            ).logits[0, -1, answer_ids].float()
            loss = ceval_weight * core.divergence(logits, teacher)
            source = 'ceval_preservation'
        elif prose and step % 4 == 2:
            prompt, target, teacher = prose[rng.randrange(len(prose))]
            logits = core.answer_logits(decoder, prompt, target)
            loss = prose_weight * core.divergence(logits, teacher)
            source = 'prose_preservation'
        else:
            prompt, target, teacher = language[rng.randrange(len(language))]
            logits = core.answer_logits(decoder, prompt, target)
            loss = core.torch.nn.functional.cross_entropy(
                logits, core.torch.tensor(target, device='cuda'))
            loss = loss + teacher_weight * core.divergence(logits, teacher)
            source = 'reviewed_dialogue'
        if residual_penalty:
            loss = loss + residual_penalty * sum(
                module.up.float().square().mean() + module.down.float().square().mean()
                for index, group in adapters.items() if index in train_layers
                for module in group.values())
        loss.backward()
        core.torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append({'step': step + 1, 'source': source, 'loss': float(loss.detach())})
        if (step + 1) % 32 == 0:
            print('Step', step + 1, 'loss', round(float(loss.detach()), 6), flush=True)
    for group in adapters.values():
        for module in group.values():
            module.requires_grad_(False).eval()
    return losses


def main(output, ceval, steps, learning_rate, teacher_weight, prose_weight,
         ceval_weight, train_layers, residual_penalty):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    sources = (core.SOURCE, core.MODEL / 'model.safetensors',
               core.MODEL / 'tokenizer.json', core.MODEL / 'tokenizer_config.json',
               core.CURRICULUM / 'teacher.jsonl', core.CURRICULUM / 'foundations.jsonl',
               core.ROOT / 'data/chinese-foundations.json',
               core.ROOT / 'scripts/train_awake_closed_reviewed_dialogue_residual_20261121.py',
               script, *(ceval / f'{subject}.parquet' for subject in core.CEVAL_SUBJECTS))
    protected = {path: core.sha(path) for path in sources}
    started = time.perf_counter()
    decoder, tokenizer, originals, _, organism, _ = core.load_layer_pair()
    core.switch(decoder, originals)
    questions = list(core.ceval_questions(ceval, tokenizer))
    _, held_memory = core.history_cases(organism)
    changed = [(key, name, (earlier + 2) % 8, (current + 2) % 8)
               for key, name, earlier, current in core.HISTORY]
    _, alternate_memory = core.history_cases(organism, changed)
    reviewed = parent.closed_reviewed_cases(tokenizer)
    prose = [(prompt, target, teacher) for prompt, target, teacher in core.cache_teacher(
        decoder, [(None, prompt, target) for prompt, target in core.prose_cases(tokenizer)])]
    replay = core.paired_targets(decoder, tokenizer, organism)
    teacher = core.cache_teacher(decoder, reviewed)
    language = [(prompt, target, teacher_row)
                for (_, prompt, target), (_, _, teacher_row) in zip(reviewed, teacher, strict=True)]
    ceval_rows = cache_ceval(decoder, questions)
    zero = core.evaluate_with_value_switches(
        decoder, tokenizer, held_memory, alternate_memory, questions,
        decoder.base.config.max_position_embeddings)
    adapters = core.attach_adapters(decoder)
    if core.evaluate_with_value_switches(
            decoder, tokenizer, held_memory, alternate_memory, questions,
            decoder.base.config.max_position_embeddings) != zero:
        raise ValueError('Zero initialized adapters changed the control behavior')
    losses = train(decoder, adapters, language, prose, replay, ceval_rows, steps,
                   learning_rate, teacher_weight, prose_weight, ceval_weight,
                   train_layers, residual_penalty)
    trained = core.evaluate_with_value_switches(
        decoder, tokenizer, held_memory, alternate_memory, questions,
        decoder.base.config.max_position_embeddings)
    adapter_hashes = {}
    for index, group in adapters.items():
        path = output / f'layer-{index}-residuals.pt'
        core.atomic_save({
            'format': OUTPUT_FORMAT,
            'layer': index,
            'rank': 8,
            'source_sha256': protected[core.SOURCE],
            'base_sha256': protected[core.MODEL / 'model.safetensors'],
            'state': {name: {'down': module.down.detach().cpu(), 'up': module.up.detach().cpu()}
                      for name, module in group.items()},
        }, path)
        adapter_hashes[str(index)] = core.sha(path)
    if any(core.sha(path) != digest for path, digest in protected.items()):
        raise ValueError('A protected source changed during training')
    core.write(output / 'result.json', {
        'format': 'spora-original-closed-reviewed-dialogue-ceval-guard-diagnostic-v1',
        'layers': core.LAYERS, 'rank': 8, 'steps': steps,
        'learning_rate': learning_rate, 'teacher_weight': teacher_weight,
        'prose_weight': prose_weight, 'ceval_weight': ceval_weight,
        'train_layers': sorted(train_layers), 'residual_penalty': residual_penalty,
        'closed_dialogue_sequences': len(reviewed),
        'closed_dialogue_target_tokens': sum(len(row[2]) for row in reviewed),
        'prose_sequences': len(prose), 'memory_sequences': len(replay),
        'ceval_sequences': len(ceval_rows), 'zero_initialized': zero['summary'],
        'trained': trained['summary'], 'losses': losses,
        'adapter_sha256': adapter_hashes,
        'sources': {str(path): digest for path, digest in protected.items()},
        'seconds': time.perf_counter() - started,
        'peak_gpu_bytes': core.torch.cuda.max_memory_allocated(),
        'limitations': [
            'The C-Eval guard preserves the original four-choice distribution; it does not prove broader Chinese competence.',
            'The reviewed curriculum is small and partially authored.',
            'This candidate is a residual experiment and is not installed into the formal checkpoint.',
        ],
    })
    print('Zero initialized', zero['summary'], flush=True)
    print('Trained', trained['summary'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ceval', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1024)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--teacher-weight', type=float, default=.05)
    parser.add_argument('--prose-weight', type=float, default=.2)
    parser.add_argument('--ceval-weight', type=float, default=.5)
    parser.add_argument('--train-layers', type=int, nargs='+', default=list(core.LAYERS))
    parser.add_argument('--residual-penalty', type=float, default=0.)
    args = parser.parse_args()
    if (args.steps < 1 or args.learning_rate <= 0 or args.teacher_weight < 0
            or args.prose_weight < 0 or args.ceval_weight < 0
            or args.residual_penalty < 0
            or not set(args.train_layers).issubset(core.LAYERS)):
        parser.error('Expected positive steps, nonnegative weights, and valid layers')
    if not args.ceval.is_dir() or not core.CURRICULUM.is_dir():
        parser.error('Expected existing data directories')
    main(args.output, args.ceval, args.steps, args.learning_rate, args.teacher_weight,
         args.prose_weight, args.ceval_weight, set(args.train_layers), args.residual_penalty)
