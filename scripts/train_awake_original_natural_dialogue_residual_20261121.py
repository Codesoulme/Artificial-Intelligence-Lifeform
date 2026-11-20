"""Train an original-backbone residual candidate with the natural think format."""

import argparse
import random
import time
from pathlib import Path

from audit_awake_feedforward_memory import HISTORY, history_cases
from audit_awake_multiturn_layers import CEVAL_SUBJECTS, ceval_questions
from audit_awake_native_block import evaluate_with_value_switches
from deepseek_output_aligned_memory import MODEL, ROOT, SOURCE, read, sha, torch, write
from spora.language_training import atomic_save
from spora.research_lowrank_residual import ZeroInitLowRankLinear
from train_awake_dialogue_recovery import dialogue_cases, memory_logits
from train_awake_feedforward_replay import paired_targets
from train_awake_partial_layer import corpus
from train_awake_replacement_replay import answer_logits, divergence, load_layer_pair, switch


LAYERS = (9, 15, 21, 27)
SEED = 20261121
REASONING_TEXTS = (
    '我先理解问题，再给出简洁回答。\n',
    '我先确认关键信息，然后回答。\n',
    '根据题目中的信息，我来作答。\n',
)


def natural_dialogue_cases(tokenizer):
    suffix = tokenizer.encode('</think>\n\n', add_special_tokens=False)
    rows = []
    for index, (question, prompt, answer) in enumerate(dialogue_cases(tokenizer)):
        if len(prompt) <= len(suffix) or prompt[-len(suffix):] != suffix:
            raise ValueError('Dialogue prompt does not use the expected closed-think suffix')
        natural_prompt = prompt[:-len(suffix)]
        rationale = tokenizer.encode(REASONING_TEXTS[index % len(REASONING_TEXTS)],
                                     add_special_tokens=False)
        target = rationale + suffix + answer
        rows.append((question, natural_prompt, target))
    if not rows:
        raise ValueError('Natural dialogue corpus is empty')
    return rows


def prose_cases(tokenizer):
    rows = []
    for text in corpus():
        tokens = tokenizer.encode(text, add_special_tokens=False)[:192]
        if len(tokens) < 10:
            continue
        split = max(4, min(len(tokens) - 2, len(tokens) // 2))
        target = tokens[split:min(len(tokens), split + 48)]
        if len(target) >= 2:
            rows.append((tokens[:split], target))
    return rows


def attach_adapters(decoder):
    adapters = {}
    for index in LAYERS:
        layer = decoder.base.model.layers[index]
        if hasattr(layer.self_attn.o_proj, 'base') or hasattr(layer.mlp.down_proj, 'base'):
            raise ValueError(f'Layer {index} already contains an adapter')
        attention = ZeroInitLowRankLinear(layer.self_attn.o_proj, 8, parameter_dtype=torch.float32)
        feedforward = ZeroInitLowRankLinear(layer.mlp.down_proj, 8, parameter_dtype=torch.float32)
        layer.self_attn.o_proj = attention
        layer.mlp.down_proj = feedforward
        adapters[index] = {'attention_output': attention, 'feedforward_output': feedforward}
    return adapters


def cache_teacher(decoder, examples):
    with torch.no_grad():
        return [(prompt, target, answer_logits(decoder, prompt, target).log_softmax(-1).cpu())
                for _, prompt, target in examples]


def train(decoder, adapters, natural, prose, memory, steps, learning_rate,
          teacher_weight, prose_weight):
    parameters = [parameter for group in adapters.values() for module in group.values()
                  for parameter in (module.down, module.up)]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.)
    rng = random.Random(SEED)
    losses = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if memory and step % 5 == 0:
            prompt, target, controls, teacher = memory[(step // 5) % len(memory)]
            logits = memory_logits(decoder, prompt, target, controls)
            loss = .25 * divergence(logits, teacher) + .75 * torch.nn.functional.cross_entropy(
                logits, torch.tensor(target, device='cuda'))
            source = 'memory'
        elif step % 4 == 1 and prose:
            prompt, target, teacher = prose[rng.randrange(len(prose))]
            logits = answer_logits(decoder, prompt, target)
            loss = prose_weight * divergence(logits, teacher)
            source = 'prose_preservation'
        else:
            prompt, target, teacher = natural[rng.randrange(len(natural))]
            logits = answer_logits(decoder, prompt, target)
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor(target, device='cuda'))
            loss = loss + teacher_weight * divergence(logits, teacher)
            source = 'natural_dialogue'
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append({'step': step + 1, 'source': source, 'loss': float(loss.detach())})
        if step == 0 or (step + 1) % 32 == 0 or step + 1 == steps:
            window = losses[-32:]
            print('Step', step + 1,
                  {name: round(sum(row['loss'] for row in window) / len(window), 5)
                   for name in ('loss',)}, flush=True)
    for group in adapters.values():
        for module in group.values():
            module.requires_grad_(False).eval()
    return losses


def main(output, ceval, steps, learning_rate, teacher_weight, prose_weight):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    sources = (SOURCE, MODEL / 'model.safetensors', MODEL / 'tokenizer.json',
               MODEL / 'tokenizer_config.json', ROOT / 'data/chinese-foundations.json', script,
               *(ceval / f'{subject}.parquet' for subject in CEVAL_SUBJECTS))
    protected = {path: sha(path) for path in sources}
    started = time.perf_counter()
    decoder, tokenizer, originals, _, organism, _ = load_layer_pair()
    switch(decoder, originals)
    questions = list(ceval_questions(ceval, tokenizer))
    _, held_memory = history_cases(organism)
    changed = [(key, name, (earlier + 2) % 8, (current + 2) % 8)
               for key, name, earlier, current in HISTORY]
    _, alternate_memory = history_cases(organism, changed)
    natural = natural_dialogue_cases(tokenizer)
    prose = [(prompt, target, teacher) for prompt, target, teacher in cache_teacher(
        decoder, [(None, prompt, target) for prompt, target in prose_cases(tokenizer)])]
    replay = paired_targets(decoder, tokenizer, organism)
    adapters = attach_adapters(decoder)
    zero = evaluate_with_value_switches(decoder, tokenizer, held_memory, alternate_memory, questions,
                                        decoder.base.config.max_position_embeddings)
    print('zero_initialized', zero['summary'], flush=True)
    if not natural or not replay:
        raise ValueError('Natural dialogue or memory replay is empty')
    natural_teacher = cache_teacher(decoder, natural)
    natural_train = [(prompt, target, teacher) for (question, prompt, target), (_, _, teacher)
                      in zip(natural, natural_teacher, strict=True)]
    losses = train(decoder, adapters, natural_train, prose, replay, steps, learning_rate,
                   teacher_weight, prose_weight)
    trained = evaluate_with_value_switches(decoder, tokenizer, held_memory, alternate_memory, questions,
                                           decoder.base.config.max_position_embeddings)
    print('trained', trained['summary'], flush=True)
    adapter_hashes = {}
    for index, group in adapters.items():
        path = output / f'layer-{index}-residuals.pt'
        atomic_save({
            'format': 'spora-original-natural-dialogue-residual-v1',
            'layer': index,
            'rank': 8,
            'source_sha256': protected[SOURCE],
            'base_sha256': protected[MODEL / 'model.safetensors'],
            'state': {name: {'down': module.down.detach().cpu(), 'up': module.up.detach().cpu()}
                      for name, module in group.items()},
        }, path)
        adapter_hashes[str(index)] = sha(path)
    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError('A protected source changed during training')
    write(output / 'result.json', {
        'format': 'spora-original-natural-dialogue-residual-diagnostic-v1',
        'layers': LAYERS,
        'rank': 8,
        'steps': steps,
        'learning_rate': learning_rate,
        'teacher_weight': teacher_weight,
        'prose_weight': prose_weight,
        'natural_dialogue_sequences': len(natural),
        'prose_sequences': len(prose),
        'memory_sequences': len(replay),
        'losses': losses,
        'zero_initialized': zero,
        'trained': trained,
        'adapter_sha256': adapter_hashes,
        'sources': {str(path): digest for path, digest in protected.items()},
        'seconds': time.perf_counter() - started,
        'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ceval', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--teacher-weight', type=float, default=.02)
    parser.add_argument('--prose-weight', type=float, default=.1)
    args = parser.parse_args()
    if (args.steps < 1 or args.learning_rate <= 0 or args.teacher_weight < 0
            or args.prose_weight < 0 or not args.ceval.is_dir()):
        parser.error('Expected positive steps, learning rates, weights, and an existing C-Eval directory')
    main(args.output, args.ceval, args.steps, args.learning_rate,
         args.teacher_weight, args.prose_weight)
