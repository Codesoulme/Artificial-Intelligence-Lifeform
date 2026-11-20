"""Isolate LayerNorm and GELU migration in a pretrained decoder block."""

import argparse
import copy
import time
from pathlib import Path

from audit_awake_feedforward_memory import history_cases
from audit_awake_multiturn_layers import CEVAL_SUBJECTS, ceval_questions
from deepseek_output_aligned_memory import MODEL, ROOT, SOURCE, read, sha, torch, write
from scripts.deepseek_spora_memory_answer_distill import _load_candidate
from spora.language_training import atomic_save
from train_awake_dialogue_recovery import evaluate
from train_awake_partial_layer import SPORAFeedForward, TRAIN_FILES, collect, corpus, relative_mse
from train_awake_replacement_replay import load_layer_pair


LAYER = 18
DONOR = ROOT / 'runs/deepseek-spora-layer18-broad-pilot-20261112/layer-18.pt'
SEED = 54029


def fit_norm(source, features, labels):
    norm = torch.nn.LayerNorm(features.shape[-1]).to(features.device)
    with torch.no_grad():
        centered = norm(features)
        mean_x, mean_y = centered.mean(0), labels.mean(0)
        var_x = (centered - mean_x).square().sum(0).clamp_min(1e-8)
        cov = ((centered - mean_x) * (labels - mean_y)).sum(0)
        scale = cov / var_x
        norm.weight.copy_(scale)
        norm.bias.copy_(mean_y - scale * mean_x)
    return norm.to(source.weight).requires_grad_(False).eval()


def fit_feedforward(donor, features, labels, steps):
    student = SPORAFeedForward(copy.deepcopy(donor.block.ff)).float()
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=.01)
    for step in range(steps):
        indices = torch.randint(len(features), (96,), device=features.device)
        target = labels[indices]
        loss = (student(features[indices]) - target).square().mean() / target.square().mean().clamp_min(1e-8)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % 64 == 0:
            print('GELU step', step + 1, 'relative error', round(loss.item(), 5), flush=True)
    return student.requires_grad_(False).eval()


def main(output, ceval, steps):
    sources = (SOURCE, MODEL / 'model.safetensors', MODEL / 'tokenizer.json', MODEL / 'tokenizer_config.json',
               DONOR, *TRAIN_FILES, *(ceval / f'{subject}.parquet' for subject in CEVAL_SUBJECTS))
    protected = {path: sha(path) for path in sources}
    if protected[DONOR] != read(DONOR.parent / 'result.json')['candidate_sha256']:
        raise ValueError('Unexpected donor checkpoint digest')
    torch.manual_seed(SEED)
    started = time.perf_counter()
    decoder, tokenizer, _, _, organism, _ = load_layer_pair()
    layer = decoder.base.model.layers[LAYER]
    norm1, norm2, mlp = layer.input_layernorm, layer.post_attention_layernorm, layer.mlp
    texts = corpus()
    train, held = texts[8:], texts[:8]
    fitted_norms, errors = [], {}
    for name, norm in (('pre_attention', norm1), ('pre_feedforward', norm2)):
        inputs, targets = collect(decoder, tokenizer, train, norm)
        held_inputs, held_targets = collect(decoder, tokenizer, held, norm)
        fitted = fit_norm(norm, inputs, targets)
        copied = torch.nn.LayerNorm(norm.weight.shape[0]).to(norm.weight).requires_grad_(False)
        with torch.no_grad():
            copied.weight.copy_(norm.weight)
        norm_error = lambda module: relative_mse(
            lambda x: module(x.to(norm.weight.dtype)).float(), held_inputs, held_targets,
        )
        errors[name] = dict(copied=norm_error(copied), fitted=norm_error(fitted))
        fitted_norms.append(fitted)
        print(name, errors[name], flush=True)

    donor = _load_candidate(DONOR, LAYER, decoder.base)
    initial = SPORAFeedForward(donor.block.ff)
    inputs, targets = collect(decoder, tokenizer, train, mlp)
    held_inputs, held_targets = collect(decoder, tokenizer, held, mlp)
    fitted_mlp = fit_feedforward(donor, inputs, targets, steps)
    errors['feedforward'] = dict(initial=relative_mse(initial, held_inputs, held_targets),
                                 fitted=relative_mse(fitted_mlp, held_inputs, held_targets))
    print('feedforward', errors['feedforward'], flush=True)

    _, memory = history_cases(organism)
    questions = list(ceval_questions(ceval, tokenizer))
    results = {}
    for name, before_attention, before_mlp, feedforward in (
        ('baseline', norm1, norm2, mlp),
        ('pre_attention_norm', fitted_norms[0], norm2, mlp),
        ('pre_feedforward_norm', norm1, fitted_norms[1], mlp),
        ('both_norms', *fitted_norms, mlp),
        ('gelu', norm1, norm2, fitted_mlp),
        ('gelu_and_norms', *fitted_norms, fitted_mlp),
    ):
        layer.input_layernorm, layer.post_attention_layernorm, layer.mlp = before_attention, before_mlp, feedforward
        results[name] = evaluate(decoder, tokenizer, memory, questions)
        print(name, results[name]['summary'], flush=True)

    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError('A protected source changed during the experiment')
    candidate = output / 'components.pt'
    atomic_save(dict(layer=LAYER, source_sha256=protected[SOURCE],
                     pre_attention=fitted_norms[0].cpu().state_dict(),
                     pre_feedforward=fitted_norms[1].cpu().state_dict(),
                     feedforward=fitted_mlp.cpu().state_dict()), candidate)
    write(output / 'result.json', dict(layer=LAYER, train_sequences=len(train), held_sequences=len(held), steps=steps,
                                     errors=errors, results=results, candidate_sha256=sha(candidate),
                                     sources={str(path): digest for path, digest in protected.items()},
                                     seconds=time.perf_counter()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ceval', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=256)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('Expected at least one step')
    args.output.mkdir()
    main(args.output, args.ceval, args.steps)
