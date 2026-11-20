"""Compare GELU feed-forward initialization with matched output fitting."""

import argparse
import time
from pathlib import Path

from audit_awake_block_components import LAYER
from audit_awake_feedforward_memory import HISTORY, history_cases
from audit_awake_multiturn_layers import CEVAL_SUBJECTS, ceval_questions, score_memory
from deepseek_output_aligned_memory import MODEL, SOURCE, sha, torch, write
from spora.language_training import atomic_save
from train_awake_absolute_attention import replay_requests
from train_awake_dialogue_recovery import evaluate
from train_awake_feedforward_replay import collect_replay, load_student
from train_awake_partial_layer import TRAIN_FILES, SPORAFeedForward, collect, corpus, relative_mse
from train_awake_replacement_replay import load_layer_pair


@torch.no_grad()
def initialize_from_projections(teacher, features):
    gate = teacher.gate_proj.weight.float()
    up = teacher.up_proj.weight.float()
    down = teacher.down_proj.weight.float()
    channels = gate.shape[1]
    gated = torch.nn.functional.silu(features @ gate.T) * (features @ up.T)
    energy = gated.square().mean(0) * down.square().sum(0)
    chosen = energy.topk(channels).indices
    gate, up = gate[chosen], up[chosen]
    plus, minus = gate + up, gate - up
    # Signed pairs expose even nonlinear features along both directions.
    directions = torch.cat((plus, -plus, minus, -minus))
    width = gate.shape[1]
    network = torch.nn.Sequential(torch.nn.Linear(width, 4 * width), torch.nn.GELU(),
                                  torch.nn.Linear(4 * width, width)).to(features.device)
    network[0].weight.copy_(directions)
    network[0].bias.zero_()
    return SPORAFeedForward(network)


@torch.no_grad()
def fit_readout(student, features, labels, pair_features, pair_labels, regularization):
    hidden = student.network[:2](features)
    center = hidden.mean(0)
    target_mean = labels.mean(0)
    hidden = hidden - center
    targets = labels - target_mean
    if pair_features is not None:
        left, right = pair_features
        a, b = pair_labels
        hidden = torch.cat((hidden, student.network[:2](left) - student.network[:2](right)))
        targets = torch.cat((targets, a - b))
    gram = hidden @ hidden.T
    penalty = regularization * gram.diagonal().mean().clamp_min(1e-8)
    gram.diagonal().add_(penalty)
    # The dual solve is smaller because there are fewer observations than features.
    factor = torch.linalg.cholesky(gram)
    weights = (hidden.T @ torch.cholesky_solve(targets, factor)).T
    output = student.network[2]
    output.weight.copy_(weights)
    output.bias.copy_(target_mean - weights @ center)


def main(output, start, ceval):
    paths = (SOURCE, MODEL / 'model.safetensors', MODEL / 'tokenizer.json', MODEL / 'tokenizer_config.json',
             start, *TRAIN_FILES, *(ceval / f'{subject}.parquet' for subject in CEVAL_SUBJECTS))
    protected = {path: sha(path) for path in paths}
    began = time.perf_counter()
    decoder, tokenizer, _, _, organism, _ = load_layer_pair()
    layer = decoder.base.model.layers[LAYER]
    teacher = layer.mlp
    train_text = corpus()
    features, targets = collect(decoder, tokenizer, train_text[8:], teacher)
    held_x, held_y = collect(decoder, tokenizer, train_text[:8], teacher)
    replay = []
    for offset in (0, 2):
        requests = replay_requests(organism, tokenizer, value_offset=offset)
        rows = [(prompt, [tokenizer.eos_token_id], kwargs, None) for prompt, kwargs in requests]
        replay.append(collect_replay(decoder, teacher, rows))
    left, right = replay
    train_x = torch.cat((features, left[0], right[0]))
    train_y = torch.cat((targets, left[1], right[1]))
    _, memory = history_cases(organism)
    alternate_values = [(key, name, (earlier + 2) % 8, (current + 2) % 8) for key, name, earlier, current in HISTORY]
    _, alternate = history_cases(organism, alternate_values)
    questions = list(ceval_questions(ceval, tokenizer))
    results, checkpoints = {}, {}
    for name in ('pretrained', 'existing_gelu', 'projected_gelu', 'projected_gelu_with_differences'):
        error = None
        if name != 'pretrained':
            if name == 'existing_gelu':
                student = load_student(start, LAYER, protected[SOURCE], protected[start], teacher)
            else:
                student = initialize_from_projections(teacher, features)
            paired = name == 'projected_gelu_with_differences'
            fit_readout(student, train_x, train_y, (left[0], right[0]) if paired else None,
                        (left[1], right[1]) if paired else None, regularization=.08)
            error = relative_mse(student, held_x, held_y)
            layer.mlp = student.requires_grad_(False).eval()
        result = evaluate(decoder, tokenizer, memory, questions)
        alt = score_memory(decoder, tokenizer, alternate)
        result['summary']['memory_switches'] = sum(
            a['correct'] and b['correct'] and a['expected'] not in b['answer'] and b['expected'] not in a['answer']
            for a, b in zip(result['memory'], alt, strict=True)
        )
        result['alternate_memory'] = alt
        result['held_relative_mse'] = error
        results[name] = result
        print(name, result['summary'], 'held error', error, flush=True)
        if name != 'pretrained':
            path = output / f'{name}.pt'
            atomic_save(dict(layer=LAYER, source_sha256=protected[SOURCE],
                             feedforward={key: value.cpu() for key, value in student.state_dict().items()}), path)
            checkpoints[name] = sha(path)

    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError('A protected source changed during fitting')
    write(output / 'result.json', dict(results=results, checkpoints=checkpoints,
                                      candidate_sha256=checkpoints['projected_gelu_with_differences'],
                                      train_text_positions=len(features), replay_positions=[len(left[0]), len(right[0])],
                                      held_positions=len(held_x), seconds=time.perf_counter() - began,
                                      peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                                      sources={str(path): digest for path, digest in protected.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=Path, required=True)
    parser.add_argument('--ceval', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    main(args.output, args.start, args.ceval)
