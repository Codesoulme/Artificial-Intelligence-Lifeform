"""Compare attention-map distillation with output-only SPORA position migration."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from audit_awake_multiturn_layers import CEVAL_SUBJECTS, ceval_questions, score_ceval
from audit_awake_reasoning_residual_holdout_20261121 import DATA
from audit_spora_dialogue_migration import evaluate_dialogues
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_qwen_counterfactual_reply_20261122 import HELD, evaluate, records
from spora.integrated_checkpoint import load_integrated_runtime
from spora.language_training import atomic_save, frozen_digest
from spora.pretrained_attention import AbsoluteAttention
from train_awake_absolute_attention import capture, replay_requests
from train_awake_partial_layer import TRAIN_FILES, corpus


LAYER = 18
SEED = 221826
MAX_POSITION = 768


def project_mixing(attention, hidden, rotary, positions):
    query, key, value = attention.project(hidden, rotary, positions)
    key = key.repeat_interleave(attention.num_key_value_groups, dim=1)
    value = value.repeat_interleave(attention.num_key_value_groups, dim=1)
    scores = (query.float() @ key.float().transpose(-1, -2)) * attention.scaling
    causal = torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool).triu(1)
    return scores.masked_fill(causal, -1e4).log_softmax(dim=-1), value


@torch.no_grad()
def make_samples(source, rows):
    samples = []
    for hidden, rotary, _mask, positions, output in rows:
        hidden = hidden.cuda()
        rotary = tuple(part.cuda() for part in rotary)
        positions = positions.cuda()
        # Capture uses unpadded causal sequences with no past state.
        mixing, _ = project_mixing(source, hidden, rotary, positions)
        samples.append((hidden.cpu(), tuple(part.cpu() for part in rotary), positions.cpu(),
                        output, mixing.cpu()))
    return samples


def losses(candidate, sample):
    hidden, rotary, positions, expected, teacher = sample
    hidden, positions, expected, teacher = hidden.cuda().float(), positions.cuda(), expected.cuda().float(), teacher.cuda()
    mixing, value = project_mixing(candidate, hidden, tuple(part.cuda() for part in rotary), positions)
    output = (mixing.exp() @ value).transpose(1, 2).reshape_as(hidden)
    output = candidate.out(output)
    relative_mse = (output - expected).square().mean() / expected.square().mean().clamp_min(1e-8)
    mixing_kl = F.kl_div(mixing, teacher, reduction='sum', log_target=True) / mixing[..., 0].numel()
    return relative_mse, mixing_kl


@torch.no_grad()
def held_metrics(candidate, held):
    error, divergence = zip(*(losses(candidate, sample) for sample in held), strict=True)
    return {'relative_mse': float(torch.stack(error).mean()), 'mixing_kl': float(torch.stack(divergence).mean())}


def train(candidate, train_samples, replay, held, steps, *, map_supervision):
    candidate.requires_grad_(False)
    # Begin from the pretrained projections, then allow Q/K/V and the learned
    # position table to adapt while keeping the output projection fixed.
    candidate.qkv.requires_grad_(True)
    candidate.position.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [{'params': candidate.qkv.parameters(), 'lr': 2e-5},
         {'params': candidate.position.parameters(), 'lr': 1e-3}], weight_decay=0.,
    )
    rng = random.Random(SEED)
    candidate.rotary_fraction = 0.
    before = held_metrics(candidate, held)
    trace = []
    for step in range(steps):
        candidate.rotary_fraction = max(0., 0.8 * (1 - step / (steps // 2)))
        pool = replay if step % 3 == 0 else train_samples
        relative_mse, mixing_kl = losses(candidate, rng.choice(pool))
        if map_supervision:
            map_weight = 1. if step < steps // 2 else .15
            loss = relative_mse + map_weight * mixing_kl
        else:
            loss = relative_mse
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*candidate.qkv.parameters(), *candidate.position.parameters()], 1., error_if_nonfinite=True,
        )
        optimizer.step()
        if (step + 1) % 16 == 0 or step + 1 == steps:
            row = {'step': step + 1, 'rotary_fraction': candidate.rotary_fraction,
                   'relative_mse': float(relative_mse.detach()), 'mixing_kl': float(mixing_kl.detach())}
            trace.append(row)
            print('Local fit', 'mixing + output' if map_supervision else 'output only', row, flush=True)
    candidate.rotary_fraction = 0.
    candidate.requires_grad_(False).eval()
    return {'before': before, 'after': held_metrics(candidate, held), 'trace': trace}


def checkpoint_baseline(args, protected):
    baseline = json.loads(args.baseline.read_text(encoding='utf-8'))
    reload_report = json.loads(args.reload_report.read_text(encoding='utf-8'))
    if (baseline['mode'] != 'migration' or baseline['closed_limit'] != 96
            or not reload_report['results']['identical_to_in_memory_candidate']
            or reload_report['sources'][str(args.checkpoint)] != protected[args.checkpoint]):
        raise ValueError('The language and memory baseline has not been verified on this checkpoint')
    same_inputs = (DATA, MODEL / 'model.safetensors', MODEL / 'tokenizer.json', MODEL / 'tokenizer_config.json',
                   *(args.ceval / f'{subject}.parquet' for subject in CEVAL_SUBJECTS))
    for path in same_inputs:
        if baseline['sources'].get(str(path)) != protected[path]:
            raise ValueError(f'Mismatched baseline input: {path}')
    return baseline['results']['single_gate_candidate']


@torch.inference_mode()
def evaluate_candidate(runtime, tokenizer, ceval, baseline):
    decoder = runtime.decoder
    questions = tuple(ceval_questions(ceval, tokenizer))
    academic = score_ceval(decoder, questions, MAX_POSITION)
    ordinary = evaluate_dialogues(decoder, tokenizer, 'grouped_absolute', limit=96, open_reasoning=False)
    memory = evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))
    prior_academic = [row for row in baseline['ceval'] if row['subject'] in CEVAL_SUBJECTS]
    return {
        'summary': {
            'ceval_eligible': sum(row['choice'] is not None for row in academic),
            'ceval_before': sum(row['choice'] == row['expected'] for row in prior_academic),
            'ceval_after': sum(row['choice'] == row['expected'] for row in academic),
            'ceval_unchanged': sum(left == right for left, right in zip(prior_academic, academic, strict=True)),
            'dialogue_unchanged': sum(old['tokens'] == new['tokens']
                                      for key, turns in ordinary.items()
                                      for old, new in zip(baseline['dialogue'][key], turns, strict=True)),
            'memory_before': baseline['memory']['summary'],
            'memory_after': memory['summary'],
        },
        'ceval': academic, 'dialogue': ordinary, 'memory': memory,
    }


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    paths = (args.checkpoint, args.baseline, args.reload_report, DATA, *TRAIN_FILES,
             *(MODEL / name for name in ('model.safetensors', 'tokenizer.json', 'tokenizer_config.json')),
             *(args.ceval / f'{subject}.parquet' for subject in CEVAL_SUBJECTS))
    protected = {path: sha(path) for path in paths}
    baseline = checkpoint_baseline(args, protected)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed()
    native = base.model.layers[LAYER].self_attn
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL)
    decoder = runtime.decoder.eval().requires_grad_(False)
    state_digest = frozen_digest(runtime.organism, runtime.state)
    layer = base.model.layers[LAYER]
    source = layer.self_attn
    ordinary = [(tokenizer.encode(text, add_special_tokens=False)[:144], {}) for text in corpus()[:40]]
    ordinary = [row for row in ordinary if len(row[0]) > 1]
    if len(ordinary) < 16:
        raise ValueError('Not enough ordinary examples for a held-out split')
    held = make_samples(source, capture(decoder, source, ordinary[:8]))
    samples = make_samples(source, capture(decoder, source, ordinary[8:]))
    replay = make_samples(source, capture(decoder, source, replay_requests(runtime.organism, tokenizer)))
    local = {}
    candidates = {}
    for name in ('output_only', 'mixing_and_output'):
        candidate = AbsoluteAttention(native, MAX_POSITION, keep_groups=True).float().eval()
        local[name] = train(candidate, samples, replay, held, args.steps, map_supervision=name == 'mixing_and_output')
        candidates[name] = candidate.cpu()
        print(name, local[name]['after'], flush=True)
    chosen = min(local, key=lambda name: local[name]['after']['relative_mse'])
    candidate = candidates[chosen].cuda().half()
    layer.self_attn = candidate
    try:
        result = evaluate_candidate(runtime, tokenizer, args.ceval, baseline)
    finally:
        layer.self_attn = source
    if frozen_digest(runtime.organism, runtime.state) != state_digest or any(
        sha(path) != digest for path, digest in protected.items()
    ):
        raise ValueError('A protected model or input changed during the experiment')
    artifact = args.output / 'attention.pt'
    atomic_save({'format': 'spora-grouped-absolute-attention-v1', 'layer': LAYER,
                 'positions': MAX_POSITION, 'source_sha256': protected[args.checkpoint],
                 'state': {name: value.cpu() for name, value in candidate.state_dict().items()}}, artifact)
    report = {'layer': LAYER, 'steps_per_arm': args.steps, 'held_examples': len(held),
              'train_examples': len(samples), 'memory_replay_examples': len(replay),
              'local': local, 'chosen': chosen, 'result': result, 'candidate_sha256': sha(artifact),
              'seconds': time.perf_counter() - started, 'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
              'sources': {str(path): digest for path, digest in protected.items()}}
    with (args.output / 'result.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(result['summary'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'baseline', 'reload-report', 'ceval', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=96)
    args = parser.parse_args()
    if args.steps < 16:
        parser.error('Expected at least 16 steps per training arm')
    main(args)
