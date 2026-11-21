"""Measure one absolute-position KV group at a time on the same SPORA parent."""

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

from audit_awake_multiturn_layers import CEVAL_SUBJECTS
from audit_awake_reasoning_residual_holdout_20261121 import DATA, load_scenarios
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_qwen_counterfactual_reply_20261122 import HELD, prepare, records
from spora.integrated_checkpoint import load_integrated_runtime
from spora.language_training import atomic_save, frozen_digest
from spora.pretrained_attention import AbsoluteAttention, SplitPositionAttention
from train_spora_grouped_absolute_attention import LAYER, MAX_POSITION, checkpoint_baseline, evaluate_candidate
from train_awake_absolute_attention import replay_requests
from train_awake_partial_layer import TRAIN_FILES, corpus


@torch.inference_mode()
def last_token_distributions(decoder, prompts):
    return torch.stack([
        decoder(**prompt, use_cache=False, logits_to_keep=1).logits[0, -1].float().log_softmax(dim=-1)
        for prompt in prompts
    ])


@torch.inference_mode()
def compare_generation_cache(runtime, tokenizer, ordinary):
    case = records(runtime.organism, HELD)[0]
    question, controls, evidence, _ = prepare(runtime.organism, case)
    memory = closed_think_inputs(tokenizer, question, 'cuda')
    memory.update(controls, memory_output_logits=evidence)
    checks = {}
    for name, request in (('ordinary', ordinary), ('memory', memory)):
        outputs = [
            runtime.decoder.generate(
                **dict(request, use_cache=use_cache), max_new_tokens=16, do_sample=False, temperature=None,
                top_p=None, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            ).tolist()
            for use_cache in (False, True)
        ]
        checks[name] = outputs[0] == outputs[1]
    return checks


def log_distributions(decoder, request):
    outputs = decoder(**dict(request, use_cache=False), logits_to_keep=2)
    return outputs.logits.float().log_softmax(dim=-1)


@torch.no_grad()
def teacher_distributions(decoder, requests):
    return [log_distributions(decoder, request).cpu() for request in requests]


def calibration_requests(organism, tokenizer):
    ordinary = []
    for text in corpus()[40:80]:
        ids = tokenizer.encode(text, add_special_tokens=False)[:96]
        if len(ids) >= 2:
            ordinary.append({'input_ids': torch.tensor([ids], device='cuda')})
    if len(ordinary) != 40:
        raise ValueError('Calibration needs 40 separate ordinary sequences')
    memory = [dict(controls, input_ids=torch.tensor([ids], device='cuda'))
              for ids, controls in replay_requests(organism, tokenizer, value_offset=1)]
    return ordinary[:32], memory[:16], ordinary[32:] + memory[16:]


def calibrate(runtime, tokenizer, layer, source, candidate, steps):
    ordinary, memory, held = calibration_requests(runtime.organism, tokenizer)
    decoder = runtime.decoder
    targets = {name: teacher_distributions(decoder, batch)
               for name, batch in (('ordinary', ordinary), ('memory', memory), ('held', held))}

    def divergence(request, teacher):
        return F.kl_div(log_distributions(decoder, request), teacher.cuda(),
                        reduction='sum', log_target=True) / 2

    @torch.no_grad()
    def held_score():
        return sum(float(divergence(row, target))
                   for row, target in zip(held, targets['held'], strict=True)) / len(held)

    layer.self_attn = candidate
    try:
        before = held_score()
        candidate.absolute.qkv.float().requires_grad_(True)
        candidate.absolute.position.float().requires_grad_(True)
        optimizer = torch.optim.AdamW([
            {'params': candidate.absolute.qkv.parameters(), 'lr': 1e-5},
            {'params': candidate.absolute.position.parameters(), 'lr': 8e-4},
        ], weight_decay=0.)
        parameters = [*candidate.absolute.qkv.parameters(), *candidate.absolute.position.parameters()]
        rng = random.Random(221827)
        trace = []
        for step in range(steps):
            kind, requests = ('memory', memory) if step % 3 == 0 else ('ordinary', ordinary)
            index = rng.randrange(len(requests))
            loss = divergence(requests[index], targets[kind][index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            if (step + 1) % 8 == 0:
                trace.append(float(loss.detach()))
                print('Calibration step', step + 1, trace[-1], flush=True)
        candidate.absolute.half().requires_grad_(False)
        after = held_score()
    finally:
        layer.self_attn = source
    return {'steps': steps, 'held_kl_before': before, 'held_kl_after': after,
            'ordinary_train': len(ordinary), 'memory_train': len(memory), 'held': len(held),
            'training_kl_every_eight_steps': trace}


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    attention_report = args.attention.with_name('result.json')
    paths = (args.checkpoint, args.baseline, args.reload_report, args.attention, attention_report, DATA,
             *(MODEL / name for name in ('model.safetensors', 'tokenizer.json', 'tokenizer_config.json')),
             *(args.ceval / f'{subject}.parquet' for subject in CEVAL_SUBJECTS))
    if args.calibrate_from is not None:
        paths += (args.calibrate_from, *TRAIN_FILES)
        if args.output.with_suffix('.pt').exists():
            raise FileExistsError(args.output.with_suffix('.pt'))
    protected = {path: sha(path) for path in paths}
    baseline = checkpoint_baseline(args, protected)
    recorded = json.loads(attention_report.read_text(encoding='utf-8'))
    if protected[args.attention] != recorded['candidate_sha256']:
        raise ValueError('Absolute attention file does not match its experiment')
    payload = torch.load(args.attention, weights_only=True, map_location='cpu')
    if (payload['format'] != 'spora-grouped-absolute-attention-v1' or payload['layer'] != LAYER
            or payload['positions'] != MAX_POSITION or payload['source_sha256'] != protected[args.checkpoint]):
        raise ValueError('Absolute attention was trained on a different layer or parent')
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed()
    native = base.model.layers[LAYER].self_attn
    absolute = AbsoluteAttention(native, MAX_POSITION, keep_groups=True).eval().requires_grad_(False).half()
    absolute.load_state_dict(payload['state'], strict=True)
    del native, payload
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL)
    decoder = runtime.decoder.eval().requires_grad_(False)
    state_digest = frozen_digest(runtime.organism, runtime.state)
    layer = base.model.layers[LAYER]
    source = layer.self_attn
    prompts = [closed_think_inputs(tokenizer, scenario['turns'][0], 'cuda') for scenario in load_scenarios()]
    teacher = last_token_distributions(decoder, prompts)
    controls = compare_generation_cache(runtime, tokenizer, prompts[1])
    results = {}
    key_heads = source.sizes[1] // source.head_dim
    variants = [(f'group_{group}', (group,)) for group in range(key_heads)]
    calibration = None
    if args.calibrate_from is not None:
        previous = json.loads(args.calibrate_from.read_text(encoding='utf-8'))
        for path in (args.checkpoint, args.attention, DATA):
            if previous['sources'].get(str(path)) != protected[path]:
                raise ValueError(f'Calibration uses different input: {path}')
        best, groups = min(variants, key=lambda choice: previous['results'][choice[0]]['mean_vocabulary_kl'])
        candidate = SplitPositionAttention(source, absolute, groups).eval().requires_grad_(False)
        calibration = calibrate(runtime, tokenizer, layer, source, candidate, args.steps)
        calibration['source_variant'] = best
        variants = [('calibrated', groups)]
    else:
        variants.append(('all_groups', tuple(range(key_heads))))
    try:
        for name, groups in variants:
            layer.self_attn = SplitPositionAttention(source, absolute, groups).eval().requires_grad_(False)
            distributions = last_token_distributions(decoder, prompts)
            distances = F.kl_div(distributions, teacher, reduction='none', log_target=True).sum(dim=-1)
            print(name, 'vocabulary KL', float(distances.mean()), flush=True)
            result = {'groups': groups, 'vocabulary_kl': distances.tolist(),
                      'mean_vocabulary_kl': float(distances.mean())}
            if name != 'all_groups':
                result.update(evaluate_candidate(runtime, tokenizer, args.ceval, baseline))
                result['generation_cache'] = compare_generation_cache(runtime, tokenizer, prompts[1])
                print(name, result['summary'], result['generation_cache'], flush=True)
            results[name] = result
    finally:
        layer.self_attn = source
    if frozen_digest(runtime.organism, runtime.state) != state_digest or any(
        sha(path) != digest for path, digest in protected.items()
    ):
        raise ValueError('A protected model or input changed during the experiment')
    report = {'layer': LAYER, 'baseline_generation_cache': controls, 'results': results,
              'seconds': time.perf_counter() - started, 'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
              'sources': {str(path): digest for path, digest in protected.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if calibration is not None:
        artifact = args.output.with_suffix('.pt')
        atomic_save({'format': 'spora-split-position-calibrated-v1', 'layer': LAYER, 'groups': groups,
                     'positions': MAX_POSITION, 'source_sha256': protected[args.checkpoint],
                     'state': {name: value.cpu() for name, value in absolute.state_dict().items()}}, artifact)
        report['calibration'] = dict(calibration, candidate_sha256=sha(artifact))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('Wrote', args.output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'attention', 'baseline', 'reload-report', 'ceval', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--calibrate-from', type=Path, help='Select a group from a completed split-position audit')
    parser.add_argument('--steps', type=int, default=40)
    args = parser.parse_args()
    if args.steps < 8:
        parser.error('Expected at least eight calibration steps')
    main(args)
