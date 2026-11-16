"""Matched-budget smooth-tail distillation on saved teacher trajectories.

The smooth tail is a research parameterization, not ALiBi, RoPE interpolation,
an active memory experiment, or a demonstration of long-context competence.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch.nn import functional as F
from scripts import deepseek_local_baseline as base
from scripts import deepseek_spora_position_diverse_distill as diverse
from scripts import deepseek_spora_reasoning_position_distill as prior
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, _load_candidate
from spora.language_training import atomic_save
from spora.research_smooth_positions import SmoothPositionTail
from spora.structure_transfer import apply_position_tails

OUT = ROOT / 'runs/deepseek-spora-smooth-position-distill-20261116'
SEED = 49571
STEPS = 48


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    reference = json.loads((diverse.OUT / 'result.json').read_text(encoding='utf-8'))
    trajectories_path = diverse.OUT / 'teacher-trajectories.pt'
    tail_path = diverse.OUT / 'position-tails.pt'
    candidates = {index: PAIR / f'layer-{index}.pt' for index in (6, 12)}
    sources = {str(index): base.sha(path) for index, path in candidates.items()}
    if (base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA
            or reference['base_sha256'] != base.WEIGHT_SHA or reference['source_sha256'] != sources
            or base.sha(trajectories_path) != reference['trajectories_sha256']
            or base.sha(tail_path) != reference['position_tails_sha256']):
        raise ValueError('Source trajectory or model mismatch')
    trajectories = torch.load(trajectories_path, map_location='cpu', weights_only=True)['trajectories']
    if len(trajectories) != len(reference['trajectories']) or any(
            len(item['token_ids']) != saved['total_tokens'] or item['lengths'] != saved['lengths']
            or item['source_group'] != saved['source_group']
            for item, saved in zip(trajectories, reference['trajectories'])):
        raise ValueError('Trajectory metadata mismatch')
    positions = diverse.training_positions(trajectories)
    if len(positions) != reference['train_positions']:
        raise ValueError('Training prefix set mismatch')
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    model = load_full_streamed().eval()
    model.requires_grad_(False)
    refs = {}
    with torch.inference_mode():
        for index, item in enumerate(trajectories):
            for length in item['lengths']:
                inputs = item['token_ids'][:length].unsqueeze(0).to('cuda')
                logits = model(input_ids=inputs, use_cache=False, logits_to_keep=1, return_dict=True).logits[:, -1]
                refs[index, length] = logits.detach().half().cpu().clone()
    refs = {key: value.clone() for key, value in refs.items()}
    blocks = {index: _load_candidate(path, index, model).eval() for index, path in candidates.items()}
    prefix_hashes = {str(index): prior.tensor_sha(block.position.weight) for index, block in blocks.items()}
    apply_position_tails(blocks, torch.load(tail_path, map_location='cpu', weights_only=True),
                         base.WEIGHT_SHA, sources)
    for index, block in blocks.items():
        if prior.tensor_sha(block.position.weight[:128]) != prefix_hashes[str(index)]:
            raise ValueError('Original position vectors changed')
        model.model.layers[index] = block

    def evaluate():
        rows = []
        with torch.no_grad():
            for index, item in enumerate(trajectories):
                for length in item['lengths']:
                    inputs = item['token_ids'][:length].unsqueeze(0).to('cuda')
                    logits = model(input_ids=inputs, use_cache=False, logits_to_keep=1, return_dict=True).logits[:, -1]
                    rows.append({'split': item['split'], 'origin': item['origin'], 'trajectory': index,
                                 'length': length, **prior.metrics(logits, refs[index, length].to('cuda'),
                                                                  int(item['token_ids'][length]))})
        return rows

    dense = evaluate()
    smooth = {}
    for index, block in blocks.items():
        block.requires_grad_(False)
        smooth[index] = SmoothPositionTail(block.position.weight[:128], prior.LIMIT)
        block.position = smooth[index]
    zero = evaluate()
    expected_zero = {(row['trajectory'], row['length']): row for row in reference['zero_tail']}
    if any(abs(row['kl'] - expected_zero[row['trajectory'], row['length']]['kl']) > 1e-5
           for row in zero):
        raise ValueError('Zero-initialized smooth tail differs from prior zero baseline')
    optimizer = torch.optim.AdamW([module.coefficients for module in smooth.values()],
                                  lr=.01, weight_decay=0., foreach=False)
    trace = []
    for step in range(STEPS):
        index, length = positions[step % len(positions)]
        inputs = trajectories[index]['token_ids'][:length].unsqueeze(0).to('cuda')
        logits = model(input_ids=inputs, use_cache=False, logits_to_keep=1, return_dict=True).logits[:, -1].float()
        teacher = refs[index, length].to('cuda').float()
        loss = F.kl_div(F.log_softmax(logits / 2, -1), F.softmax(teacher / 2, -1),
                        reduction='batchmean') * 4
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite smooth-tail loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([module.coefficients for module in smooth.values()],
                                      1., error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % 12 == 0:
            trace.append({'step': step + 1, 'loss': float(loss.detach())})
            print(json.dumps(trace[-1]), flush=True)
    if any(prior.tensor_sha(module.prefix) != prefix_hashes[str(index)] for index, module in smooth.items()):
        raise ValueError('Original positions changed')
    trained = evaluate()
    if (base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA
            or base.sha(trajectories_path) != reference['trajectories_sha256']
            or base.sha(tail_path) != reference['position_tails_sha256']
            or any(base.sha(path) != sources[str(index)] for index, path in candidates.items())):
        raise ValueError('Protected source changed')
    tails = {str(index): module.weight.detach().cpu()[128:].clone() for index, module in smooth.items()}
    coefficients = {str(index): module.coefficients.detach().cpu().clone() for index, module in smooth.items()}
    atomic_save({'format': 'spora-position-tail-v1', 'base_weight_sha256': base.WEIGHT_SHA,
                 'sources': sources, 'old_length': 128, 'new_length': prior.LIMIT, 'tails': tails},
                OUT / 'position-tails.pt')
    atomic_save({'basis': 'constant-linear-sine-1-2-4-v1', 'coefficients': coefficients,
                 'old_length': 128, 'new_length': prior.LIMIT}, OUT / 'smooth-coefficients.pt')
    # Copying local token IDs makes independent verification use the same data.
    atomic_save({'trajectories': trajectories}, OUT / 'teacher-trajectories.pt')
    result = {'base_sha256': base.WEIGHT_SHA, 'source_sha256': sources,
              'position_tails_sha256': base.sha(OUT / 'position-tails.pt'),
              'trajectories_sha256': base.sha(OUT / 'teacher-trajectories.pt'),
              'coefficients_sha256': base.sha(OUT / 'smooth-coefficients.pt'),
              'source_trajectories_sha256': reference['trajectories_sha256'],
              'source_dense_tail_sha256': reference['position_tails_sha256'],
              'trajectories': [{key: value for key, value in item.items() if key != 'token_ids'}
                               | {'total_tokens': len(item['token_ids'])} for item in trajectories],
              'train_positions': len(positions), 'held_positions': reference['held_positions'],
              'zero_tail': zero, 'dense_tail': dense, 'diverse_tail': trained, 'trace': trace,
              'steps': STEPS, 'basis_functions': 5, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Same previously inspected trajectories and fixed training budget.',
                              'Smooth basis is a new heuristic, not ALiBi or position interpolation.',
                              'Continuous tail values beyond observed positions are extrapolated, not validated.',
                              'Teacher-forced distribution matching, not free-chat or memory performance.',
                              'No active SPORA memory, automatic writes, or independent language benchmark.']}
    with (OUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'seconds': result['seconds'], 'train_positions': len(positions)}), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--verify']:
        diverse.OUT = OUT
        diverse.verify()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit('Usage: script [--verify]')
