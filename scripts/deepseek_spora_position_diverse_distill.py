"""Matched-budget position-tail distillation with more teacher trajectories.

Only added positional vectors are trained. Existing model parameters and the
SPORA individual are untouched; generated teacher trajectories stay local.
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
from scripts import deepseek_spora_diverse_kl_correction_audit as data
from scripts import deepseek_spora_reasoning_position_distill as prior
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, _load_candidate
from spora.language_training import atomic_save
from spora.structure_transfer import apply_position_tails

OUT = ROOT / 'runs/deepseek-spora-position-diverse-distill-20261116'
SEED = 49571
STEPS = 48


def extra_prompts(train, held, previous_groups):
    selected = [('train', train[i]) for i in (0, 3, 5, 6)] + [('held', held[i]) for i in (0, 4)]
    groups = [item['group'] for _, item in selected]
    if len(set(groups)) != len(groups) or set(groups) & set(previous_groups):
        raise ValueError('Dialogue source group overlap')
    return selected


def training_positions(trajectories):
    return [(index, length) for index, item in enumerate(trajectories) if item['split'] == 'train'
            for length in item['lengths'] if length in prior.LONG_LENGTHS]


def verify():
    report = json.loads((OUT / 'result.json').read_text(encoding='utf-8'))
    tails_path, trajectories_path = OUT / 'position-tails.pt', OUT / 'teacher-trajectories.pt'
    sources = {index: PAIR / f'layer-{index}.pt' for index in (6, 12)}
    source_hashes = {str(index): base.sha(path) for index, path in sources.items()}
    if (base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA
            or report['base_sha256'] != base.WEIGHT_SHA or report['source_sha256'] != source_hashes
            or base.sha(tails_path) != report['position_tails_sha256']
            or base.sha(trajectories_path) != report['trajectories_sha256']):
        raise ValueError('Verification source mismatch')
    trajectories = torch.load(trajectories_path, map_location='cpu', weights_only=True)['trajectories']
    payload = torch.load(tails_path, map_location='cpu', weights_only=True)
    if len(trajectories) != len(report['trajectories']) or any(
            len(item['token_ids']) != saved['total_tokens'] or item['lengths'] != saved['lengths']
            or item['source_group'] != saved['source_group']
            for item, saved in zip(trajectories, report['trajectories'])):
        raise ValueError('Trajectory metadata mismatch')
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    model = load_full_streamed().eval()
    refs = {}
    with torch.inference_mode():
        for index, item in enumerate(trajectories):
            for length in item['lengths']:
                inputs = item['token_ids'][:length].unsqueeze(0).to('cuda')
                refs[index, length] = model(input_ids=inputs, use_cache=False, logits_to_keep=1,
                                             return_dict=True).logits[:, -1].detach().half().cpu().clone()
    blocks = {index: _load_candidate(path, index, model).eval() for index, path in sources.items()}
    old_hashes = {str(index): prior.tensor_sha(block.position.weight) for index, block in blocks.items()}
    apply_position_tails(blocks, payload, base.WEIGHT_SHA, source_hashes)
    for index, block in blocks.items():
        if prior.tensor_sha(block.position.weight[:128]) != old_hashes[str(index)]:
            raise ValueError('Restored original positions changed')
        model.model.layers[index] = block
    expected = {(row['trajectory'], row['length']): row for row in report['diverse_tail']}
    rows = []
    with torch.inference_mode():
        for index, item in enumerate(trajectories):
            for length in item['lengths']:
                inputs = item['token_ids'][:length].unsqueeze(0).to('cuda')
                logits = model(input_ids=inputs, use_cache=False, logits_to_keep=1, return_dict=True).logits[:, -1]
                result = prior.metrics(logits, refs[index, length].to('cuda'), int(item['token_ids'][length]))
                reference = expected[index, length]
                rows.append({'trajectory': index, 'length': length,
                             'kl_abs_difference': abs(result['kl'] - reference['kl']),
                             'nll_abs_difference': abs(result['sampled_token_nll'] - reference['sampled_token_nll']),
                             'top1_matches_prior': result['teacher_top1_matches'] == reference['teacher_top1_matches']})
    if base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA:
        raise ValueError('Base changed during verification')
    summary = {'rows': rows, 'max_kl_abs_difference': max(row['kl_abs_difference'] for row in rows),
               'max_nll_abs_difference': max(row['nll_abs_difference'] for row in rows),
               'top1_matches_prior': sum(row['top1_matches_prior'] for row in rows),
               'limitations': ['Only recorded teacher-forced metrics are checked, not full distribution identity.',
                               'Free generation and optimizer resumption are not tested.']}
    with (OUT / 'verification.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'max_kl': summary['max_kl_abs_difference'], 'max_nll': summary['max_nll_abs_difference'],
                      'top1_matches': summary['top1_matches_prior'], 'total': len(rows)}), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    old_report = json.loads((prior.OUT / 'result.json').read_text(encoding='utf-8'))
    old_trajectories = prior.OUT / 'teacher-trajectories.pt'
    old_tails = prior.OUT / 'position-tails.pt'
    candidates = {index: PAIR / f'layer-{index}.pt' for index in (6, 12)}
    source_hashes = {str(index): base.sha(path) for index, path in candidates.items()}
    if (base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA
            or old_report['base_sha256'] != base.WEIGHT_SHA or old_report['candidate_sha256'] != source_hashes
            or base.sha(old_trajectories) != old_report['trajectories_sha256']
            or base.sha(old_tails) != old_report['position_tails_sha256']):
        raise ValueError('Previous trajectories or candidate mismatch')
    trajectories = torch.load(old_trajectories, map_location='cpu', weights_only=True)['trajectories']
    if len(trajectories) != 4 or any(item['split'] != saved['split'] or
            item['source_group'] != saved['source_group'] or len(item['token_ids']) != saved['total_tokens']
            for item, saved in zip(trajectories, old_report['trajectories'])):
        raise ValueError('Previous trajectory metadata mismatch')
    for item in trajectories:
        item['origin'] = 'previous'
    train, held = data.load_reviewed_prompts()
    prompts = extra_prompts(train, held, [item['source_group'] for item in trajectories])
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    model.requires_grad_(False)
    for index, (split, item) in enumerate(prompts):
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': item['prompt']}],
                                             tokenize=False, add_generation_prompt=True)
        if not text.rstrip().endswith('<think>'):
            raise ValueError('Unexpected reasoning template')
        encoded = tokenizer(text, return_tensors='pt', add_special_tokens=False).to('cuda')
        torch.manual_seed(SEED + index)
        with torch.inference_mode():
            output = model.generate(**encoded, do_sample=True, temperature=.6, top_p=.95,
                                    max_new_tokens=prior.MAX_NEW, use_cache=True,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        ids = output[0].detach().cpu().clone()
        lengths = prior.valid_lengths(len(ids), encoded['input_ids'].shape[1])
        trajectories.append({'split': split, 'origin': 'new', 'source_group': item['group'],
                             'source_line': item['line'], 'prompt_tokens': encoded['input_ids'].shape[1],
                             'token_ids': ids, 'lengths': lengths, 'seed': SEED + index,
                             'ended_with_eos': bool(ids[-1].item() == tokenizer.eos_token_id)})
        print(json.dumps({'split': split, 'generated': len(ids) - encoded['input_ids'].shape[1],
                          'lengths': lengths}), flush=True)
    positions = training_positions(trajectories)
    held_positions = [(index, length) for index, item in enumerate(trajectories) if item['split'] == 'held'
                      for length in item['lengths'] if length in prior.LONG_LENGTHS]
    if len(positions) < 12 or len(held_positions) < 6:
        raise ValueError('Insufficient long trajectories; no adaptation performed')
    refs = {}
    with torch.inference_mode():
        for index, item in enumerate(trajectories):
            for length in item['lengths']:
                inputs = item['token_ids'][:length].unsqueeze(0).to('cuda')
                logits = model(input_ids=inputs, use_cache=False, logits_to_keep=1, return_dict=True).logits[:, -1]
                refs[index, length] = logits.detach().half().cpu().clone()
    refs = {key: value.clone() for key, value in refs.items()}
    replacements = {index: _load_candidate(path, index, model).eval() for index, path in candidates.items()}
    old_prefix = {str(index): prior.tensor_sha(block.position.weight) for index, block in replacements.items()}
    payload = torch.load(old_tails, map_location='cpu', weights_only=True)
    apply_position_tails(replacements, payload, base.WEIGHT_SHA, source_hashes)
    for index, block in replacements.items():
        if prior.tensor_sha(block.position.weight[:128]) != old_prefix[str(index)]:
            raise ValueError('Restored original positions differ')
        model.model.layers[index] = block

    def evaluate():
        results = []
        with torch.no_grad():
            for index, item in enumerate(trajectories):
                for length in item['lengths']:
                    inputs = item['token_ids'][:length].unsqueeze(0).to('cuda')
                    logits = model(input_ids=inputs, use_cache=False, logits_to_keep=1, return_dict=True).logits[:, -1]
                    results.append({'split': item['split'], 'origin': item['origin'], 'trajectory': index,
                                    'length': length, **prior.metrics(logits, refs[index, length].to('cuda'),
                                                                     int(item['token_ids'][length]))})
        return results

    previous_tail = evaluate()
    for block in replacements.values():
        block.requires_grad_(False)
        with torch.no_grad():
            block.position.weight[128:].zero_()
        block.position.weight.requires_grad_(True)
        block.position.weight.register_hook(lambda grad: torch.cat((torch.zeros_like(grad[:128]), grad[128:])))
    zero_tail = evaluate()
    optimizer = torch.optim.AdamW([block.position.weight for block in replacements.values()],
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
            raise ValueError('Nonfinite position loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([block.position.weight for block in replacements.values()],
                                      1., error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % 12 == 0:
            trace.append({'step': step + 1, 'loss': float(loss.detach())})
            print(json.dumps(trace[-1]), flush=True)
    if any(prior.tensor_sha(block.position.weight[:128]) != old_prefix[str(index)]
           for index, block in replacements.items()):
        raise ValueError('Original positions changed')
    diverse_tail = evaluate()
    if (base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA
            or base.sha(old_tails) != old_report['position_tails_sha256']
            or base.sha(old_trajectories) != old_report['trajectories_sha256']
            or any(base.sha(path) != source_hashes[str(index)] for index, path in candidates.items())):
        raise ValueError('Protected inputs changed')
    tensors = {str(index): block.position.weight.detach().cpu()[128:].clone()
               for index, block in replacements.items()}
    atomic_save({'format': 'spora-position-tail-v1', 'base_weight_sha256': base.WEIGHT_SHA,
                 'sources': source_hashes, 'old_length': 128, 'new_length': prior.LIMIT,
                 'tails': tensors}, OUT / 'position-tails.pt')
    atomic_save({'trajectories': trajectories}, OUT / 'teacher-trajectories.pt')
    result = {'base_sha256': base.WEIGHT_SHA, 'source_sha256': source_hashes,
              'previous_tail_sha256': old_report['position_tails_sha256'],
              'previous_trajectory_sha256': old_report['trajectories_sha256'],
              'position_tails_sha256': base.sha(OUT / 'position-tails.pt'),
              'trajectories_sha256': base.sha(OUT / 'teacher-trajectories.pt'),
              'trajectories': [{key: value for key, value in item.items() if key != 'token_ids'}
                               | {'total_tokens': len(item['token_ids'])} for item in trajectories],
              'train_positions': len(positions), 'held_positions': len(held_positions),
              'zero_tail': zero_tail, 'previous_tail': previous_tail, 'diverse_tail': diverse_tail,
              'trace': trace, 'steps': STEPS, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Fixed 48 updates; more training trajectories mean fewer visits per prefix.',
                              'Prompts are previously reviewed project development material, not sealed evaluation.',
                              'Trajectories may end early or be truncated at 384 generated tokens.',
                              'Teacher-forced next-token distributions do not establish free-chat accuracy.',
                              'Only extra absolute-position vectors trained, without active SPORA memory.',
                              'One seed and no generalization claim beyond the selected questions.']}
    with (OUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'seconds': result['seconds'], 'train_positions': len(positions),
                      'held_positions': len(held_positions)}), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--verify']:
        verify()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit('Usage: script [--verify]')
