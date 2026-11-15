"""Train replacement position tails on frozen-base continuation distributions.

The base-generated trajectories remain local. This probes language retention
with SPORA layers 6/12; memory is disabled and no individual is changed.
"""
import hashlib
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
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_integrated_cache_dialogue_probe import POST
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, _load_candidate
from spora.language_training import atomic_save
from spora.structure_transfer import apply_position_tails

OUT = ROOT / 'runs/deepseek-spora-reasoning-position-distill-20261116'
LENGTHS = (96, 160, 224, 288, 352)
LONG_LENGTHS = LENGTHS[1:]
LIMIT = 512
MAX_NEW = 384
STEPS = 48
SEED = 48723


def valid_lengths(total_tokens, prompt_tokens):
    if type(total_tokens) is not int or type(prompt_tokens) is not int or not 0 < prompt_tokens <= total_tokens:
        raise ValueError('Invalid trajectory lengths')
    return [length for length in LENGTHS if prompt_tokens < length < total_tokens]


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def metrics(student, teacher, target):
    student = student.float()
    teacher = teacher.float()
    kl = F.kl_div(F.log_softmax(student, -1), F.softmax(teacher, -1), reduction='batchmean')
    nll = F.cross_entropy(student, torch.tensor([target], device=student.device))
    return {'kl': float(kl), 'teacher_top1_matches': bool(student.argmax(-1).item() == teacher.argmax(-1).item()),
            'sampled_token_nll': float(nll)}


def verify():
    report = json.loads((OUT / 'result.json').read_text(encoding='utf-8'))
    tails_path = OUT / 'position-tails.pt'
    trajectory_path = OUT / 'teacher-trajectories.pt'
    sources = {index: PAIR / f'layer-{index}.pt' for index in (6, 12)}
    source_hashes = {str(index): base.sha(path) for index, path in sources.items()}
    if (base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA
            or report['base_sha256'] != base.WEIGHT_SHA or report['candidate_sha256'] != source_hashes
            or base.sha(tails_path) != report['position_tails_sha256']
            or base.sha(trajectory_path) != report['trajectories_sha256']):
        raise ValueError('Verification source mismatch')
    trajectories = torch.load(trajectory_path, map_location='cpu', weights_only=True)['trajectories']
    tails = torch.load(tails_path, map_location='cpu', weights_only=True)
    if len(trajectories) != len(report['trajectories']):
        raise ValueError('Trajectory count mismatch')
    for trajectory, recorded in zip(trajectories, report['trajectories']):
        if (len(trajectory['token_ids']) != recorded['total_tokens']
                or trajectory['lengths'] != recorded['lengths'] or trajectory['split'] != recorded['split']):
            raise ValueError('Trajectory metadata mismatch')
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    model = load_full_streamed().eval()
    refs = {}
    with torch.inference_mode():
        for index, trajectory in enumerate(trajectories):
            for length in trajectory['lengths']:
                prefix = trajectory['token_ids'][:length].unsqueeze(0).to('cuda')
                refs[index, length] = model(input_ids=prefix, use_cache=False, logits_to_keep=1,
                                             return_dict=True).logits[:, -1].detach().half().cpu().clone()
    replacements = {index: _load_candidate(path, index, model).eval() for index, path in sources.items()}
    old_hashes = {str(index): tensor_sha(block.position.weight) for index, block in replacements.items()}
    apply_position_tails(replacements, tails, base.WEIGHT_SHA, source_hashes)
    for index, block in replacements.items():
        if tensor_sha(block.position.weight[:128]) != old_hashes[str(index)]:
            raise ValueError('Original position vectors changed on restore')
        model.model.layers[index] = block
    recorded = {(row['trajectory'], row['length']): row for row in report['after']}
    rows = []
    with torch.inference_mode():
        for index, trajectory in enumerate(trajectories):
            for length in trajectory['lengths']:
                prefix = trajectory['token_ids'][:length].unsqueeze(0).to('cuda')
                logits = model(input_ids=prefix, use_cache=False, logits_to_keep=1, return_dict=True).logits[:, -1]
                value = metrics(logits, refs[index, length].to('cuda'), int(trajectory['token_ids'][length]))
                prior = recorded[index, length]
                rows.append({'trajectory': index, 'length': length,
                             'kl_abs_difference': abs(value['kl'] - prior['kl']),
                             'nll_abs_difference': abs(value['sampled_token_nll'] - prior['sampled_token_nll']),
                             'top1_matches_prior': value['teacher_top1_matches'] == prior['teacher_top1_matches']})
    if base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA:
        raise ValueError('Base changed during verification')
    result = {'rows': rows, 'max_kl_abs_difference': max(row['kl_abs_difference'] for row in rows),
              'max_nll_abs_difference': max(row['nll_abs_difference'] for row in rows),
              'top1_matches_prior': sum(row['top1_matches_prior'] for row in rows),
              'limitations': ['Checks aggregate metrics on saved teacher-forced prefixes, not exact full distributions.',
                              'No resumed optimizer or free-generation equivalence tested.']}
    with (OUT / 'verification.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'max_kl': result['max_kl_abs_difference'], 'max_nll': result['max_nll_abs_difference'],
                      'top1_matches': result['top1_matches_prior'], 'total': len(rows)}), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    previous = json.loads((POST.parent / 'result.json').read_text(encoding='utf-8'))
    candidates = {index: PAIR / f'layer-{index}.pt' for index in (6, 12)}
    source_hashes = {str(index): base.sha(path) for index, path in candidates.items()}
    if (base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA
            or base.sha(POST) != previous['candidate_sha256']
            or any(source_hashes[str(index)] != previous['upstream_pair_sha256'][str(index)] for index in candidates)):
        raise ValueError('Base or source candidate mismatch')
    train, held = data.load_reviewed_prompts()
    # Separate source conversations and deterministic fixed selection.
    prompts = [('train', train[1]), ('train', train[2]), ('held', held[2]), ('held', held[3])]
    if len({item['group'] for _, item in prompts}) != len(prompts):
        raise ValueError('Prompt source groups overlap')
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    model.requires_grad_(False)
    trajectories = []
    for index, (split, item) in enumerate(prompts):
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': item['prompt']}],
                                             tokenize=False, add_generation_prompt=True)
        if not text.rstrip().endswith('<think>'):
            raise ValueError('Unexpected natural reasoning prompt')
        encoded = tokenizer(text, return_tensors='pt', add_special_tokens=False).to('cuda')
        torch.manual_seed(SEED + index)
        with torch.inference_mode():
            output = model.generate(**encoded, do_sample=True, temperature=.6, top_p=.95,
                                    max_new_tokens=MAX_NEW, use_cache=True,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        ids = output[0].detach().cpu().clone()
        available = valid_lengths(len(ids), encoded['input_ids'].shape[1])
        trajectories.append({'split': split, 'source_group': item['group'], 'source_line': item['line'],
                             'prompt_tokens': encoded['input_ids'].shape[1], 'token_ids': ids,
                             'lengths': available, 'seed': SEED + index,
                             'ended_with_eos': bool(ids[-1].item() == tokenizer.eos_token_id)})
        print(json.dumps({'split': split, 'generated': len(ids) - encoded['input_ids'].shape[1],
                          'available_lengths': available}), flush=True)
    train_positions = [(i, length) for i, trajectory in enumerate(trajectories) if trajectory['split'] == 'train'
                       for length in trajectory['lengths'] if length in LONG_LENGTHS]
    held_positions = [(i, length) for i, trajectory in enumerate(trajectories) if trajectory['split'] == 'held'
                      for length in trajectory['lengths'] if length in LONG_LENGTHS]
    if len(train_positions) < 4 or len(held_positions) < 4:
        raise ValueError('Insufficient long teacher continuations; no training performed')
    refs = {}
    with torch.inference_mode():
        for index, trajectory in enumerate(trajectories):
            for length in trajectory['lengths']:
                prefix = trajectory['token_ids'][:length].unsqueeze(0).to('cuda')
                logits = model(input_ids=prefix, use_cache=False, logits_to_keep=1, return_dict=True).logits[:, -1]
                refs[(index, length)] = logits.detach().half().cpu().clone()
    refs = {key: value.clone() for key, value in refs.items()}
    replacements = {index: _load_candidate(path, index, model).eval() for index, path in candidates.items()}
    old_hashes = {}
    for index, block in replacements.items():
        if block.max_length != 128:
            raise ValueError('Unexpected source position length')
        block.extend_positions(LIMIT)
        block.requires_grad_(False)
        block.position.weight.requires_grad_(True)
        block.position.weight.register_hook(lambda grad: torch.cat((torch.zeros_like(grad[:128]), grad[128:])))
        old_hashes[str(index)] = tensor_sha(block.position.weight[:128])
        model.model.layers[index] = block

    def evaluate():
        rows = []
        with torch.no_grad():
            for index, trajectory in enumerate(trajectories):
                for length in trajectory['lengths']:
                    prefix = trajectory['token_ids'][:length].unsqueeze(0).to('cuda')
                    logits = model(input_ids=prefix, use_cache=False, logits_to_keep=1, return_dict=True).logits[:, -1]
                    rows.append({'split': trajectory['split'], 'trajectory': index, 'length': length,
                                 **metrics(logits, refs[(index, length)].to('cuda'),
                                           int(trajectory['token_ids'][length]))})
        return rows

    before = evaluate()
    optimizer = torch.optim.AdamW([block.position.weight for block in replacements.values()],
                                  lr=.01, weight_decay=0., foreach=False)
    trace = []
    for step in range(STEPS):
        index, length = train_positions[step % len(train_positions)]
        prefix = trajectories[index]['token_ids'][:length].unsqueeze(0).to('cuda')
        logits = model(input_ids=prefix, use_cache=False, logits_to_keep=1, return_dict=True).logits[:, -1].float()
        teacher = refs[(index, length)].to('cuda').float()
        loss = F.kl_div(F.log_softmax(logits / 2, -1), F.softmax(teacher / 2, -1),
                        reduction='batchmean') * 4
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite distillation loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([block.position.weight for block in replacements.values()],
                                      1., error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % 12 == 0:
            trace.append({'step': step + 1, 'loss': float(loss.detach())})
            print(json.dumps(trace[-1]), flush=True)
    if any(tensor_sha(block.position.weight[:128]) != old_hashes[str(index)]
           for index, block in replacements.items()):
        raise ValueError('An original position changed')
    after = evaluate()
    if (base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA
            or any(base.sha(path) != source_hashes[str(index)] for index, path in candidates.items())):
        raise ValueError('Base or source candidate changed')
    tails = {str(index): block.position.weight.detach().cpu()[128:].clone()
             for index, block in replacements.items()}
    atomic_save({'format': 'spora-position-tail-v1', 'base_weight_sha256': base.WEIGHT_SHA,
                 'sources': source_hashes, 'old_length': 128, 'new_length': LIMIT,
                 'tails': tails}, OUT / 'position-tails.pt')
    atomic_save({'trajectories': trajectories}, OUT / 'teacher-trajectories.pt')
    report = {'base_sha256': base.WEIGHT_SHA, 'candidate_sha256': source_hashes,
              'position_tails_sha256': base.sha(OUT / 'position-tails.pt'),
              'trajectories_sha256': base.sha(OUT / 'teacher-trajectories.pt'),
              'trajectories': [{key: value for key, value in item.items() if key != 'token_ids'}
                               | {'total_tokens': len(item['token_ids'])} for item in trajectories],
              'train_positions': len(train_positions), 'held_positions': len(held_positions),
              'before': before, 'after': after, 'trace': trace, 'steps': STEPS,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Four previously reviewed dialogue prompts, two training and two held source conversations.',
                              'Teacher-generated trajectories are truncated at 384 new tokens when no EOS appears.',
                              'Only selected prefix positions are supervised; teacher forcing is not free-chat quality.',
                              'Only added absolute-position embeddings are trained; other parameters and memory are frozen.',
                              'Zero tail initialization and one seed; old positions are preserved.',
                              'No active SPORA memory, autonomous writing, or long-context acceptance.']}
    with (OUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'seconds': report['seconds'], 'train_positions': len(train_positions),
                      'held_positions': len(held_positions)}), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--verify']:
        verify()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit('Usage: script [--verify]')
