"""Compare one uninterrupted update with a fresh-process resumed update.

This is an isolated, deterministic-input research audit, not a production
recovery protocol or an evaluation of autonomous learning.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch.nn import functional as F
from scripts import deepseek_local_baseline as base
from scripts import deepseek_spora_adapter_restart_audit as previous
from scripts import deepseek_spora_diverse_kl_correction_audit as data
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_online_lora_memory_pilot import OnlineDelta
from spora import research_adapter_checkpoint as checkpoint

OUT = ROOT / 'runs/deepseek-spora-resume-equivalence-20261116'
START = 144
TOTAL = 12
MID = 6
FACTS = {'小林': '豆浆', '阿宁': '红茶', '小雨': '果汁'}


def compare(left, right, name='root'):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            return [{'path': name, 'reason': 'shape or dtype mismatch'}]
        if torch.equal(left, right):
            return []
        delta = (left.float() - right.float()).abs().max().item() if left.numel() else 0.
        return [{'path': name, 'max_abs_difference': delta}]
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return [{'path': name, 'reason': 'keys mismatch'}]
        return [item for key in left for item in compare(left[key], right[key], f'{name}.{key}')]
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        if len(left) != len(right):
            return [{'path': name, 'reason': 'length mismatch'}]
        return [item for index, (a, b) in enumerate(zip(left, right))
                for item in compare(a, b, f'{name}[{index}]')]
    return [] if left == right else [{'path': name, 'reason': 'value mismatch'}]


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def run(mode):
    started = time.perf_counter()
    if mode == 'reference':
        OUT.mkdir(parents=True, exist_ok=False)
    elif not OUT.is_dir():
        raise ValueError('Run reference first')
    if (OUT / f'{mode}.json').exists() or (OUT / f'{mode}.pt').exists():
        raise FileExistsError(mode)

    binding = json.loads((previous.OUT / 'binding.json').read_text(encoding='utf-8'))
    feature_sha, files = previous.binding()
    weight_sha = base.sha(base.MODEL / 'model.safetensors')
    if (weight_sha != base.WEIGHT_SHA or weight_sha != binding['base_sha256']
            or feature_sha != binding['feature_sha256'] or files != binding['files']
            or base.sha(previous.OUT / 'candidate.pt') != binding['candidate_sha256']):
        raise ValueError('Prior candidate or base binding mismatch')
    torch.manual_seed(data.SEED)
    torch.set_num_threads(2)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    model.requires_grad_(False)
    modules = []
    for layer in previous.LAYERS:
        mlp = model.model.layers[layer].mlp
        mlp.down_proj = OnlineDelta(mlp.down_proj, 8)
        modules.append(mlp.down_proj)
    params = [parameter for module in modules for part in (module.down, module.up)
              for parameter in part.parameters()]
    optimizer = torch.optim.AdamW(params, lr=.001, weight_decay=0, foreach=False)
    source = OUT / 'mid.pt' if mode == 'resume' else previous.OUT / 'candidate.pt'
    if mode == 'resume':
        mid = json.loads((OUT / 'mid.json').read_text(encoding='utf-8'))
        if base.sha(source) != mid['candidate_sha256']:
            raise ValueError('Intermediate checkpoint digest mismatch')
    info = checkpoint.restore(source, modules, base_sha256=weight_sha, feature_sha256=feature_sha,
                              layers=previous.LAYERS, optimizer=optimizer,
                              allow_legacy_optimizer=(mode != 'resume'))
    begin, end = (MID, TOTAL) if mode == 'resume' else (0, MID if mode == 'mid' else TOTAL)
    if info['steps'] != START + begin:
        raise ValueError('Unexpected starting step')

    def loss_for(question, answer):
        inputs = pilot.closed_think_inputs(tokenizer, question, 'cuda')['input_ids']
        tokens = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
        target = torch.tensor([tokens], dtype=torch.long, device='cuda')
        full = torch.cat((inputs, target[:, :-1]), dim=1)
        scores = model(input_ids=full, use_cache=False, return_dict=True).logits
        scores = scores[:, inputs.shape[1] - 1:inputs.shape[1] - 1 + target.shape[1]].float()
        return F.cross_entropy(scores.reshape(-1, scores.shape[-1]), target.reshape(-1))

    names = tuple(FACTS)
    losses = []
    for step in range(begin, end):
        name = names[(step // len(data.TRAIN_TEMPLATES)) % len(names)]
        template = data.TRAIN_TEMPLATES[step % len(data.TRAIN_TEMPLATES)]
        unknown = data.TRAIN_UNKNOWN[(step // len(data.TRAIN_TEMPLATES)) % len(data.TRAIN_UNKNOWN)]
        optimizer.zero_grad(set_to_none=True)
        fact_loss = loss_for(template.format(name=name), FACTS[name])
        fact_loss.backward()
        unknown_loss = loss_for(template.format(name=unknown), '我不知道。')
        (0.5 * unknown_loss).backward()
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append({'step': step, 'fact': float(fact_loss.detach()),
                       'unknown': float(unknown_loss.detach())})
        print(json.dumps({'mode': mode, 'step': step + 1, 'loss': losses[-1]['fact']}), flush=True)
    checkpoint.save(OUT / f'{mode}.pt', modules, optimizer, base_sha256=weight_sha,
                    feature_sha256=feature_sha, layers=previous.LAYERS, steps=START + end)
    result = {'mode': mode, 'start': begin, 'end': end, 'losses': losses,
              'candidate_sha256': base.sha(OUT / f'{mode}.pt')}
    if mode != 'mid':
        questions = [(group, template.format(name=name)) for name in FACTS
                     for group, template in (('known_train', data.TRAIN_TEMPLATES[0]),
                                             ('known_held', data.HELD_TEMPLATES[0]))]
        questions += [('unknown', data.HELD_TEMPLATES[0].format(name=name)) for name in data.HELD_UNKNOWN]
        train, held = data.load_reviewed_prompts()
        questions += [('ordinary_train', item['prompt']) for item in train]
        questions += [('ordinary_held', item['prompt']) for item in held]
        result['answers'] = [{'group': group, 'question': question,
                              'answer': pilot.generate(model, tokenizer, question)}
                             for group, question in questions]
    if base.sha(base.MODEL / 'model.safetensors') != weight_sha:
        raise ValueError('Base weights changed')
    result['seconds'] = time.perf_counter() - started
    write_json(OUT / f'{mode}.json', result)
    if mode == 'resume':
        reference = json.loads((OUT / 'reference.json').read_text(encoding='utf-8'))
        mid = json.loads((OUT / 'mid.json').read_text(encoding='utf-8'))
        for label in ('reference', 'mid'):
            meta = reference if label == 'reference' else mid
            if base.sha(OUT / f'{label}.pt') != meta['candidate_sha256']:
                raise ValueError('Audit checkpoint digest mismatch')
        a = torch.load(OUT / 'reference.pt', map_location='cpu', weights_only=True)
        b = torch.load(OUT / 'resume.pt', map_location='cpu', weights_only=True)
        report = {'checkpoint_differences': compare(a, b),
                  'loss_differences': compare(reference['losses'], mid['losses'] + result['losses']),
                  'answer_differences': compare(reference['answers'], result['answers']),
                  'limitations': ['12 fixed, labeled updates with deterministic data ordering.',
                                  'One machine and software environment; no crash or power-loss test.',
                                  'Exact continuation not established for stochastic training or other hardware.']}
        write_json(OUT / 'comparison.json', report)
        print(json.dumps({key: len(value) for key, value in report.items() if key.endswith('_differences')}),
              flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('reference', 'mid', 'resume'))
    run(parser.parse_args().mode)
