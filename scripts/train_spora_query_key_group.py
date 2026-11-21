"""Align one new absolute-position query/key group to the frozen parent."""

import random

import torch
from torch.nn import functional as F

from train_awake_absolute_attention import capture, replay_requests
from train_awake_partial_layer import TRAIN_FILES, corpus


@torch.no_grad()
def collect(runtime, tokenizer, source, group, pairs):
    texts = corpus()[-40:]
    ordinary = [(tokenizer.encode(text, add_special_tokens=False)[:112], {}) for text in texts]
    ordinary = [row for row in ordinary if len(row[0]) > 1]
    memory = replay_requests(runtime.organism, tokenizer, value_offset=4)
    if len(ordinary) < 16:
        raise ValueError('The local fit requires separate training and validation sequences')
    half = source.head_dim // 2
    coordinates = torch.tensor([*pairs, *(half + pair for pair in pairs)], device='cuda')
    start = group * source.num_key_value_groups
    stop = start + source.num_key_value_groups
    samples = []
    for hidden, rotary, mask, positions, _ in capture(runtime.decoder, source, ordinary + memory):
        hidden, positions = hidden.cuda(), positions.cuda()
        query, key, value = source.project(hidden, tuple(part.cuda() for part in rotary), positions)
        query = query[:, start:stop].float()
        key = key[:, group:group + 1].float()
        value = value[:, group:group + 1].float()
        if mask is None:
            mask = torch.zeros((1, 1, hidden.shape[1], hidden.shape[1]), device='cuda').masked_fill(
                torch.ones((hidden.shape[1], hidden.shape[1]), device='cuda', dtype=torch.bool).triu(1), -1e4,
            )
        else:
            mask = mask.cuda().float()
        teacher = ((query @ key.transpose(-1, -2)) * source.scaling + mask).log_softmax(-1)
        features = (hidden + source.position(positions)).float()
        output = teacher.exp() @ value
        samples.append(tuple(tensor.cpu() for tensor in (features, query, key, value, mask, teacher, output)))
    ordinary_samples = samples[:len(ordinary)]
    memory_samples = samples[len(ordinary):]
    return coordinates, ordinary_samples[8:], memory_samples[4:], ordinary_samples[:8] + memory_samples[:4]


def fit_query_key_group(runtime, tokenizer, source, group, pairs, weight, bias, steps):
    coordinates, ordinary, memory, validation = collect(runtime, tokenizer, source, group, pairs)
    width = coordinates.numel()
    heads = source.num_key_value_groups
    weight = torch.nn.Parameter(weight.detach().cuda().float())
    bias = torch.nn.Parameter(bias.detach().cuda().float())
    optimizer = torch.optim.AdamW((weight, bias), lr=8e-6, weight_decay=0.)

    def errors(sample, *, quantize=False):
        features, query, key, value, mask, teacher, output = (part.cuda() for part in sample)
        w, b = (weight.half(), bias.half()) if quantize else (weight, bias)
        projected = F.linear(features.to(w.dtype), w, b).float()
        q, k = projected.split((heads * width, width), dim=-1)
        q = query.index_copy(-1, coordinates, q.unflatten(-1, (heads, width)).transpose(1, 2))
        k = key.index_copy(-1, coordinates, k.unsqueeze(1))
        mixing = ((q @ k.transpose(-1, -2)) * source.scaling + mask).log_softmax(-1)
        mixing_kl = F.kl_div(mixing, teacher, log_target=True, reduction='sum') / mixing[..., 0].numel()
        output_error = ((mixing.exp() @ value) - output).square().mean() / output.square().mean().clamp_min(1e-8)
        return mixing_kl, output_error

    @torch.no_grad()
    def held():
        values = torch.stack([torch.stack(errors(sample, quantize=True)) for sample in validation]).mean(0)
        return {'attention_kl': float(values[0]), 'relative_output_mse': float(values[1])}

    before = held()
    rng = random.Random(221829)
    for step in range(steps):
        mixing, output = errors(rng.choice(memory if step % 3 == 0 else ordinary))
        optimizer.zero_grad(set_to_none=True)
        (mixing + output).backward()
        torch.nn.utils.clip_grad_norm_((weight, bias), 1., error_if_nonfinite=True)
        optimizer.step()
    summary = {'before': before, 'after': held(), 'steps': steps, 'ordinary': len(ordinary),
               'memory': len(memory), 'validation': len(validation)}
    return weight.detach().half().cpu(), bias.detach().half().cpu(), summary
