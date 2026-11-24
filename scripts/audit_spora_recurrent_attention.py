"""Align one migrated SPORA layer to a recurrent history representation."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache, DynamicLayer

from scripts.audit_spora_dialogue_working_state import encode
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_checkpoint import load_integrated_runtime
from spora.language_training import atomic_save, frozen_digest
from spora.pretrained_attention import SPORAFusedAttention, SPORAResidualBlock
from spora.research_recurrent_attention import RecurrentAttention
from train_awake_partial_layer import TRAIN_FILES, corpus


SEED = 24136


class RecurrentDecodeLayer(DynamicLayer):
    """Carry the bounded recurrent state beside the native cache layers."""

    def __init__(self, core):
        super().__init__()
        self.core = core
        self.state = None

    def consume(self, query, key, value, *, include_far):
        output, self.state = self.core(query, key, value, state=self.state, include_far=include_far)
        self.keys, self.values = self.state.key, self.state.value
        self.is_initialized = True
        self.device, self.dtype = self.keys.device, self.keys.dtype
        return output

    def get_seq_length(self):
        return 0 if self.state is None else self.state.length

    def reset(self):
        self.state = self.keys = self.values = None
        self.is_initialized = False

    def update(self, key_states, value_states, cache_kwargs=None):
        raise ValueError('The recurrent layer needs query, key and value together')

    def crop(self, max_length):
        if max_length < 0 or max_length < self.get_seq_length():
            raise ValueError('The recurrent layer cannot undo folded history')

    def reorder_cache(self, beam_idx):
        if beam_idx.shape != (1,) or int(beam_idx[0]) != 0:
            raise ValueError('The recurrent layer supports a single greedy sequence')

    def batch_repeat_interleave(self, repeats):
        if repeats != 1:
            raise ValueError('The recurrent layer supports a single greedy sequence')

    def batch_select_indices(self, indices):
        self.reorder_cache(indices)


class AttentionReplacement(nn.Module):
    def __init__(self, source, core, *, include_far=True):
        super().__init__()
        self.source = source
        self.core = core
        self.include_far = include_far

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None,
                position_ids=None, cache_position=None, **kwargs):
        if hidden_states.shape[0] != 1 or attention_mask is not None:
            raise ValueError('Recurrent attention probe takes one unpadded causal sequence')
        qkv = projections(self.source, hidden_states, position_embeddings, position_ids)
        if past_key_values is None:
            output, _ = self.core(*qkv, include_far=self.include_far)
        else:
            if type(past_key_values) is not DynamicCache or past_key_values.offloading:
                raise ValueError('Recurrent attention probe expects a non-offloaded dynamic cache')
            layers, index = past_key_values.layers, self.source.layer_idx
            if index == len(layers):
                layers.append(RecurrentDecodeLayer(self.core))
            layer = layers[index]
            if type(layer) is DynamicLayer and layer.get_seq_length() == 0:
                layer = layers[index] = RecurrentDecodeLayer(self.core)
            if not isinstance(layer, RecurrentDecodeLayer) or layer.core is not self.core:
                raise ValueError('Recurrent layer cache provenance differs from the active attention')
            if cache_position is None or int(cache_position[0]) != layer.get_seq_length():
                raise ValueError('Recurrent layer cache requires append-only positions')
            output = layer.consume(*qkv, include_far=self.include_far)
        mixed = output.transpose(1, 2).flatten(start_dim=-2).contiguous()
        return self.source.out(mixed.to(self.source.out.weight.dtype)).to(hidden_states.dtype), None


def projections(source, hidden, rope, position_ids):
    query, key, value = source.project(hidden.to(source.qkv.weight.dtype), rope, position_ids)
    groups = query.shape[1] // key.shape[1]
    return query, key.repeat_interleave(groups, dim=1), value.repeat_interleave(groups, dim=1)


def attention_block(decoder, index):
    layer = decoder.base.model.layers[index]
    block = layer.layer if hasattr(layer, 'layer') else layer
    if isinstance(block, SPORAResidualBlock):
        block = block.base
    if not isinstance(getattr(block, 'self_attn', None), SPORAFusedAttention):
        raise ValueError('Recurrent migration expects the fused SPORA attention projections')
    return block


@contextmanager
def replace(block, replacement):
    original = block.self_attn
    block.self_attn = replacement
    try:
        yield
    finally:
        block.self_attn = original


@torch.no_grad()
def capture(decoder, source, sequences):
    features, scores = [], []
    current = {}

    def pre(_module, args, kwargs):
        hidden = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
        q, k, v = projections(source, hidden, kwargs['position_embeddings'], kwargs.get('position_ids'))
        current['qkv'] = tuple(tensor.cpu() for tensor in (q, k, v))
        current['mix'] = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True).half().cpu()

    handle = source.register_forward_pre_hook(pre, with_kwargs=True)
    try:
        for ids in sequences:
            probs = last_logits(decoder, ids)
            scores.append(probs.cpu())
            features.append(current.copy())
    finally:
        handle.remove()
    return features, scores


def last_logits(decoder, ids):
    output = decoder(input_ids=torch.tensor([ids], device=decoder.base.device), use_cache=False, logits_to_keep=1)
    return output.logits[0, -1].float().log_softmax(dim=-1)


@torch.no_grad()
def evaluate(decoder, sequences, targets):
    kl, top = [], []
    for ids, target in zip(sequences, targets, strict=True):
        predicted = last_logits(decoder, ids)
        target = target.to(predicted.device)
        kl.append(float(F.kl_div(predicted, target, log_target=True, reduction='sum')))
        top.append(bool(predicted.argmax() == target.argmax()))
    return {'kl': kl, 'mean_kl': sum(kl) / len(kl), 'top1_matches': sum(top), 'count': len(top)}


@torch.no_grad()
def mix_error(core, rows, device, *, include_far=True):
    errors = []
    for item in rows:
        qkv = tuple(tensor.to(device) for tensor in item['qkv'])
        mixed, _ = core(*qkv, include_far=include_far)
        target = item['mix'].to(device).float()
        suffix = slice(2 * core.block, None)
        found, expected = mixed[:, :, suffix], target[:, :, suffix]
        errors.append(float((found - expected).square().sum() / expected.square().sum().clamp_min(1e-8)))
    return sum(errors) / len(errors)


@torch.no_grad()
def check_chunking(core, item, device):
    qkv = tuple(tensor.to(device) for tensor in item['qkv'])
    expected, end = core(*qkv)
    state, chunks = None, []
    for start in range(0, qkv[0].shape[-2], 7):
        mixed, state = core(*(tensor[:, :, start:start + 7] for tensor in qkv), state=state)
        if state.key.shape[-2] > 2 * core.block:
            raise ValueError('Recent attention state grew beyond its bound')
        chunks.append(mixed)
    torch.testing.assert_close(torch.cat(chunks, dim=-2), expected, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(state.matrix, end.matrix)
    torch.testing.assert_close(state.normalizer, end.normalizer)
    return {'tokens': state.length, 'maximum_recent': 2 * core.block, 'summary_shape': list(state.matrix.shape),
            'max_absolute_error': float((torch.cat(chunks, dim=-2) - expected).abs().max())}


def fit_mix(core, rows, device, steps):
    optimizer = torch.optim.AdamW(core.parameters(), lr=.002)
    rng = random.Random(SEED)
    trace = []
    for step in range(steps):
        item = rng.choice(rows)
        mixed, _ = core(*(x.to(device) for x in item['qkv']))
        target = item['mix'].to(device).float()
        tail = slice(2 * core.block, None)
        loss = F.mse_loss(mixed[:, :, tail], target[:, :, tail]) / target[:, :, tail].square().mean().clamp_min(1e-8)
        optimize(core, optimizer, loss)
        if (step + 1) % 16 == 0 or step + 1 == steps:
            trace.append(float(loss.detach()))
            print('Attention mixing', step + 1, trace[-1], flush=True)
    return trace


def optimize(core, optimizer, loss):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(core.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()


def fit_predictions(decoder, core, sequences, targets, steps):
    optimizer = torch.optim.AdamW(core.parameters(), lr=.0005)
    trace = []
    for step in range(steps):
        index = step % len(sequences)
        predicted = last_logits(decoder, sequences[index])
        loss = F.kl_div(predicted, targets[index].to(predicted.device), log_target=True, reduction='sum')
        optimize(core, optimizer, loss)
        if (step + 1) % 8 == 0 or step + 1 == steps:
            trace.append(float(loss.detach()))
            print('Vocabulary alignment', step + 1, trace[-1], flush=True)
    return trace


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    protected = {path: sha(path) for path in (args.checkpoint, args.revision, *TRAIN_FILES)}
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(args.checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    device = decoder.base.device
    digest = frozen_digest(runtime.organism, runtime.state)
    block = attention_block(decoder, args.layer)
    source = block.self_attn
    training_ids = []
    for text in corpus():
        ids = tokenizer.encode(text, add_special_tokens=False)[:96]
        if len(ids) > 2 * args.block + 8:
            training_ids.append(ids)
        if len(training_ids) == 48:
            break
    if len(training_ids) != 48:
        raise ValueError('Expected 48 distinct calibration passages')
    reference = json.loads(args.revision.read_text(encoding='utf-8'))
    if reference['individual_sha256'] != digest:
        raise ValueError('Dialogue comparison uses a different individual')
    train, prose = training_ids[:36], training_ids[36:]
    dialogue = [encode(tokenizer, [*row['history'], {'role': 'user', 'content': row['case']['question']}])
                for row in reference['oracle']]
    held = prose + dialogue
    if {tuple(ids) for ids in train} & {tuple(ids) for ids in held} or min(map(len, held)) <= 2 * args.block:
        raise ValueError('Invalid fixed-split teacher data')
    local_train, predictions = capture(decoder, source, train)
    local_held, teacher = capture(decoder, source, held)
    heads = source.sizes[0] // source.head_dim
    core = RecurrentAttention(heads, source.head_dim, block=args.block).to(device)
    candidate = AttentionReplacement(source, core)
    metrics = {}
    for name, include_far in (('exact_recent', False), ('untrained', True)):
        candidate.include_far = include_far
        with replace(block, candidate):
            metrics[name] = evaluate(decoder, held, teacher)
        metrics[name]['relative_mix_error'] = mix_error(core, local_held, device, include_far=include_far)
        print(name, metrics[name]['mean_kl'], metrics[name]['relative_mix_error'], flush=True)
    mix_trace = fit_mix(core, local_train, device, args.steps)
    with replace(block, candidate):
        metrics['aligned_mixing'] = evaluate(decoder, held, teacher)
        metrics['aligned_mixing']['relative_mix_error'] = mix_error(core, local_held, device)
        prediction_trace = fit_predictions(decoder, core, train, predictions, args.prediction_steps)
        metrics['aligned_predictions'] = evaluate(decoder, held, teacher)
        metrics['aligned_predictions']['relative_mix_error'] = mix_error(core, local_held, device)
    stream = check_chunking(core, local_held[-1], device)
    if (frozen_digest(runtime.organism, runtime.state) != digest
            or any(sha(path) != value for path, value in protected.items())):
        raise ValueError('A recurrent attention experiment input changed')
    path = args.output / 'attention.pt'
    atomic_save({'format': 'spora-recurrent-attention-v1', 'layer': args.layer, 'block': args.block,
                 'checkpoint_sha256': protected[args.checkpoint], 'core': core.cpu().state_dict()}, path)
    result = {'metrics': metrics, 'mix_trace': mix_trace, 'prediction_trace': prediction_trace,
              'stream': stream, 'layer': args.layer, 'block': args.block, 'train': len(train), 'held_prose': len(prose),
              'held_dialogue': len(dialogue), 'individual_sha256': digest, 'candidate_sha256': sha(path),
              'sources': {str(path): value for path, value in protected.items()},
              'peak_bytes': torch.cuda.max_memory_allocated()}
    with (args.output / 'comparison.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print({name: (row['mean_kl'], row['top1_matches']) for name, row in metrics.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('revision', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--layer', type=int, default=19)
    parser.add_argument('--block', type=int, default=8)
    parser.add_argument('--steps', type=int, default=160)
    parser.add_argument('--prediction-steps', type=int, default=32)
    parameters = parser.parse_args()
    if min(parameters.block, parameters.steps, parameters.prediction_steps) < 1 or parameters.layer < 0:
        parser.error('Use a nonnegative layer and positive training and block sizes')
    main(parameters)
