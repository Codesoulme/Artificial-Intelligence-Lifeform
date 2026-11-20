"""Transplant pretrained attention into a single SPORA-style QKV projection."""

import argparse
import time
from pathlib import Path

from audit_awake_multiturn_layers import CASES, encode, load_layer_pair, score_memory
from audit_awake_feedforward_memory import history_cases
from deepseek_output_aligned_memory import MODEL, SOURCE, sha, torch, write
from spora.integrated_open_base import SPORAGatedFeedForward
from train_awake_replacement_replay import generate
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb


class FusedAttention(torch.nn.Module):
    """Use one input projection while preserving rotary positions and attention weights."""

    def __init__(self, source, *, keep_groups=False):
        super().__init__()
        if source.config._attn_implementation != 'sdpa' or source.sliding_window is not None:
            raise ValueError('Expected full Qwen2 SDPA attention')
        self.layer_idx = source.layer_idx
        self.head_dim = source.head_dim
        self.scaling = source.scaling
        self.is_causal = source.is_causal
        groups = source.num_key_value_groups
        self.num_key_value_groups = groups if keep_groups else 1
        query, key, value = source.q_proj, source.k_proj, source.v_proj

        def expand(tensor):
            heads = tensor.detach().unflatten(0, (-1, self.head_dim))
            return heads.repeat_interleave(groups, dim=0).flatten(0, 1)

        if key.out_features * groups != query.out_features or value.out_features != key.out_features:
            raise ValueError('Incompatible attention projection dimensions')
        key_weight = key.weight.detach() if keep_groups else expand(key.weight)
        value_weight = value.weight.detach() if keep_groups else expand(value.weight)
        key_bias = key.bias.detach() if keep_groups else expand(key.bias)
        value_bias = value.bias.detach() if keep_groups else expand(value.bias)
        self.sizes = (query.out_features, key_weight.shape[0], value_weight.shape[0])
        weights = torch.cat((query.weight.detach(), key_weight, value_weight))
        bias = torch.cat((query.bias.detach(), key_bias, value_bias))
        self.qkv = torch.nn.Linear(query.in_features, sum(self.sizes), device='meta')
        self.qkv.weight = torch.nn.Parameter(weights, requires_grad=False)
        self.qkv.bias = torch.nn.Parameter(bias, requires_grad=False)
        self.out = source.o_proj

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None,
                cache_position=None, **kwargs):
        shape = hidden_states.shape[:-1]
        query, key, value = (projection.reshape(*shape, -1, self.head_dim).transpose(1, 2)
                             for projection in self.qkv(hidden_states).split(self.sizes, dim=-1))
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        if past_key_values is not None:
            position = dict(sin=sin, cos=cos, cache_position=cache_position)
            key, value = past_key_values.update(key, value, self.layer_idx, position)
        hidden, weights = sdpa_attention_forward(self, query, key, value, attention_mask,
                                                 dropout=0., scaling=self.scaling, **kwargs)
        return self.out(hidden.reshape(*shape, -1).contiguous()), weights


@torch.inference_mode()
def main(output, index, keep_groups=False, fuse_feedforward=False):
    before_digests = {path: sha(path) for path in (SOURCE, MODEL / 'model.safetensors')}
    start = time.perf_counter()
    decoder, tokenizer, _, replacements, organism, _ = load_layer_pair()
    if not 0 <= index < len(decoder.base.model.layers) or index in {*replacements, *decoder.layer_indices}:
        raise ValueError('Select an original decoder layer without a memory wrapper')
    layer = decoder.base.model.layers[index]
    origin = layer.self_attn
    mlp = layer.mlp
    bridge = FusedAttention(origin, keep_groups=keep_groups).eval()
    fused_mlp = SPORAGatedFeedForward(mlp).eval() if fuse_feedforward else mlp
    _, memory = history_cases(organism)
    prompts = [encode(tokenizer, [dict(role='user', content=case[0])]) for case in CASES]
    results = {}
    for name, attention in (('original', origin), ('fused', bridge)):
        layer.self_attn = attention
        layer.mlp = fused_mlp if name == 'fused' else mlp
        rows = []
        for case, prompt in zip(CASES, prompts, strict=True):
            inputs = torch.tensor([prompt], device='cuda')
            logits = decoder(input_ids=inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float().cpu()
            tokens = generate(decoder, tokenizer, prompt, 32)
            rows.append((logits, tokens, tokenizer.decode(tokens, skip_special_tokens=True).strip()))
        memory_rows = score_memory(decoder, tokenizer, memory)
        results[name] = (rows, memory_rows)
        print(name, 'memory', sum(row['correct'] for row in memory_rows), '/', len(memory_rows), flush=True)
    layer.self_attn = origin
    layer.mlp = mlp
    ordinary = [dict(question=case[0], logits_max_abs_difference=(old[0] - new[0]).abs().max().item(),
                     generation_identical=old[1] == new[1], original=old[2], fused=new[2])
                for case, old, new in zip(CASES, results['original'][0], results['fused'][0], strict=True)]
    if any(sha(path) != digest for path, digest in before_digests.items()):
        raise ValueError('A source checkpoint changed during the audit')
    old_count = sum(parameter.numel() for parameter in origin.parameters())
    new_count = sum(parameter.numel() for parameter in bridge.parameters())
    write(output, dict(layer=index, keep_groups=keep_groups, fuse_feedforward=fuse_feedforward,
                       ordinary=ordinary, memory={name: rows[1] for name, rows in results.items()},
                       original_parameters=old_count, fused_parameters=new_count,
                       sources={str(path): digest for path, digest in before_digests.items()},
                       seconds=time.perf_counter()-start, peak_gpu_bytes=torch.cuda.max_memory_allocated()))
    print('Identical generations', sum(row['generation_identical'] for row in ordinary), '/', len(ordinary),
          'max logit difference', max(row['logits_max_abs_difference'] for row in ordinary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layer', type=int, default=18)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--keep-groups', action='store_true', help='Preserve grouped key and value dimensions')
    parser.add_argument('--fuse-feedforward', action='store_true', help='Also fuse the pretrained feed-forward input projections')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose an unused output path')
    main(args.output, args.layer, args.keep_groups, args.fuse_feedforward)
