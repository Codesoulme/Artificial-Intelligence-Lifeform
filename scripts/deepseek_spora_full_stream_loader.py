"""Research-only full Qwen loader avoiding file-backed PyTorch storage."""
import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha


def stream_fp16_tensor(stream, item, data_start, file_size, device):
    """Convert BF16 weights in bounded chunks, including the vocabulary matrices."""
    shape = item['shape']
    start, end = item['data_offsets']
    if (item['dtype'] != 'BF16' or any(type(dim) is not int or dim < 0 for dim in shape)
            or type(start) is not int or type(end) is not int or start < 0
            or end - start != math.prod(shape) * 2 or data_start + end > file_size):
        raise ValueError('Invalid BF16 tensor range')
    result = torch.empty(shape, dtype=torch.float16, device=device)
    flat = result.view(-1)
    if not flat.numel():
        return result
    buffer = bytearray(min(end - start, 8 * 1024 * 1024))
    chunk_elements = len(buffer) // 2
    stream.seek(data_start + start)
    for offset in range(0, flat.numel(), chunk_elements):
        count = min(chunk_elements, flat.numel() - offset)
        if stream.readinto(memoryview(buffer)[:count * 2]) != count * 2:
            raise ValueError('Truncated BF16 tensor')
        source = torch.frombuffer(buffer, dtype=torch.bfloat16, count=count)
        flat[offset:offset + count].copy_(source)
    return result


def load_full_streamed(*, skip_replaced=(), stage_embeddings=False, stage_output_projection=False):
    """Load verified base on CUDA; skipped meta layers MUST be replaced before use.

    ``skip_replaced`` is research-only and saves peak memory when a verified
    integrated checkpoint replaces those exact layers immediately afterwards.
    The caller must verify all meta tensors are gone before inference.
    """
    return load_qwen_streamed(MODEL, WEIGHT_SHA, skip_replaced=skip_replaced, stage_embeddings=stage_embeddings,
                              stage_output_projection=stage_output_projection)


def load_qwen_streamed(model_dir, expected_sha, *, skip_replaced=(), stage_embeddings=False,
                       stage_output_projection=False):
    path = Path(model_dir) / 'model.safetensors'
    if sha(path) != expected_sha:
        raise ValueError('Base weight digest mismatch')
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    if config.model_type != 'qwen2':
        raise ValueError('Expected Qwen2 weights')
    if not isinstance(skip_replaced, tuple) or any(type(i) is not int or not 0 <= i < config.num_hidden_layers
                                                   for i in skip_replaced) or len(set(skip_replaced)) != len(skip_replaced):
        raise ValueError('Invalid layers to be replaced')
    with torch.device('meta'):
        base = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, attn_implementation='sdpa')
    base.model.rotary_emb = Qwen2RotaryEmbedding(config=config)
    expected = base.state_dict()
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('Truncated safetensors header')
        header_size = struct.unpack('<Q', prefix)[0]
        file_size = path.stat().st_size
        if header_size > 100_000_000 or header_size > file_size - 8:
            raise ValueError('Invalid safetensors header length')
        header = json.loads(stream.read(header_size))
        tied_head = {'lm_head.weight'} if config.tie_word_embeddings and 'lm_head.weight' not in header else set()
        if set(header) - {'__metadata__'} != set(expected) - tied_head:
            raise ValueError('Base tensor key mismatch')
        weights = {}
        skipped = {name for name in expected if any(name.startswith(f'model.layers.{index}.') for index in skip_replaced)}
        if skip_replaced and not skipped:
            raise ValueError('No tensors matched replacement layers')
        for name, reference in expected.items():
            if name in tied_head:
                continue
            if header[name]['shape'] != list(reference.shape):
                raise ValueError(f'Base tensor shape mismatch: {name}')
            if name in skipped:
                continue
            host = ((stage_embeddings and name in ('model.embed_tokens.weight', 'lm_head.weight'))
                    or (stage_output_projection and name == 'lm_head.weight'))
            device = 'cpu' if host else 'cuda'
            weights[name] = stream_fp16_tensor(stream, header[name], 8 + header_size, file_size, device)
        incompatible = base.load_state_dict(weights, strict=not (skipped or tied_head), assign=True)
        if set(incompatible.missing_keys) != skipped | tied_head or incompatible.unexpected_keys:
            raise ValueError('Unexpected missing or extra base tensors')
        if tied_head:
            base.tie_weights()
        del weights
    base.model.rotary_emb.to('cuda')
    if (any(param.is_meta and name not in skipped for name, param in base.named_parameters())
            or any(buf.is_meta for buf in base.buffers())):
        raise ValueError('Unmaterialized base tensors')
    base.generation_config = GenerationConfig.from_pretrained(model_dir, local_files_only=True)
    base.requires_grad_(False).eval()
    return base
