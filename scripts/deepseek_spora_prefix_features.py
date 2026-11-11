"""Research-only reduced-memory feature model for layers 6 and 18.

Load the first 21 Qwen layers and both SPORA replacements. This is not a
generation model and must be checked against a full-model reference before
using its features to draw conclusions.

Transformers records Qwen decoder layers but not the two replacement classes.
An extra layer beyond the recorded index is needed because its final captured
state is replaced with the final model-normalized output.
"""
import sys
import json
import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoConfig, AutoModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_memory_answer_distill import _load_candidate
from scripts.deepseek_spora_recalibrate_name_router import SOURCE, OUTPUT as ROUTER
from spora.integrated_checkpoint import replacement_feature_variant_sha256


def read_bf16_tensor(stream, header, data_start, file_size, name):
    """Read a validated BF16 tensor without PyTorch's file-backed storage."""
    item = header[name]
    if item['dtype'] != 'BF16':
        raise ValueError(f'Unsupported tensor type for {name}')
    start, end = item['data_offsets']
    shape = item['shape']
    if (not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start
            or any(not isinstance(dim, int) or dim < 0 for dim in shape)
            or end - start != math.prod(shape) * 2 or data_start + end > file_size):
        raise ValueError(f'Invalid tensor range or shape: {name}')
    stream.seek(data_start + start)
    buffer = bytearray(stream.read(end - start))
    if len(buffer) != end - start:
        raise ValueError(f'Truncated tensor: {name}')
    return torch.frombuffer(buffer, dtype=torch.bfloat16).reshape(shape)


def load_prefix():
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Base weight digest mismatch')
    router_file = ROUTER / 'router.pt'
    router_report = __import__('json').loads((ROUTER / 'result.json').read_text(encoding='utf-8'))
    if sha(router_file) != router_report['router_sha256']:
        raise ValueError('Router digest mismatch')
    router = torch.load(router_file, map_location='cpu', weights_only=True)
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    if config.model_type != 'qwen2' or config.num_hidden_layers < 21:
        raise ValueError('Unsupported prefix architecture')
    config.num_hidden_layers = 21
    with torch.device('meta'):
        base = AutoModel.from_config(config, torch_dtype=torch.float16, attn_implementation='sdpa')
    # Rotary frequencies are nonpersistent buffers and are not in safetensors.
    base.rotary_emb = Qwen2RotaryEmbedding(config=config)
    required = set(base.state_dict())
    weights = {}
    path = MODEL / 'model.safetensors'
    with path.open('rb') as source:
        header_size = struct.unpack('<Q', source.read(8))[0]
        if header_size > 100_000_000 or header_size > path.stat().st_size - 8:
            raise ValueError('Invalid safetensors header size')
        header = json.loads(source.read(header_size))
        data_start = 8 + header_size
        for name in sorted(required):
            disk_name = 'model.' + name
            if disk_name not in header:
                raise ValueError(f'Missing prefix tensor: {disk_name}')
            tensor = read_bf16_tensor(source, header, data_start, path.stat().st_size, disk_name)
            weights[name] = tensor.to(device='cuda', dtype=torch.float16)
    base.load_state_dict(weights, strict=True, assign=True)
    del weights
    base.rotary_emb.to('cuda')
    blocks = {index: _load_candidate(SOURCE / f'layer-{index}.pt', index, base).eval()
              for index in (6, 12)}
    variant = replacement_feature_variant_sha256(MODEL, blocks)
    if variant != router.get('feature_variant_sha256'):
        raise ValueError('Prefix replacements differ from runtime router provenance')
    for index, block in blocks.items():
        base.layers[index] = block
    if any(param.is_meta for param in base.parameters()) or any(buf.is_meta for buf in base.buffers()):
        raise ValueError('Prefix model contains unmaterialized tensors')
    base.requires_grad_(False).eval()
    return base, router, variant
