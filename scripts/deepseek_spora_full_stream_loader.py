"""Research-only full Qwen loader avoiding file-backed PyTorch storage."""
import json
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
from scripts.deepseek_spora_prefix_features import read_bf16_tensor


def load_full_streamed():
    """Load verified base on CUDA; callers must separately validate behavior."""
    path = MODEL / 'model.safetensors'
    if sha(path) != WEIGHT_SHA:
        raise ValueError('Base weight digest mismatch')
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    if config.model_type != 'qwen2' or config.tie_word_embeddings:
        raise ValueError('Unsupported architecture or tied embeddings')
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
        if set(header) - {'__metadata__'} != set(expected):
            raise ValueError('Base tensor key mismatch')
        weights = {}
        for name, reference in expected.items():
            if header[name]['shape'] != list(reference.shape):
                raise ValueError(f'Base tensor shape mismatch: {name}')
            tensor = read_bf16_tensor(stream, header, 8 + header_size, file_size, name)
            weights[name] = tensor.to(device='cuda', dtype=torch.float16)
        base.load_state_dict(weights, strict=True, assign=True)
        del weights
    base.model.rotary_emb.to('cuda')
    if any(param.is_meta for param in base.parameters()) or any(buf.is_meta for buf in base.buffers()):
        raise ValueError('Unmaterialized base tensors')
    base.generation_config = GenerationConfig.from_pretrained(MODEL, local_files_only=True)
    base.requires_grad_(False).eval()
    return base
