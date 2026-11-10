"""Smoke-test autoregressive generation with one SPORA block replacement."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from spora.structure_transfer import SPORALayerReplacement

SOURCE = ROOT / 'runs/deepseek-spora-block-distill-layer12-logit-warmstart-20261111/replacement.pt'
OUTPUT = ROOT / 'runs/deepseek-spora-cached-generation-20261111.json'
LAYER = 12
PROMPTS = ('你好，请用一句话介绍你自己。', '小林过去喜欢咖啡，现在喜欢水。她过去喜欢什么？')


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base model hash mismatch')
    checkpoint = torch.load(SOURCE, map_location='cpu', weights_only=True)
    metadata = checkpoint.get('metadata', {})
    if (metadata.get('base_weight_sha256') != WEIGHT_SHA or metadata.get('layer') != LAYER
            or metadata.get('max_length') != 128):
        raise ValueError('Replacement checkpoint provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float16,
        attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    replacement = SPORALayerReplacement(base.config.hidden_size, base.config.num_attention_heads,
                                        max_length=128, layer_idx=LAYER).to('cuda')
    replacement.load_state_dict(checkpoint['state'], strict=True)
    replacement.eval()
    original_layer = base.model.layers[LAYER]
    base.model.layers[LAYER] = replacement
    rows = []
    try:
        with torch.inference_mode():
            for prompt in PROMPTS:
                rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                                         tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(rendered, return_tensors='pt').to('cuda')
                torch.cuda.synchronize()
                generated = base.generate(**inputs, max_new_tokens=8, do_sample=False,
                                          pad_token_id=tokenizer.eos_token_id,
                                          eos_token_id=tokenizer.eos_token_id)
                new_tokens = generated[0, inputs['input_ids'].shape[1]:]
                rows.append({'prompt': prompt, 'generated_tokens': int(new_tokens.numel()),
                             'text': tokenizer.decode(new_tokens, skip_special_tokens=True),
                             'finite': bool(torch.isfinite(generated.float()).all())})
                torch.cuda.synchronize()
    finally:
        base.model.layers[LAYER] = original_layer
    report = {
        'status': 'spora_block_replacement_cached_generation_smoke',
        'base_weight_sha256': WEIGHT_SHA,
        'replacement_sha256': _digest(SOURCE),
        'layer': LAYER,
        'rows': rows,
        'elapsed_seconds': time.perf_counter() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'device': torch.cuda.get_device_name(0),
        'limitations': [
            'Only one SPORA replacement layer and two short prompts.',
            'Eight-token greedy smoke test, not quality or open-chat acceptance.',
            'Replacement uses learned absolute positions and is limited to its trained 128-position range.',
            'No memory reader was attached at layer 12 in this smoke test.',
        ],
    }
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'generated_tokens': [row['generated_tokens'] for row in rows],
                      'elapsed_seconds': report['elapsed_seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
