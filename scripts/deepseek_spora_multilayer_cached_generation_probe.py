"""Smoke-test autoregressive generation with SPORA candidates at layers 6 and 12."""
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

CANDIDATES = {
    6: ROOT / 'runs/deepseek-spora-block-distill-layer6-20261111/replacement.pt',
    12: ROOT / 'runs/deepseek-spora-block-distill-layer12-logit-warmstart-20261111/replacement.pt',
}
OUTPUT = ROOT / 'runs/deepseek-spora-multilayer-cached-generation-20261111.json'
PROMPTS = ('你好，请用一句话介绍你自己。', '冬天对着玻璃哈气，为什么玻璃会变模糊？')


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
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float16,
        attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    original = {}
    replacement_hashes = {}
    for layer_index, path in CANDIDATES.items():
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if (payload['metadata'].get('base_weight_sha256') != WEIGHT_SHA
                or payload['metadata'].get('layer') != layer_index):
            raise ValueError(f'Candidate provenance mismatch at layer {layer_index}')
        replacement = SPORALayerReplacement(base.config.hidden_size, base.config.num_attention_heads,
                                            max_length=payload['metadata']['max_length'],
                                            layer_idx=layer_index).to('cuda').eval()
        replacement.load_state_dict(payload['state'], strict=True)
        original[layer_index] = base.model.layers[layer_index]
        base.model.layers[layer_index] = replacement
        replacement_hashes[str(layer_index)] = _digest(path)
    rows = []
    try:
        with torch.inference_mode():
            for prompt in PROMPTS:
                rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                                         tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(rendered, return_tensors='pt').to('cuda')
                generated = base.generate(**inputs, max_new_tokens=8, do_sample=False,
                                          pad_token_id=tokenizer.eos_token_id,
                                          eos_token_id=tokenizer.eos_token_id)
                new_tokens = generated[0, inputs['input_ids'].shape[1]:]
                rows.append({'prompt': prompt, 'generated_tokens': int(new_tokens.numel()),
                             'text': tokenizer.decode(new_tokens, skip_special_tokens=True),
                             'finite': bool(torch.isfinite(generated.float()).all())})
    finally:
        for layer_index, module in original.items():
            base.model.layers[layer_index] = module
    report = {
        'status': 'two_spora_layer_cached_generation_smoke',
        'base_weight_sha256': WEIGHT_SHA,
        'candidate_sha256': replacement_hashes,
        'layers': sorted(original),
        'rows': rows,
        'elapsed_seconds': time.perf_counter() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'device': torch.cuda.get_device_name(0),
        'limitations': [
            'Two prompts and eight greedy tokens each; not a response-quality evaluation.',
            'Layer-6 and layer-12 blocks were distilled separately, not jointly trained.',
            'The SPORA memory reader is not active in this generation smoke test.',
            'Both replacement blocks use 128-position learned absolute embeddings.',
        ],
    }
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'generated_tokens': [row['generated_tokens'] for row in rows],
                      'elapsed_seconds': report['elapsed_seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
