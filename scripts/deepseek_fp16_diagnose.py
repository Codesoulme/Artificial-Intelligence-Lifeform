"""Locate nonfinite FP16 activations in a single read-only DeepSeek forward."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha

OUTPUT = ROOT / 'runs/deepseek-r1-1.5b-fp16-diagnostic-20261109.json'


def run(detail=False, sdpa=False):
    output_path = (OUTPUT.with_name('deepseek-r1-1.5b-fp16-sdpa-20261109.json') if sdpa else
                   OUTPUT.with_name('deepseek-r1-1.5b-fp16-detail-20261109.json') if detail else OUTPUT)
    if output_path.exists():
        raise FileExistsError(output_path)
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Weight mismatch')
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    attention = 'sdpa' if sdpa else 'eager'
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation=attention).to('cuda').eval()
    stats, handles = [], []
    def hook(index):
        def inspect(_module, _inputs, output):
            x = output[0] if isinstance(output, tuple) else output
            finite = torch.isfinite(x)
            stats.append({'layer': index, 'all_finite': bool(finite.all()),
                          'nonfinite': int((~finite).sum()),
                          'max_abs_finite': float(x[finite].abs().max()) if bool(finite.any()) else None})
        return inspect
    try:
        if detail:
            layer = model.model.layers[0]
            modules = [('embedding', model.model.embed_tokens), ('input_norm', layer.input_layernorm),
                       ('query', layer.self_attn.q_proj), ('key', layer.self_attn.k_proj),
                       ('value', layer.self_attn.v_proj), ('attention', layer.self_attn),
                       ('post_norm', layer.post_attention_layernorm), ('mlp_gate', layer.mlp.gate_proj),
                       ('mlp_up', layer.mlp.up_proj), ('mlp_down', layer.mlp.down_proj), ('layer0', layer)]
        else:
            modules = list(enumerate(model.model.layers))
        for i, layer in modules:
            handles.append(layer.register_forward_hook(hook(i)))
        formatted = tokenizer.apply_chat_template([{'role': 'user', 'content': '你好，请简单介绍一下你能做什么。'}],
                                                   tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors='pt', add_special_tokens=False).to('cuda')
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits[:, -1, :].float()
            logits_finite = bool(torch.isfinite(logits).all())
            nonfinite_logits = int((~torch.isfinite(logits)).sum())
    finally:
        for handle in handles:
            handle.remove()
    report = {'status': 'fp16_forward_diagnostic_only', 'weights_sha256': WEIGHT_SHA, 'attention': attention,
              'layers': stats, 'logits_finite': logits_finite, 'nonfinite_logits': nonfinite_logits,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Single prompt, single forward; this does not diagnose all generation positions.']}
    with output_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detail', action='store_true')
    parser.add_argument('--sdpa', action='store_true')
    args = parser.parse_args()
    run(args.detail, args.sdpa)
