"""Compare whole-prefix and chunked inference in the actual replacement model."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_local_baseline as base
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_memory_answer_distill import OUTPUT, _load_candidate


def main(output_path):
    if output_path.exists():
        raise FileExistsError(output_path)
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    paths = {i: OUTPUT / f'layer-{i}.pt' for i in (6, 12)}
    hashes = {str(i): base.sha(path) for i, path in paths.items()}
    model = load_full_streamed(skip_replaced=(6, 12))
    for i, path in paths.items():
        model.model.layers[i] = _load_candidate(path, i, model).eval()
    if any(p.is_meta for p in model.parameters()):
        raise ValueError('Unmaterialized model parameter')
    prompts = ('你好，请简单介绍一下你自己。',
               '我喜欢散步。你能推荐三个适合散步的话题吗？',
               '假如今天下雨，我们可以在家做哪些事情？')
    rows = []
    for prompt in prompts:
        inputs = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                               add_generation_prompt=True, return_tensors='pt').to('cuda')
        length = inputs.shape[1]
        boundaries = (0, length // 3, 2 * length // 3, length)
        if length > 128 or any(a >= b for a, b in zip(boundaries, boundaries[1:])):
            raise ValueError('Invalid audit prompt length')
        with torch.inference_mode():
            whole = model(input_ids=inputs, use_cache=False).logits.float()
            cache = None
            pieces = []
            for a, b in zip(boundaries, boundaries[1:]):
                out = model(input_ids=inputs[:, a:b], past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                pieces.append(out.logits.float())
            chunked = torch.cat(pieces, dim=1)
        delta = (whole - chunked).abs()
        rows.append({'prompt': prompt, 'tokens': length, 'chunks': list(boundaries),
                     'max_abs_logit_error': float(delta.max()),
                     'mean_abs_logit_error': float(delta.mean()),
                     'matching_top1': int((whole.argmax(-1) == chunked.argmax(-1)).sum()),
                     'total_positions': length})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    if (base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA
            or any(base.sha(path) != hashes[str(i)] for i, path in paths.items())):
        raise ValueError('Protected inputs changed')
    report = {'base_sha256': base.WEIGHT_SHA, 'candidate_sha256': hashes, 'rows': rows,
              'seconds': time.perf_counter() - start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Three short fixed prompts; no active memory or generated answers.',
                              'Append-only dynamic cache, no padding or beam search.',
                              'Floating point inference may differ across execution shapes.']}
    with output_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'runs/deepseek-spora-chunk-cache-audit-20261116.json')
    main(parser.parse_args().output)
