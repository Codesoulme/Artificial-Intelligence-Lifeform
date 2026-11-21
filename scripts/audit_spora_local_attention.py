"""Compare the SPORA design's 256-token window with full attention."""

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_qwen_counterfactual_reply_20261122 import HELD, WORDS, prepare, records
from spora.integrated_checkpoint import load_integrated_runtime
from spora.language_training import frozen_digest
from spora.research_local_attention import local_attention


@torch.inference_mode()
def cache_probe(decoder, ids):
    cache = DynamicCache(config=decoder.base.config)
    torch.cuda.synchronize()
    started = time.perf_counter()
    decoder(input_ids=ids[:, :-1], past_key_values=cache, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    prefill = time.perf_counter() - started
    streamed = decoder(input_ids=ids[:, -1:], past_key_values=cache, use_cache=True, logits_to_keep=1).logits
    ordinary = decoder(input_ids=ids, use_cache=False, logits_to_keep=1).logits
    delta = (ordinary.float() - streamed.float()).abs()
    return {'length': ids.shape[-1], 'prefill_seconds': prefill,
            'cache_tokens_by_layer': [layer.keys.shape[-2] for layer in cache.layers],
            'cache_bytes': sum(t.numel() * t.element_size() for layer in cache.layers for t in (layer.keys, layer.values)),
            'continuation_argmax_equal': bool(ordinary.argmax(-1).eq(streamed.argmax(-1)).all()),
            'max_logit_delta': float(delta.max())}, ordinary[0, -1].float().log_softmax(-1)


@torch.inference_mode()
def answer(decoder, tokenizer, text, controls=None):
    prompt = closed_think_inputs(tokenizer, text, 'cuda')
    result = decoder.generate(**prompt, **dict(controls or {}, use_cache=True), max_new_tokens=12,
                              do_sample=False, temperature=None, top_p=None,
                              eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    tail = result[0, prompt['input_ids'].shape[-1]:].tolist()
    return {'text': tokenizer.decode(tail, skip_special_tokens=True).strip(), 'tokens': tail,
            'prompt_tokens': prompt['input_ids'].shape[-1]}


def main(checkpoint, output):
    if output.exists():
        raise FileExistsError(output)
    prose = ROOT / 'data/chinese-pilot/source.json'
    sources = {path: sha(path) for path in (checkpoint, prose)}
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL)
    decoder = runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    raw = '\n'.join(json.loads(prose.read_text(encoding='utf-8'))['train'])
    source_tokens = tokenizer.encode(raw, add_special_tokens=False)
    filler = tokenizer.decode(source_tokens[:480])
    fixtures = records(runtime.organism, HELD)
    memory_fixtures = [prepare(runtime.organism, fixtures[i]) for i in (0, 1, 6, 7, 16, 17, 22, 23)]
    compatible = decoder.compatible_block_indices
    variants = {'full': (), 'alternating': tuple(i for i in compatible if i % 4 != 0), 'all_local': compatible}
    baseline = {}
    results = {}
    for variant, layers in variants.items():
        with local_attention(decoder, layers, 256) if layers else nullcontext():
            probes = []
            for length in (128, 256, 257, 512):
                if len(source_tokens) < length:
                    raise ValueError('The probe needs more ordinary text')
                ids = torch.tensor([source_tokens[:length]], device='cuda')
                row, dist = cache_probe(decoder, ids)
                if variant == 'full':
                    baseline[length] = dist
                row['parent_vocabulary_kl'] = float(F.kl_div(dist, baseline[length], reduction='sum', log_target=True))
                probes.append(row)
            named = f'资料开头：木牌上写着云桥站。\n下文与地名无关。\n{filler}\n请只回答资料开头的站名。'
            ordinary = {'short': answer(decoder, tokenizer, '五减二是多少？只回答数字。'),
                        'long_recall': answer(decoder, tokenizer, named)}
            memory = []
            for text, controls, evidence, target in memory_fixtures:
                response = answer(decoder, tokenizer, f'{filler}\n\n请回答以下记录问题：{text}',
                                  dict(controls, memory_output_logits=evidence))
                memory.append(dict(response, question=text, expected=WORDS[target]))
            results[variant] = {'layers': layers, 'cache_probes': probes, 'ordinary': ordinary, 'memory': memory}
            print(variant, 'cache bytes at 512', probes[-1]['cache_bytes'],
                  'long memory exact', sum(row['text'] == row['expected'] for row in memory), flush=True)
    if individual != frozen_digest(runtime.organism, runtime.state) or any(
        sha(path) != digest for path, digest in sources.items()
    ):
        raise ValueError('The window probe changed an input or the individual')
    record = {'window': 256, 'results': results, 'seconds': time.perf_counter() - started,
              'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
              'sources': {str(path): digest for path, digest in sources.items()}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    params = parser.parse_args()
    main(params.checkpoint, params.output)
