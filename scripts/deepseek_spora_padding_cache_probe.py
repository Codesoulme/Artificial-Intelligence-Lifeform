"""Check real-model left-padding and cached continuation with SPORA blocks."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, _load_candidate

POST = ROOT/'runs/deepseek-spora-layer18-sequence-distill-20261111/layer-18.pt'
OUTPUT = ROOT/'runs/deepseek-spora-padding-cache-probe-20261111.json'
TEXTS = ('今天天气不错。', '请用一句简单的话解释为什么春天会有花朵。')


def positions(mask):
    return (mask.long().cumsum(-1) - 1).clamp_min(0)


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base model hash mismatch')
    prior = json.loads((POST.parent/'result.json').read_text(encoding='utf-8'))
    if sha(POST) != prior['candidate_sha256'] or any(
            sha(PAIR/f'layer-{i}.pt') != prior['upstream_pair_sha256'][str(i)] for i in (6, 12)):
        raise ValueError('Candidate provenance mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    originals = {i: base.model.layers[i] for i in (6, 12, 18)}
    replacements = {i: _load_candidate(PAIR/f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    replacements[18] = _load_candidate(POST, 18, base).eval()
    batch = tokenizer(list(TEXTS), return_tensors='pt', padding=True).to('cuda')
    mask = batch['attention_mask']
    pos = positions(mask)
    result = {}
    with torch.inference_mode():
        for mode, replaced in (('original', ()), ('pair', (6, 12)), ('triple', (6, 12, 18))):
            for index in (6, 12, 18):
                base.model.layers[index] = replacements[index] if index in replaced else originals[index]
            full = base(**batch, position_ids=pos, use_cache=False).logits[:, -1].float()
            singles = []
            for text in TEXTS:
                item = tokenizer(text, return_tensors='pt').to('cuda')
                one = base(**item, position_ids=positions(item['attention_mask']), use_cache=False)
                singles.append(one.logits[:, -1].float())
            singles = torch.cat(singles, dim=0)
            cached = base(**batch, position_ids=pos, use_cache=True)
            next_token = full.argmax(-1, keepdim=True)
            extended_mask = torch.cat((mask, torch.ones_like(next_token)), dim=1)
            extended_ids = torch.cat((batch['input_ids'], next_token), dim=1)
            extended = base(input_ids=extended_ids, attention_mask=extended_mask,
                            position_ids=positions(extended_mask), use_cache=False).logits[:, -1].float()
            continuation = base(input_ids=next_token, attention_mask=extended_mask,
                                position_ids=positions(extended_mask)[:, -1:],
                                past_key_values=cached.past_key_values, use_cache=True).logits[:, -1].float()
            result[mode] = {
                'batch_vs_single_max_logit_delta': float((full-singles).abs().max()),
                'batch_vs_single_argmax_equal': bool(torch.equal(full.argmax(-1), singles.argmax(-1))),
                'prefill_vs_full_max_logit_delta': float((full-cached.logits[:, -1].float()).abs().max()),
                'cached_vs_full_continuation_max_logit_delta': float((extended-continuation).abs().max()),
                'cached_vs_full_continuation_argmax_equal': bool(torch.equal(extended.argmax(-1), continuation.argmax(-1))),
            }
            print(json.dumps({mode: result[mode]}), flush=True)
    report = {'texts': TEXTS, 'batch_attention_mask': mask.tolist(), 'results': result,
              'base_sha256': WEIGHT_SHA, 'pair_sha256': {str(i): sha(PAIR/f'layer-{i}.pt') for i in (6, 12)},
              'layer18_sha256': sha(POST), 'elapsed_seconds': time.perf_counter()-started,
              'limitations': ['Two short standalone prompts; no integrated memory reader.',
                              'FP16 floating point comparisons may vary by hardware.',
                              'Parity checks do not measure language quality or open chat.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    run()
