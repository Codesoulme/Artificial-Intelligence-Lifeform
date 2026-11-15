"""Compare Chinese generation and fixed continuations across structural layers.

No memory read, trained response, or model modification is performed.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch.nn import functional as F
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_addressed_read import closed_think_inputs

OUT = ROOT / 'runs/deepseek-spora-language-layer-comparison-20261116.json'
CASES = (
    ('arithmetic', '小明有三本书，又买了两本，送给朋友一本。他现在有几本书？', '他现在有四本书。'),
    ('practical', '我今天有点累，给我一个简单的休息建议。', '可以先休息一会儿，喝点水。'),
    ('writing', '请用一句话向迟到的同事表示歉意。', '抱歉让你久等了。'),
    ('language', '“开心”的反义词是什么？只回答一个词。', '难过'),
    ('reasoning', '如果天正在下雨，出门前可以带什么？', '可以带一把伞。'),
    ('instruction', '请只回复两个字：你好。', '你好'),
    ('multi_turn', '用户：我喜欢安静的地方。\n助手：你喜欢在安静的地方做什么？\n用户：看书。\n助手：', '你平时喜欢看什么类型的书？'),
    ('uncertainty', '你知道我今天早餐吃了什么吗？', '我不知道你今天早餐吃了什么。'),
)


def score_and_generate(base, tokenizer, question, reference):
    prompt = closed_think_inputs(tokenizer, question, 'cuda')
    prefix = prompt['input_ids'].shape[1]
    target_ids = tokenizer.encode(reference, add_special_tokens=False)
    target = torch.tensor([target_ids], dtype=torch.long, device='cuda')
    ids = torch.cat((prompt['input_ids'], target), dim=1)
    with torch.inference_mode():
        logits = base(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits.float()
        scores = logits[:, prefix - 1:prefix - 1 + len(target_ids)]
        nll = F.cross_entropy(scores.reshape(-1, scores.shape[-1]), target.reshape(-1), reduction='sum')
        generated = base.generate(**prompt, use_cache=False, max_new_tokens=64, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    tokens = generated[0, prefix:].tolist()
    return {'answer': tokenizer.decode(tokens, skip_special_tokens=True).strip(), 'token_ids': tokens,
            'ended': tokenizer.eos_token_id in tokens, 'reference_token_count': len(target_ids),
            'reference_nll_sum': float(nll), 'reference_nll_per_token': float(nll) / len(target_ids)}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    source_sha = p.sha(p.runtime.CHECKPOINT)
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Base checksum mismatch')
    torch.manual_seed(68214)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    rows = [{'category': kind, 'question': question, 'reference': ref} for kind, question, ref in CASES]
    with tempfile.TemporaryDirectory(prefix='spora-language-compare-') as temp:
        p.runtime.WORKING_CHECKPOINT = Path(temp) / 'unused.pt'
        base = p.load_full_streamed().eval()
        original_layers = {i: base.model.layers[i] for i in (6, 12)}
        chat = p.runtime.load_chat(base_loader=lambda: base)
        state_digest = p.frozen_digest(chat.organism, chat.state)
        replacements = {i: base.model.layers[i] for i in original_layers}
        for variant, active in (('original', ()), ('layer12', (12,)), ('both', (6, 12))):
            for i in original_layers:
                base.model.layers[i] = replacements[i] if i in active else original_layers[i]
            for i, row in enumerate(rows):
                row[variant] = score_and_generate(base, chat.tokenizer, row['question'], row['reference'])
                print(json.dumps({'variant': variant, 'done': i + 1, 'total': len(rows)}), flush=True)
        for i in original_layers:
            base.model.layers[i] = replacements[i]
        if p.frozen_digest(chat.organism, chat.state) != state_digest:
            raise ValueError('Individual changed')
    if p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(p.runtime.CHECKPOINT) != source_sha:
        raise ValueError('Protected file changed')
    summary = {}
    for name in ('original', 'layer12', 'both'):
        token_count = sum(row[name]['reference_token_count'] for row in rows)
        summary[name] = {'cases': len(rows), 'reference_tokens': token_count,
                         'mean_reference_nll': sum(row[name]['reference_nll_sum'] for row in rows) / token_count,
                         'eos_within_budget': sum(row[name]['ended'] for row in rows),
                         'same_generation_as_original': sum(row[name]['token_ids'] == row['original']['token_ids'] for row in rows)}
    result = {'rows': rows, 'summary': summary, 'base_sha256': base_sha, 'checkpoint_sha256': source_sha,
              'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Authored small sample and one possible fixed continuation per prompt.',
                              'Reference likelihood is not general chat quality or factual correctness.',
                              'One-turn prompts include simulated history; no self-generated multi-turn evaluation.',
                              'Greedy decoding and 64-token cap may truncate valid responses.',
                              'Memory is off; one-layer replacement is untrained as an independent candidate.',
                              'Original and replaced layers coexist only within an isolated experiment.']}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
