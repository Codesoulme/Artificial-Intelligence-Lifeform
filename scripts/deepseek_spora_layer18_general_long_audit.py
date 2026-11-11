"""Compare longer general Chinese continuations with and without a third graft."""
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoTokenizer
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, _load_candidate

OUTPUT = ROOT / 'runs/deepseek-spora-layer18-general-long-audit-20261112.json'
LAYERS = {
    'sequence': ROOT / 'runs/deepseek-spora-layer18-sequence-distill-20261111',
    'broad': ROOT / 'runs/deepseek-spora-layer18-broad-pilot-20261112',
}
QUESTIONS = (
    '请简单解释一年为什么有四季。', '如何有礼貌地向他人求助？',
    '介绍一条小河的景色。', '请解释什么是耐心。',
    '写一句适合早晨的问候。', '怎样整理一间凌乱的房间？',
    '说说植物生长需要哪些条件。', '给第一次学骑车的人一个建议。',
    '介绍夏天的一个特点。', '请用简单的话解释什么是公正。',
    '雨天出门需要注意什么？', '怎样让朋友感受到被认真倾听？',
)
MAX_TOKENS = 48


def repetition(ids, n):
    groups = [tuple(ids[i:i+n]) for i in range(len(ids) - n + 1)]
    if not groups:
        return 0.0
    return sum(max(0, count - 1) for count in Counter(groups).values()) / len(groups)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checks = {}
    for name, directory in LAYERS.items():
        report = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        layer = directory / 'layer-18.pt'
        if sha(layer) != report['candidate_sha256'] or any(
                sha(PAIR / f'layer-{i}.pt') != report['upstream_pair_sha256'][str(i)] for i in (6, 12)):
            raise ValueError('Layer provenance mismatch')
        checks[name] = report['candidate_sha256']
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed(skip_replaced=(6, 12)).eval()
    for i in (6, 12):
        base.model.layers[i] = _load_candidate(PAIR / f'layer-{i}.pt', i, base).eval()
    original18 = base.model.layers[18]
    if any(p.is_meta for p in base.parameters()) or any(b.is_meta for b in base.buffers()):
        raise ValueError('Unmaterialized model after replacement')
    encoded = [closed_think_inputs(tokenizer, q, 'cuda') for q in QUESTIONS]
    if any(x['input_ids'].shape[1] + MAX_TOKENS > 128 for x in encoded):
        raise ValueError('Question plus generation exceeds candidate position limit')
    outputs = {}
    for mode in ('pair', 'sequence', 'broad'):
        if mode == 'pair':
            base.model.layers[18] = original18
        else:
            base.model.layers[18] = _load_candidate(LAYERS[mode] / 'layer-18.pt', 18, base).eval()
        results = []
        for question, inputs in zip(QUESTIONS, encoded):
            with torch.inference_mode():
                ids = base.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False, use_cache=False,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            tail = ids[0, inputs['input_ids'].shape[1]:].tolist()
            eos = tokenizer.eos_token_id
            terminated = bool(tail and eos is not None and tail[-1] == eos)
            content = tail[:-1] if terminated else tail
            row = {'question': question, 'answer': tokenizer.decode(content, skip_special_tokens=True).strip(),
                   'token_count': len(content), 'terminated_eos': terminated,
                   'repeat_2gram_fraction': repetition(content, 2),
                   'repeat_3gram_fraction': repetition(content, 3)}
            results.append(row)
            print(json.dumps({'mode': mode, 'index': len(results), 'tokens': len(content),
                              'repeat_3gram': round(row['repeat_3gram_fraction'], 3)}), flush=True)
        outputs[mode] = results
    summary = {}
    for mode, rows in outputs.items():
        summary[mode] = {'cases': len(rows), 'ended_with_eos': sum(r['terminated_eos'] for r in rows),
                         'any_repeated_3gram': sum(r['repeat_3gram_fraction'] > 0 for r in rows),
                         'mean_3gram_repeat_fraction': sum(r['repeat_3gram_fraction'] for r in rows) / len(rows)}
        if mode != 'pair':
            summary[mode]['exact_same_as_pair'] = sum(r['answer'] == base_row['answer']
                                                       for r, base_row in zip(rows, outputs['pair']))
    report = {'status': 'general_generation_development_audit', 'summary': summary, 'rows': outputs,
              'base_sha256': WEIGHT_SHA, 'candidate_sha256': checks, 'max_new_tokens': MAX_TOKENS,
              'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Twelve authored short prompts; some themes overlap prior development and training.',
                              'Exact output match is not quality or semantic accuracy; teacher is not ground truth.',
                              'N-gram repetition is a rough indicator and may include legitimate repetition.',
                              'Generation capped at 48 tokens; reaching cap does not prove nontermination.',
                              'Memory readers are inactive; no memory or individual-state evaluation.',
                              'Research streamed loader; original base layers 6/12 omitted and replaced before use.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
