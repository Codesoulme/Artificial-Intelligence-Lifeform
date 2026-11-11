"""Small held-wording strict-format comparison for a paired-layer candidate."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoTokenizer
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_memory_answer_distill import _load_candidate
from scripts.deepseek_spora_pair_constraints_pilot import trainer

OUTPUT = ROOT / 'runs/deepseek-spora-pair-constraints-format-eval-20261112.json'
CASES = (
    ('只输出一个阿拉伯数字：九减四等于多少？', '5'),
    ('只输出一个阿拉伯数字：七加二等于多少？', '9'),
    ('只输出一个汉字：春夏秋冬中，夏天之前是哪个季节？', '春'),
    ('只输出一个汉字：上、下中，与上相反的是？', '下'),
    ('只输出一个数字：十二减八等于多少？', '4'),
    ('只输出一个汉字：黑、白中，与黑色相对的是？', '白'),
    ('只输出一个大写英文字母：A 后面紧接着是什么？', 'B'),
    ('只输出一个汉字：春夏秋冬中，冬天之前是哪个季节？', '秋'),
    ('只输出一个数字：八减二等于多少？', '6'),
    ('只输出一个汉字：东西南北中，与东相反的是？', '西'),
    ('只输出一个汉字：大小中，大的反义词是？', '小'),
    ('只输出一个大写英文字母：D 前面紧挨着是什么？', 'C'),
)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if {q for q, _ in CASES} & {q for q, _ in trainer.CONSTRAINT_EXAMPLES}:
        raise ValueError('Exact training question overlap')
    info = json.loads((trainer.OUTPUT / 'result.json').read_text(encoding='utf-8'))
    for i in (6, 12):
        if (sha(trainer.PAIR / f'layer-{i}.pt') != info['source_pair_sha256'][str(i)]
                or sha(trainer.OUTPUT / f'layer-{i}.pt') != info['candidate_sha256'][str(i)]):
            raise ValueError('Pair provenance mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed().eval()
    originals = {i: base.model.layers[i] for i in (6, 12)}
    rows = {}
    for mode in ('raw', 'parent', 'candidate'):
        for i in (6, 12):
            base.model.layers[i] = (originals[i] if mode == 'raw' else
                _load_candidate((trainer.PAIR if mode == 'parent' else trainer.OUTPUT) / f'layer-{i}.pt', i, base).eval())
        if any(p.is_meta for p in base.parameters()) or any(b.is_meta for b in base.buffers()):
            raise ValueError('Unmaterialized layer')
        outputs = []
        for question, expected in CASES:
            prefix = closed_think_inputs(tokenizer, question, 'cuda')
            if prefix['input_ids'].shape[1] + 32 > 128:
                raise ValueError('Position limit exceeded')
            with torch.inference_mode():
                ids = base.generate(**prefix, max_new_tokens=32, do_sample=False, use_cache=False,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            answer = tokenizer.decode(ids[0, prefix['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            outputs.append({'question': question, 'expected': expected, 'answer': answer, 'exact': answer == expected})
            print(json.dumps({'mode': mode, 'index': len(outputs), 'exact': outputs[-1]['exact']}), flush=True)
        rows[mode] = outputs
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base changed')
    summary = {mode: {'cases': len(items), 'strict_exact': sum(r['exact'] for r in items)}
               for mode, items in rows.items()}
    report = {'status': 'authored_strict_format_development_eval', 'summary': summary, 'rows': rows,
              'base_sha256': WEIGHT_SHA, 'candidate_sha256': info['candidate_sha256'],
              'source_pair_sha256': info['source_pair_sha256'], 'seconds': time.perf_counter() - started,
              'limitations': ['Twelve authored deterministic short-output tasks; not natural conversation.',
                              'Exact question strings differ from examples, but format and themes overlap.',
                              'Strict scoring rejects punctuation or explanations even if content is correct.',
                              'No factual knowledge, memory, user identity or broad safety claims.',
                              'Research loader, closed reasoning tag, greedy decoding, 32-token cap.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
