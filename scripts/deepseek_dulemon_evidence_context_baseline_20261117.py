"""Context-only baseline on development pairs, without accessing SPORA state.

This provides a reference for later memory-state experiments; it does not
demonstrate persistent model memory. Raw dialogue remains in local runs/.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from scripts import deepseek_local_baseline as base
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.prepare_dulemon_evidence_pairs_20261117 import OUT as DATA, MANIFEST

OUT = ROOT / 'runs/deepseek-dulemon-evidence-context-baseline-20261117.json'
INSTRUCTION = ('请只依据下面对话回答问题。若对话没有足够信息，请只回答“不知道”；'
               '若有明确答案，请尽量简短，不要解释。\n')


def classify(before_answer, after_answer, expected):
    """Literal indicators, not a semantic correctness or safety verdict."""
    if not all(isinstance(x, str) for x in (before_answer, after_answer, expected)) or not expected:
        raise ValueError('Expected nonempty text answers')
    return {'before_contains_unknown': '不知道' in before_answer,
            'before_leaks_expected': expected in before_answer,
            'after_contains_expected': expected in after_answer,
            'after_contains_unknown': '不知道' in after_answer}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = {'base': base.MODEL/'model.safetensors', 'data': DATA, 'manifest': MANIFEST}
    hashes = {k: base.sha(v) for k, v in paths.items()}
    data = json.loads(DATA.read_text(encoding='utf-8'))
    if (hashes['base'] != base.WEIGHT_SHA or hashes['manifest'] != data['manifest_sha256']
            or len(data['rows']) != 16):
        raise ValueError('Source mismatch')
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    # The existing replacement layers have a 64-position limit, too short for
    # these full dialogues. This is the original base, not the hybrid model.
    model = load_full_streamed().eval()
    model.requires_grad_(False)
    results = []
    with torch.inference_mode():
        for row in data['rows']:
            conversation = '\n'.join(f"{turn['speaker']}：{turn['text']}" for turn in row['context'])
            prompt = closed_think_inputs(tokenizer, INSTRUCTION + conversation + '\n问题：' + row['question'], 'cuda')
            generated = model.generate(**prompt, max_new_tokens=32, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            ids = generated[0, prompt['input_ids'].shape[1]:].tolist()
            results.append({'session': row['session'], 'condition': row['condition'],
                            'output': tokenizer.decode(ids, skip_special_tokens=True).strip(), 'token_ids': ids,
                            'terminated': bool(ids and ids[-1] == tokenizer.eos_token_id)})
            print(json.dumps({'completed': len(results), 'total': len(data['rows'])}), flush=True)
    if any(base.sha(v) != hashes[k] for k, v in paths.items()):
        raise ValueError('Protected input changed')
    metrics = []
    for i in range(0, len(results), 2):
        before, after = results[i:i+2]
        gold_before, gold_after = data['rows'][i:i+2]
        if (before['condition'], after['condition']) != ('before', 'after') or before['session'] != after['session']:
            raise ValueError('Invalid pair ordering')
        metrics.append({'session': before['session'], **classify(before['output'], after['output'], gold_after['answer'])})
    result = {'status': 'context_only_development_baseline', 'rows': results, 'pair_indicators': metrics,
              'summary': {name: sum(r[name] for r in metrics) for name in
                          ('before_contains_unknown', 'before_leaks_expected', 'after_contains_expected', 'after_contains_unknown')},
              'source_sha256': hashes, 'seconds': time.perf_counter()-start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Original DeepSeek base without SPORA replacement layers or internal memory.',
                              'Synthetic questions, eight manually selected development sessions, single annotator.',
                              'Context includes the actual evidence; no SPORA internal memory is accessed.',
                              'Literal matches are diagnostic only, not correctness or independent human evaluation.',
                              'Unknown before evidence is a tentative annotation and may miss implicit evidence.',
                              'Raw dialogue and outputs remain local due to source rights uncertainty.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
