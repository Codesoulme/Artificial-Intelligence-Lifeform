"""Read-only instruction-based direct-answer intent development probe.

The candidate is the same frozen base with two SPORA replacement layers; it
receives no stored facts. This is not an autonomous memory gate or deployment.
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
from scripts.deepseek_intent_fixed_challenge_20261117 import CASES, GATE
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, _load_candidate

OUT = ROOT / 'runs/deepseek-spora-intent-self-judge-pilot-20261117.json'
OLD = ROOT / 'runs/deepseek-spora-intent-fixed-challenge-20261117.json'
INSTRUCTION = ('判断用户是否明确要求直接回答某个人喜欢的饮品名称。'
               '询问颜色、价格、其他属性、要求删除或保密、翻译、普通聊天都回答否。'
               '即使不认识此人，只要要求其饮品名称也回答是。'
               '只输出一个字：是或否。\n用户输入：\n')


def parse_decision(output):
    """Accept a small closed set of complete affirmative/negative replies."""
    value = output.strip()
    if value.startswith('</think>'):
        value = value[len('</think>'):].strip()
    if value.endswith(('。', '.', '！', '!')):
        value = value[:-1].strip()
    return True if value in ('是', '是的') else False if value in ('否', '不是') else None


def summary(rows):
    result = {}
    for group in sorted({r['group'] for r in rows}):
        selected = [r for r in rows if r['group'] == group]
        result[group] = {'total': len(selected), 'positive': sum(r['read'] for r in selected),
                         'accepted': sum(r['decision'] is True for r in selected),
                         'refused': sum(r['decision'] is False for r in selected),
                         'invalid': sum(r['decision'] is None for r in selected),
                         'correct': sum(r['decision'] is not None and r['decision'] == r['read'] for r in selected)}
    return result


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = {'base': base.MODEL/'model.safetensors', 'layer6': PAIR/'layer-6.pt',
             'layer12': PAIR/'layer-12.pt', 'cases': CASES, 'previous': OLD}
    before = {k: base.sha(v) for k, v in paths.items()}
    prior = json.loads(OLD.read_text(encoding='utf-8'))
    cases = json.loads(CASES.read_text(encoding='utf-8'))
    if (before['base'] != base.WEIGHT_SHA or prior['source_sha256']['cases'] != before['cases']
            or len(cases) != 36 or len(prior['rows']) != len(cases)):
        raise ValueError('Source mismatch')
    if any(r['question'] != c['question'] or r['read'] != c['read']
           for r, c in zip(prior['rows'], cases)):
        raise ValueError('Question ordering or labels differ')
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed(skip_replaced=(6, 12)).eval()
    for i in (6, 12):
        model.model.layers[i] = _load_candidate(paths[f'layer{i}'], i, model).eval()
    model.requires_grad_(False)
    rows = []
    with torch.inference_mode():
        for case, old in zip(cases, prior['rows']):
            prompt = closed_think_inputs(tokenizer, INSTRUCTION + case['question'], 'cuda')
            output = model.generate(**prompt, max_new_tokens=4, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            tokens = output[0, prompt['input_ids'].shape[1]:].tolist()
            decoded = tokenizer.decode(tokens, skip_special_tokens=True)
            rows.append({**case, 'output': decoded, 'token_ids': tokens,
                         'decision': parse_decision(decoded), 'prior_accepted': old['accepted']})
            if len(rows) % 6 == 0:
                print(json.dumps({'completed': len(rows)}), flush=True)
    if any(base.sha(path) != before[k] for k, path in paths.items()):
        raise ValueError('Protected source changed')
    result = {'status': 'development_instruction_gate_not_deployed', 'summary': summary(rows),
              'rows': rows, 'source_sha256': before, 'seconds': time.perf_counter()-started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Synthetic challenge, reviewed and used previously; not an independent evaluation.',
                              'Same base model with replacement layers; no memory read or identity routing.',
                              'System instruction is external context, not an internal learned gate.',
                              'Output parsing rejects nonexact completions; no calibration or extra tokens.',
                              'Not tested against adversarial user text, long conversation or ordinary generation.',
                              'Comparison with prior probe is exploratory and the challenge informed this prompt.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
