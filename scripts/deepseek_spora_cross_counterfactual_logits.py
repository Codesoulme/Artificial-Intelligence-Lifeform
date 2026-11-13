"""Probe fixed-prefix token evidence under counterfactual memory and ablation."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_bridge_reload_audit as audit

torch, p = audit.torch, audit.p
audit.SOURCE = ROOT / 'runs/continuous-reader-cross-attention-pilot-20261113'
audit.OUT = ROOT / 'runs/deepseek-spora-cross-counterfactual-logits-v2-20261113'
audit.EXTRA_WORDS = ('桑葚', '枇杷')
audit.EXTRA_KEYS = (3,)
audit.EXTRA_GROUP_PREFIX = 'unseen'
audit.EXTRA_ONLY = True
audit.VERIFY_CODEC_READS = True
audit.PARITY_VALUE_INDICES = (0, 1, 3, 9, 16, 17)
PAIRS = (('苹果', '香蕉'), ('桑葚', '枇杷'))


def observe(*, decoder, tokenizer, spec, cross_modules, group, name, expected, question):
    pair = next((pair for pair in PAIRS if expected in pair), None)
    if pair is None:
        return
    if len(cross_modules) != 1:
        raise ValueError('Expected one memory cross-attention layer')
    prompt = p.closed_think_inputs(tokenizer, question, 'cuda')
    suffix = f'{name}喜欢的水果是'
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    ids = []
    for value in pair:
        full = tokenizer.encode(suffix + value, add_special_tokens=False)
        if full[:len(suffix_ids)] != suffix_ids or len(full) <= len(suffix_ids):
            raise ValueError('Tokenizer merges across the fixed answer boundary')
        ids.append(full[len(suffix_ids)])
    if ids[0] == ids[1]:
        raise ValueError('Pair must have distinct first fact tokens')
    prefix = torch.cat((prompt['input_ids'], torch.tensor([suffix_ids], device='cuda')), dim=1)
    scores = {}
    module = cross_modules[0]
    previous = module.enabled
    try:
        for enabled in (True, False):
            module.enabled = enabled
            with torch.inference_mode():
                kwargs = {**spec, 'use_cache': False}
                logits = decoder(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                                 **kwargs).logits[0, -1].float()
                log_probs = logits.log_softmax(-1)
                scores['on' if enabled else 'off'] = {
                    'logits': [float(logits[i]) for i in ids],
                    'log_probabilities': [float(log_probs[i]) for i in ids],
                    'ranks': [int((logits > logits[i]).sum()) + 1 for i in ids],
                    'margin_first_minus_second': float(logits[ids[0]] - logits[ids[1]]),
                    'top_token': int(logits.argmax()),
                }
    finally:
        module.enabled = previous
    row = {'group': group, 'name': name, 'question': question, 'memory_fact': expected,
           'prefix': suffix, 'pair': pair, 'fact_first_token_ids': ids, 'scores': scores}
    with (audit.OUT / 'logit_rows.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps({'scored': expected, 'group': group}), flush=True)


audit.PROBE_OBSERVER = observe
if __name__ == '__main__':
    audit.main()
