"""Held-form free generation and forced-misroute diagnostics for the gate."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch import nn
from torch.nn import functional as F
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.deepseek_spora_online_lora_memory_pilot import OnlineDelta
from scripts.deepseek_spora_sentence_gate_pilot import SENTENCE, ADAPTER, OUT as GATE, checked
from scripts.deepseek_spora_sentence_lora_pilot import make_state
from spora.memory_output import MemoryOutputAdapter

OUTPUT = ROOT / 'runs/deepseek-spora-sentence-gate-freerun-20261114.json'
FACT_CASES = (
    (2, '我想确认你所记录的{name}的饮料喜好，请写成一句话。', (3, 7)),
    (3, '请完整回答：你所知道的{name}平时爱喝哪种饮品？', (2, 4)),
    (4, '从你的记录来看，{name}喜欢喝什么？用一句话说。', (1, 5)),
    (5, '关于{name}的饮品选择，请依据记忆写一句简短回答。', (4, 6)),
)
MISROUTES = (
    ('ordinary', '写一句鼓励朋友努力学习的话。'),
    ('ordinary', '请说明一天有多少小时。'),
    ('negation', '不要告诉我小雨喜欢喝什么饮料。'),
    ('wrong_relation', '小雨喜欢的饮品是什么颜色？'),
)


def generate(chat, question, args, bias, gate, *, policy, gain=1.0):
    prefix = closed_think_inputs(chat.tokenizer, question, 'cuda')
    ids = prefix['input_ids']
    fired_at = None
    for step in range(24):
        with torch.inference_mode():
            kwargs = {'use_cache': False} if policy == 'no_memory' else args
            output = chat.decoder(input_ids=ids, attention_mask=torch.ones_like(ids),
                                  output_hidden_states=True, return_dict=True, **kwargs)
            logits = output.logits[:, -1].float()
            if policy == 'gated':
                hidden = output.hidden_states[-1][:, -1].float().cpu()
                probability = float(gate(F.layer_norm(hidden, (hidden.shape[-1],))).sigmoid()[0, 0])
                if fired_at is None and probability >= 0.5:
                    logits = logits + gain * bias
                    fired_at = step
            token = logits.argmax(-1, keepdim=True)
        ids = torch.cat((ids, token), dim=1)
        if int(token[0, 0]) == chat.tokenizer.eos_token_id:
            break
    tail = ids[0, prefix['input_ids'].shape[1]:].tolist()
    return {'answer': chat.tokenizer.decode(tail, skip_special_tokens=True).strip(),
            'token_ids': tail, 'fired_at': fired_at, 'ended': chat.tokenizer.eos_token_id in tail}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    sentence, sentence_path, sentence_sha = checked(SENTENCE, 'adapter.pt', 'adapter_sha256')
    adapter_data, adapter_path, adapter_sha = checked(ADAPTER, 'adapter.pt', 'adapter_sha256')
    gate_data, gate_path, gate_sha = checked(GATE, 'gate.pt', 'gate_sha256')
    if (base_sha != p.WEIGHT_SHA or sentence['base_sha256'] != base_sha
            or sentence['checkpoint_sha256'] != checkpoint_sha or adapter_data['base_sha256'] != base_sha
            or gate_data['base_sha256'] != base_sha or gate_data['checkpoint_sha256'] != checkpoint_sha
            or gate_data['sentence_sha256'] != sentence_sha or gate_data['output_sha256'] != adapter_sha):
        raise ValueError('Candidate provenance mismatch')
    for layer in (6, 12):
        path = ROOT / 'runs/deepseek-spora-diverse-memory-replay-20261111' / f'layer-{layer}.pt'
        if adapter_data['candidate_sha256'][str(layer)] != p.sha(path):
            raise ValueError('Output adapter layer mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix='spora-gate-free-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        original_digest = p.frozen_digest(chat.organism, chat.state)
        for layer, weights in zip(sentence['layers'], sentence['adapters']):
            mlp = chat.base.model.layers[layer].mlp
            module = OnlineDelta(mlp.down_proj, sentence['rank'])
            loaded = module.load_state_dict(weights, strict=False)
            missing = {'base.weight'} | ({'base.bias'} if module.base.bias is not None else set())
            if set(loaded.missing_keys) != missing or loaded.unexpected_keys:
                raise ValueError('Sentence adapter mismatch')
            mlp.down_proj = module.eval()
        adapter = MemoryOutputAdapter(**adapter_data['config']).to('cuda').eval()
        adapter.load_state_dict(adapter_data['state'], strict=True)
        gate = nn.Linear(gate_data['state']['weight'].shape[1], 1).eval()
        gate.load_state_dict(gate_data['state'], strict=True)
        cases = [('fact', i, template.format(name=NAMES[i]), value)
                 for i, template, values in FACT_CASES for value in values]
        cases += [(group, 2, question, 7) for group, question in MISROUTES]
        for group, index, question, value in cases:
            state = make_state(chat, KEYS[index], value)
            state_digest = p.frozen_digest(chat.organism, state)
            args, evidence = p.runtime._prepare_memory_read(chat.organism, state, KEYS[index], 'current', None)
            with torch.inference_mode():
                bias = adapter(evidence).float()
            policies = (('no_memory', 'no_memory', 0), ('reader', 'reader', 0),
                        ('gate_1', 'gated', 1), ('gate_2', 'gated', 2))
            results = {}
            for label, policy, gain in policies:
                result = generate(chat, question, args, bias, gate, policy=policy, gain=gain)
                answer = result['answer']
                result['contains_target'] = p.WORDS[value] in answer
                result['other_values'] = [w for w in p.WORDS if w != p.WORDS[value] and w in answer]
                results[label] = result
            if p.frozen_digest(chat.organism, state) != state_digest:
                raise ValueError('Controlled state changed')
            row = {'group': group, 'question': question, 'expected': p.WORDS[value] if group == 'fact' else None,
                   'forced_evidence': p.WORDS[value], 'native_correct': int(evidence.argmax(-1)[0]) == value,
                   'results': results}
            rows.append(row)
            print(json.dumps({'done': len(rows), 'group': group,
                              'answers': {k: v['answer'] for k, v in results.items()}}, ensure_ascii=True), flush=True)
        if p.frozen_digest(chat.organism, chat.state) != original_digest:
            raise ValueError('Original state changed')
    for path, digest in ((p.MODEL / 'model.safetensors', base_sha), (p.runtime.CHECKPOINT, checkpoint_sha),
                         (sentence_path, sentence_sha), (adapter_path, adapter_sha), (gate_path, gate_sha)):
        if p.sha(path) != digest:
            raise ValueError('Protected file changed')
    summary = {}
    for group in sorted({r['group'] for r in rows}):
        subset = [r for r in rows if r['group'] == group]
        summary[group] = {'cases': len(subset)}
        for label in ('no_memory', 'reader', 'gate_1', 'gate_2'):
            summary[group][label] = {'contains_target': sum(r['results'][label]['contains_target'] for r in subset),
                                     'other_value_mentions': sum(bool(r['results'][label]['other_values']) for r in subset),
                                     'fired': sum(r['results'][label]['fired_at'] is not None for r in subset),
                                     'ended': sum(r['results'][label]['ended'] for r in subset)}
    report = {'rows': rows, 'summary': summary, 'base_sha256': base_sha, 'checkpoint_sha256': checkpoint_sha,
              'sentence_sha256': sentence_sha, 'adapter_sha256': adapter_sha, 'gate_sha256': gate_sha,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Authored prompts not used in this gate training but possible overlap with prior work.',
                              'Manually supplied numeric addresses, finite values, and controlled states bypass routing.',
                              'Non-fact examples deliberately force a memory misroute, not observed runtime routing.',
                              'Substring counts are not semantic correctness or instruction following.',
                              'Gain and gate were chosen using earlier development cases, not independently calibrated.',
                              'Uncached greedy decoding is an experimental implementation.']}
    with OUTPUT.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
