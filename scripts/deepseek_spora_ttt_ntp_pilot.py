"""Probe a TTT-NTP-style fast-weight graft on native DeepSeek MLPs."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_local_baseline as p
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.research_ttt_ntp import fresh_memory, install, sequential_ntp_write, use_memory


OUT = ROOT / 'runs/deepseek-spora-ttt-ntp-pilot-v2-20261114'
LAYERS = (6, 12)
ETAS = (0.1, 0.5, 1.0)
RIDGE = 1.0
FACTS = ('小林喜欢喝酸奶。', '阿宁喜欢喝豆浆。')
QUESTIONS = ('请只回答小林喜欢喝什么饮料。', '请只回答阿宁喜欢喝什么饮料。')
EXPECTED = ('酸奶', '豆浆')


def generate(model, tokenizer, text, memory=None):
    prompt = closed_think_inputs(tokenizer, text, 'cuda')
    kwargs = dict(**prompt, use_cache=True, max_new_tokens=24, do_sample=False,
                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    with torch.inference_mode():
        if memory is None:
            result = model.generate(**kwargs)
        else:
            with use_memory(memory):
                result = model.generate(**kwargs)
    tail = result[0, prompt['input_ids'].shape[1]:].tolist()
    return {'answer': tokenizer.decode(tail, skip_special_tokens=True).strip(),
            'token_ids': tail}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    source_sha = p.sha(p.MODEL / 'model.safetensors')
    if source_sha != p.WEIGHT_SHA:
        raise ValueError('Base model digest mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True,
                                                trust_remote_code=False)
    with tempfile.TemporaryDirectory() as directory:
        model = load_full_streamed().eval()
        wrappers = install(model, LAYERS)
        before_digest = p.sha(p.MODEL / 'model.safetensors')
        ordinary = ('你好，请简单介绍自己。', '请写一句鼓励学习的话。')
        ordinary_before = [generate(model, tokenizer, question) for question in ordinary]
        context = tokenizer(''.join(FACTS), return_tensors='pt', add_special_tokens=False)
        context = {key: value.to('cuda') for key, value in context.items()}
        captured = {}
        handles = []
        for layer_id in LAYERS:
            def capture(module, args, output, layer_id=layer_id):
                captured[layer_id] = args[0].detach().float()[0]
            handles.append(wrappers[layer_id].base.register_forward_hook(capture))
        with torch.inference_mode():
            result = model(**context, output_hidden_states=True, use_cache=False,
                           return_dict=True)
        for handle in handles:
            handle.remove()
        if set(captured) != set(LAYERS):
            raise ValueError('Did not capture every native MLP key stream')
        hidden = result.hidden_states
        targets = {}
        for layer_id in LAYERS:
            layer_hidden = hidden[layer_id + 1][0].detach().float()
            keys = captured[layer_id][:-1]
            targets[layer_id] = layer_hidden[1:]
            if keys.shape[0] != targets[layer_id].shape[0]:
                raise ValueError('Causal key/target alignment mismatch')
        keys = {layer_id: captured[layer_id][:-1] for layer_id in LAYERS}
        baseline = {question: generate(model, tokenizer, question) for question in QUESTIONS}
        rows = []
        scans = []
        probe = keys[LAYERS[0]][:1].reshape(1, 1, -1).to(
            dtype=wrappers[LAYERS[0]].base.weight.dtype)
        for eta in ETAS:
            memory = fresh_memory(wrappers)
            sequential_ntp_write(memory, keys, targets, eta=eta, ridge=RIDGE)
            with use_memory(memory):
                base_probe = wrappers[LAYERS[0]].base(probe)
                adapted_probe = wrappers[LAYERS[0]](probe)
            scan = {'eta': eta,
                    'fast_weight_norms': {str(layer_id): float(weight.norm())
                                          for layer_id, weight in memory.items()},
                    'probe_delta_norm': float((adapted_probe - base_probe).norm()),
                    'rows': []}
            for fact, question, expected in zip(FACTS, QUESTIONS, EXPECTED):
                with_memory = generate(model, tokenizer, question, memory)
                row = {'eta': eta, 'fact': fact, 'question': question, 'expected': expected,
                       'no_memory': baseline[question], 'memory': with_memory,
                       'no_memory_contains': expected in baseline[question]['answer'],
                       'memory_contains': expected in with_memory['answer'],
                       'tokens_changed': baseline[question]['token_ids'] != with_memory['token_ids']}
                scan['rows'].append(row)
                rows.append(row)
                print(json.dumps({'eta': eta, 'completed': len(rows), 'expected': expected,
                                  'memory_contains': row['memory_contains'],
                                  'tokens_changed': row['tokens_changed']},
                                 ensure_ascii=False), flush=True)
            scans.append(scan)
        ordinary_after = [generate(model, tokenizer, question) for question in ordinary]
        if ordinary_before != ordinary_after:
            raise ValueError('Inactive memory changed ordinary generation')
        if p.sha(p.MODEL / 'model.safetensors') != before_digest:
            raise ValueError('Base model file changed')
    OUT.mkdir()
    summary = {'etas': list(ETAS), 'rows': len(rows),
               'no_memory_contains': sum(r['no_memory_contains'] for r in rows),
               'memory_contains': sum(r['memory_contains'] for r in rows),
               'by_eta': {str(eta): {'memory_contains': sum(r['memory_contains'] for r in rows if r['eta'] == eta),
                                     'tokens_changed': sum(r['tokens_changed'] for r in rows if r['eta'] == eta)}
                          for eta in ETAS}}
    report = {
        'status': 'ttt_ntp_native_mlp_graft_pilot', 'summary': summary, 'rows': rows,
        'layers': list(LAYERS), 'etas': list(ETAS), 'ridge': RIDGE,
        'scans': scans,
        'base_sha256': source_sha, 'seconds': time.perf_counter() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'limitations': [
            'Two authored Chinese fact sentences and two direct questions only.',
            'The context is tokenized and the query/key write is program-selected.',
            'The fast weight is an experimental per-call state, not yet persistent or naturally routed.',
            'No training of the projection or NTP layer was performed; this is a viability probe.',
            'Contains checks only whether the expected word appears, not full dialogue quality.',
        ],
    }
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
