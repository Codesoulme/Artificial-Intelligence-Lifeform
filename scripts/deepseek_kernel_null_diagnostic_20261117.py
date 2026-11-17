"""Train an isolated kernel on counterfactual values across three relations."""
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_learned_kernel_probe_20261117 import LearnedKernel
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from spora.research_response_token_loss import response_token_loss
from spora.research_memory_value_transform import MemoryValueTransform
from spora.research_first_token_pair_loss import pair_loss
from spora.research_null_memory_residual import NullMemoryResidual

SOURCE = ROOT / 'runs/deepseek-kernel-null-residual-20261117'
ORIGIN = ROOT / 'runs/deepseek-kernel-full-answer-train-20261117/result.json'
OUT = ROOT / 'runs/deepseek-kernel-null-diagnostic-20261117.json'
NAMES = ('小林', '小周')
HELD_NAME = '小赵'
RELATIONS = {
    'drink': {'train': ('咖啡', '牛奶'), 'held': '豆浆', 'fact': '{n}喜欢喝{v}。',
              'question': '{n}喜欢喝什么？请简洁回答。', 'answer': '{n}喜欢喝{v}。'},
    'sport': {'train': ('游泳', '跑步'), 'held': '篮球', 'fact': '{n}喜欢的运动是{v}。',
              'question': '{n}喜欢什么运动？请简洁回答。', 'answer': '{n}喜欢的运动是{v}。'},
    'job': {'train': ('医生', '厨师'), 'held': '教师', 'fact': '{n}的职业是{v}。',
            'question': '{n}从事什么工作？请简洁回答。', 'answer': '{n}的职业是{v}。'},
}


def build_cases():
    train, held = [], []
    for relation, spec in RELATIONS.items():
        for name in NAMES + (HELD_NAME,):
            for value in spec['train'] + (spec['held'],):
                case = {'name': name, 'relation': relation, 'value': value,
                        'fact': spec['fact'].format(n=name, v=value),
                        'question': spec['question'].format(n=name),
                        'answer': spec['answer'].format(n=name, v=value)}
                (train if name in NAMES and value in spec['train'] else held).append(case)
    return train, held


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    paths = {'base': MODEL / 'model.safetensors', 'candidate': SOURCE / 'candidate.pt',
             'report': SOURCE / 'result.json', 'origin': ORIGIN}
    hashes = {name: sha(path) for name, path in paths.items()}
    candidate_report = json.loads(paths['report'].read_text(encoding='utf8'))
    report = json.loads(paths['origin'].read_text(encoding='utf8'))
    if hashes['base'] != WEIGHT_SHA or hashes['candidate'] != candidate_report['candidate_sha256']:
        raise ValueError('Protected input mismatch')
    payload = torch.load(paths['candidate'], map_location='cpu', weights_only=True)
    if payload['base_sha256'] != WEIGHT_SHA or payload['layer'] != 5 or payload['head'] != 0:
        raise ValueError('Candidate mismatch')
    train, held = build_cases()
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    kernel = LearnedKernel().to('cuda')
    kernel.load_state_dict(payload['state'], strict=True)
    transform = MemoryValueTransform(128).to('cuda')
    transform.load_state_dict(payload['value_transform'], strict=True)
    null_residual = NullMemoryResidual(768).to('cuda')
    null_residual.load_state_dict(payload['null_residual'], strict=True)
    attn = model.model.layers[5].self_attn
    captured, active = {}, {'kv': None, 'positions': None, 'calls': 0}

    def record(name):
        return lambda mod, args, out: captured.__setitem__(name, out.detach())

    def inject(mod, args):
        if active['kv'] is None:
            return None
        hidden = args[0]
        if isinstance(active['kv'], str):
            if active['kv'] != 'null':
                raise ValueError('Invalid state')
            active['calls'] += 1
            return (null_residual(hidden),) + args[1:]
        n = hidden.shape[1]
        raw = captured['q'].view(1, n, 12, 128).transpose(1, 2)
        cos, sin = model.model.rotary_emb(torch.zeros(1, n, 128, device='cuda', dtype=raw.dtype), active['positions'])
        rotated, _ = apply_rotary_pos_emb(raw, raw, cos, sin)
        query = rotated[0, :6].float().reshape(-1, 128)
        keys, values = active['kv']
        qf = F.softplus(kernel.qmap(F.layer_norm(query, (128,)))) + 1e-5
        kf = F.softplus(kernel.kmap(F.layer_norm(keys, (128,)))) + 1e-5
        pred = transform(qf @ (kf.T @ values) / (qf @ kf.sum(0)).unsqueeze(-1))
        delta = pred.reshape(6, n, 128).permute(1, 0, 2).reshape(1, n, 768)
        active['calls'] += 1
        return (torch.cat((hidden[:, :, :768] + delta.to(hidden.dtype), hidden[:, :, 768:]), -1),) + args[1:]

    hooks = [getattr(attn, name + '_proj').register_forward_hook(record(name)) for name in ('q', 'k', 'v')]
    hooks.append(attn.o_proj.register_forward_pre_hook(inject))

    def make_state(fact):
        ids = tok(fact, return_tensors='pt', add_special_tokens=False).input_ids.to('cuda')
        active['kv'] = None
        model.model(input_ids=ids, use_cache=False)
        n = ids.shape[1]
        k = captured['k'].view(1, n, 2, 128).transpose(1, 2)
        v = captured['v'].view(1, n, 2, 128).transpose(1, 2)
        cos, sin = model.model.rotary_emb(torch.zeros(1, n, 128, device='cuda', dtype=k.dtype),
                                         torch.arange(n, device='cuda').unsqueeze(0))
        _, k = apply_rotary_pos_emb(k, k, cos, sin)
        return k[0, 0].float(), v[0, 0].float()

    def generate(question, state):
        ids = closed_think_inputs(tok, question, 'cuda')['input_ids']
        first = ids.shape[1]
        active['calls'] = 0
        for _ in range(24):
            active['kv'] = state
            active['positions'] = torch.arange(report['offset'], report['offset'] + ids.shape[1], device='cuda').unsqueeze(0)
            nxt = model(input_ids=ids, position_ids=active['positions'], use_cache=False).logits[:, -1].argmax(-1).reshape(1, 1)
            ids = torch.cat((ids, nxt), 1)
            if nxt.item() == tok.eos_token_id:
                break
        output_ids = ids[0, first:].tolist()
        return {'token_ids': output_ids, 'answer': tok.decode(output_ids, skip_special_tokens=True).strip(),
                'terminated': bool(output_ids and output_ids[-1] == tok.eos_token_id), 'calls': active['calls']}

    try:
        with torch.no_grad():
            states = {(c['name'], c['relation'], c['value']): make_state(c['fact']) for c in train + held}
        def state(case):
            return states[(case['name'], case['relation'], case['value'])]
        def inspect(cases):
            rows = []
            with torch.no_grad():
                for case in cases:
                    output = generate(case['question'], state(case))
                    rows.append({**{k: case[k] for k in ('name', 'relation', 'value')}, **output,
                                 'target_in_output': case['value'] in output['answer']})
            return rows
        diagnostics = []
        with torch.no_grad():
            for case in train + held:
                # One observation per question, not one per value.
                if case['value'] != RELATIONS[case['relation']]['train'][0]:
                    continue
                prompt = closed_think_inputs(tok, case['question'], 'cuda')['input_ids']
                answer_ids = tok.encode('我不知道。', add_special_tokens=False) + [tok.eos_token_id]
                ids = torch.cat((prompt, torch.tensor([answer_ids], device='cuda')), 1)
                variants = {}
                for label, kv in (('off', None), ('null', 'null')):
                    active['kv'] = kv
                    active['positions'] = torch.arange(report['offset'], report['offset'] + ids.shape[1], device='cuda').unsqueeze(0)
                    logits = model(input_ids=ids, position_ids=active['positions'], use_cache=False).logits.float()
                    loss = float(response_token_loss(logits, ids, prompt.shape[1]))
                    output = generate(case['question'], kv)
                    variants[label] = {'loss': loss, **output}
                diagnostics.append({'name': case['name'], 'relation': case['relation'], 'split': 'held' if case['name'] == HELD_NAME else 'train', 'variants': variants})
                print(json.dumps({'completed': len(diagnostics), 'total': 9}), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    if any(sha(path) != hashes[name] for name, path in paths.items()):
        raise ValueError('Protected input changed')
    result = {'rows': diagnostics, 'source_sha256': hashes, 'null_norm': float(null_residual.delta.detach().norm()),
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Teacher-forced loss is not free-generation accuracy.',
                              'Small artificial fixed questions; train and held differ by person only.',
                              'Null state is manually supplied; no memory presence detection or persistence.',
                              'No training, formal individual changes, or deployment.']}
    with OUT.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'rows': len(diagnostics), 'null_norm': result['null_norm']}), flush=True)


if __name__ == '__main__':
    main()
