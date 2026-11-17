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

SOURCE = ROOT / 'runs/deepseek-kernel-full-answer-train-20261117'
OUT = ROOT / 'runs/deepseek-kernel-null-lr30-20261117'
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
    torch.manual_seed(51913)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    paths = {'base': MODEL / 'model.safetensors', 'candidate': SOURCE / 'candidate.pt',
             'report': SOURCE / 'result.json'}
    hashes = {name: sha(path) for name, path in paths.items()}
    report = json.loads(paths['report'].read_text(encoding='utf8'))
    if hashes['base'] != WEIGHT_SHA or hashes['candidate'] != report['candidate_sha256']:
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
        before = inspect([c for c in held if c['name'] == HELD_NAME])
        def paired_logits(case, kv):
            prefix = case['answer'].split(case['value'])
            if len(prefix) != 2:
                raise ValueError('Answer value boundary ambiguous')
            before = tok.encode(prefix[0], add_special_tokens=False)
            full = tok.encode(prefix[0] + case['value'], add_special_tokens=False)
            if full[:len(before)] != before or len(full) <= len(before):
                raise ValueError('Answer token boundary mismatch')
            prompt = closed_think_inputs(tok, case['question'], 'cuda')['input_ids']
            ids = torch.cat((prompt, torch.tensor([before], device='cuda')), 1)
            active['kv'] = kv
            active['positions'] = torch.arange(report['offset'], report['offset'] + ids.shape[1], device='cuda').unsqueeze(0)
            return model(input_ids=ids, position_ids=active['positions'], use_cache=False).logits[:, -1].float(), full[len(before)]
        parameters = list(kernel.parameters()) + list(transform.parameters()) + list(null_residual.parameters())
        optimizer = torch.optim.AdamW([
            {'params': list(kernel.parameters()) + list(transform.parameters()), 'lr': 1e-4},
            {'params': list(null_residual.parameters()), 'lr': 3e-3},
        ])
        rng = random.Random(51912)
        order = list(train)
        null_questions = [c for c in train if c['value'] == RELATIONS[c['relation']]['train'][0]]
        null_rng = random.Random(51914)
        coverage = {f"{c['name']}:{c['relation']}": 0 for c in null_questions}
        trace = []
        for step in range(192):
            if step % len(order) == 0:
                rng.shuffle(order)
            case = order[step % len(order)]
            is_null = step % 3 == 0
            if is_null:
                index = (step // 3) % len(null_questions)
                if index == 0:
                    null_rng.shuffle(null_questions)
                case = null_questions[index]
                coverage[f"{case['name']}:{case['relation']}"] += 1
            prompt = closed_think_inputs(tok, case['question'], 'cuda')['input_ids']
            answer_ids = tok.encode('我不知道。' if is_null else case['answer'], add_special_tokens=False) + [tok.eos_token_id]
            ids = torch.cat((prompt, torch.tensor([answer_ids], device='cuda')), 1)
            active['kv'] = 'null' if is_null else state(case)
            active['positions'] = torch.arange(report['offset'], report['offset'] + ids.shape[1], device='cuda').unsqueeze(0)
            logits = model(input_ids=ids, position_ids=active['positions'], use_cache=False).logits.float()
            loss = response_token_loss(logits, ids, prompt.shape[1])
            if not is_null:
                alternative = next(c for c in train if c['name'] == case['name'] and c['relation'] == case['relation']
                                   and c['value'] != case['value'])
                logits_a, target_a = paired_logits(case, state(case))
                logits_b, target_b = paired_logits(alternative, state(alternative))
                loss = loss + pair_loss(logits_a, logits_b, target_a, target_b)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_null = None if null_residual.delta.grad is None else float(null_residual.delta.grad.norm())
            trace.append({'step': step + 1, 'mode': 'null' if is_null else 'known',
                          'loss': float(loss.detach()), 'null_grad_norm': grad_null})
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step in (0, 63, 127, 191):
                print(json.dumps({'step': step + 1, 'loss': float(loss.detach())}), flush=True)
        after = inspect(held)
        train_outputs = inspect(train)
        paired_audit = []
        for relation in RELATIONS:
            paired = [c for c in held if c['name'] == HELD_NAME and c['relation'] == relation
                      and c['value'] in RELATIONS[relation]['train']]
            if len(paired) != 2:
                raise ValueError('Missing held pair')
            outputs = [r for r in after if r['name'] == HELD_NAME and r['relation'] == relation
                       and r['value'] in RELATIONS[relation]['train']]
            paired_audit.append({'relation': relation,
                'both_exact': all(r['answer'] == next(c['answer'] for c in paired if c['value'] == r['value']) for r in outputs),
                'both_contain': all(r['target_in_output'] for r in outputs),
                'outputs_differ': outputs[0]['token_ids'] != outputs[1]['token_ids']})
        controls = []
        with torch.no_grad():
            for case in [c for c in train if c['value'] == RELATIONS[c['relation']]['train'][0]]:
                controls.append({'relation': case['relation'], 'name': case['name'], 'condition': 'train_null',
                                 **generate(case['question'], 'null')})
            for relation in RELATIONS:
                case = next(c for c in held if c['name'] == HELD_NAME and c['relation'] == relation and c['value'] in RELATIONS[relation]['train'])
                other = next(c for c in held if c['name'] == HELD_NAME and c['relation'] != relation and c['value'] in RELATIONS[c['relation']]['train'])
                for label, kv in (('off', None), ('null', 'null'), ('other_relation', state(other))):
                    controls.append({'relation': relation, 'condition': label, **generate(case['question'], kv)})
            for question in ('三加四等于多少？', '路口红灯亮了应该怎么做？'):
                for label, kv in (('off', None), ('null', 'null')):
                    controls.append({'relation': 'ordinary', 'condition': label, 'question': question, **generate(question, kv)})
    finally:
        for hook in hooks:
            hook.remove()
    if any(sha(path) != hashes[name] for name, path in paths.items()):
        raise ValueError('Protected input changed')
    OUT.mkdir()
    torch.save({'state': {k: v.detach().cpu() for k, v in kernel.state_dict().items()},
                'value_transform': {k: v.detach().cpu() for k, v in transform.state_dict().items()},
                'null_residual': {k: v.detach().cpu() for k, v in null_residual.state_dict().items()},
                'base_sha256': WEIGHT_SHA, 'parent_sha256': hashes['candidate'], 'layer': 5, 'head': 0}, OUT / 'candidate.pt')
    result = {'before': before, 'after': after, 'train': train_outputs, 'controls': controls, 'paired_audit': paired_audit,
              'null_coverage': coverage, 'trace': trace, 'seed': 51913, 'null_schedule_seed': 51914,
              'null_learning_rate': 3e-3, 'other_learning_rate': 1e-4,
              'candidate_sha256': sha(OUT / 'candidate.pt'), 'source_sha256': hashes, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Tiny authored examples, fixed phrases and one seed.',
                  'Manually selected temporary states, no autonomous parsing or natural memory writing.',
                  'Different memory architecture from existing persistent organism; no state saved or recovered.',
                  'String presence is not semantic correctness; no actual routing or identity checks.',
                  'Null activation is manually supplied and is not automatic missing-memory detection.',
                  'No no-memory gradient exists without explicit null residual activation.']}
    with (OUT / 'result.json').open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({'train': sum(r['target_in_output'] for r in train_outputs),
                      'held': sum(r['target_in_output'] for r in after), 'held_total': len(after)}), flush=True)


if __name__ == '__main__':
    main()
