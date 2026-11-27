"""Isolated relation-gate experiment; no production state is read or written."""
import json, random, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_learned_kernel_probe_20261117 import LearnedKernel
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from spora.research_memory_value_transform import MemoryValueTransform
from spora.research_response_token_loss import response_token_loss

SOURCE = ROOT / 'runs/deepseek-kernel-full-answer-train-20261117'
OUT = ROOT / 'runs/research/relation-gate-20261127-v1'
SEED, STEPS, LR = 112727, 64, 0.001
REL = {'drink': ('最喜欢的饮品', ('豆浆', '咖啡', '牛奶', '可乐'), ('绿茶', '果汁')),
       'fruit': ('最喜欢的水果', ('苹果', '香蕉', '西瓜', '葡萄'), ('芒果', '桃子')),
       'job': ('职业', (), ('医生', '老师'))}
NAMES, HELD = ('小林', '小周'), '小赵'
NORMAL_TRAIN = ('请计算一加二。', '请用一句话解释什么是雨。')
NORMAL_HELD = ('请计算三加四。', '请说一句鼓励人的话。')


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(256, 32), nn.Tanh(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, q, keys):
        features = torch.cat((F.layer_norm(q, (128,)),
                              F.layer_norm(keys.mean(0), (128,)).expand_as(q)), -1)
        return 2 * torch.sigmoid(self.net(features))


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    def save(name, data):
        tmp = OUT / (name + '.tmp')
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        tmp.replace(OUT / name)
    try:
        torch.manual_seed(SEED); random.seed(SEED); torch.set_num_threads(2)
        torch.cuda.reset_peak_memory_stats()
        inputs = {'base': MODEL / 'model.safetensors', 'candidate': SOURCE / 'candidate.pt',
                  'source_report': SOURCE / 'result.json'}
        hashes = {k: sha(v) for k, v in inputs.items()}
        report = json.loads(inputs['source_report'].read_text(encoding='utf8'))
        if hashes['base'] != WEIGHT_SHA or hashes['candidate'] != report['candidate_sha256']:
            raise ValueError('Protected source hash mismatch')
        ckpt = torch.load(inputs['candidate'], map_location='cpu', weights_only=True)
        if ckpt['base_sha256'] != WEIGHT_SHA or ckpt['layer'] != 5 or ckpt['head'] != 0:
            raise ValueError('Unexpected candidate')
        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        kernel = LearnedKernel().to('cuda').eval(); kernel.load_state_dict(ckpt['state']); kernel.requires_grad_(False)
        transform = MemoryValueTransform(128).to('cuda').eval(); transform.load_state_dict(ckpt['value_transform']); transform.requires_grad_(False)
        gate = Gate().to('cuda'); attn = model.model.layers[5].self_attn
        captured, active = {}, {'kv': None, 'positions': None, 'legacy': False}
        def record(name):
            return lambda mod, args, output: captured.__setitem__(name, output.detach())
        def inject(mod, args):
            if active['kv'] is None: return None
            hidden = args[0]; n = hidden.shape[1]
            raw = captured['q'].view(1, n, 12, 128).transpose(1, 2)
            cos, sin = model.model.rotary_emb(torch.zeros(1, n, 128, device='cuda', dtype=raw.dtype), active['positions'])
            rotated, _ = apply_rotary_pos_emb(raw, raw, cos, sin)
            query = rotated[0, :6].float().reshape(-1, 128); keys, values = active['kv']
            qf = F.softplus(kernel.qmap(F.layer_norm(query, (128,)))) + 1e-5
            kf = F.softplus(kernel.kmap(F.layer_norm(keys, (128,)))) + 1e-5
            pred = transform(qf @ (kf.T @ values) / (qf @ kf.sum(0)).unsqueeze(-1))
            if not active['legacy']: pred = pred * gate(query, keys)
            delta = pred.reshape(6, n, 128).permute(1, 0, 2).reshape(1, n, 768)
            return (torch.cat((hidden[:, :, :768] + delta.to(hidden.dtype), hidden[:, :, 768:]), -1),) + args[1:]
        hooks = [getattr(attn, name + '_proj').register_forward_hook(record(name)) for name in ('q', 'k', 'v')]
        hooks.append(attn.o_proj.register_forward_pre_hook(inject))
        states = {}
        def subject(name, rel): return name + ('的' if rel == 'job' else '') + REL[rel][0]
        def fact(name, rel, value): return f'{subject(name, rel)}是{value}。'
        def question(name, rel): return f'{subject(name, rel)}是什么？请简洁回答。'
        def prompt(q): return closed_think_inputs(tok, q, 'cuda')['input_ids']
        all_pairs = [(n, r, v) for n in NAMES + (HELD,) for r, (_, train, held) in REL.items() for v in train + held]
        with torch.no_grad():
            for pair in all_pairs:
                ids = tok(fact(*pair), return_tensors='pt', add_special_tokens=False).input_ids.to('cuda')
                active['kv'] = None; model.model(input_ids=ids, use_cache=False); n = ids.shape[1]
                k = captured['k'].view(1, n, 2, 128).transpose(1, 2)
                v = captured['v'].view(1, n, 2, 128).transpose(1, 2)
                cos, sin = model.model.rotary_emb(torch.zeros(1, n, 128, device='cuda', dtype=k.dtype),
                                                torch.arange(n, device='cuda').unsqueeze(0))
                _, k = apply_rotary_pos_emb(k, k, cos, sin)
                states[pair] = (k[0, 0].float().detach(), v[0, 0].float().detach())
        offset = max(report['offset'], max(x[0].shape[0] for x in states.values()))
        def forward(ids, state=None, legacy=False):
            active.update(kv=state, legacy=legacy,
                          positions=torch.arange(offset, offset + ids.shape[1], device='cuda').unsqueeze(0))
            return model(input_ids=ids, position_ids=active['positions'], use_cache=False).logits.float()
        def response_loss(q, state, answer):
            p = prompt(q); answer_ids = tok.encode(answer, add_special_tokens=False) + [tok.eos_token_id]
            ids = torch.cat((p, torch.tensor([answer_ids], device='cuda')), 1)
            return response_token_loss(forward(ids, state), ids, p.shape[1])
        def gen(q, state, legacy=False):
            ids = prompt(q); start = ids.shape[1]
            with torch.no_grad():
                for _ in range(24):
                    t = forward(ids, state, legacy)[:, -1].argmax(-1).reshape(1, 1)
                    ids = torch.cat((ids, t), 1)
                    if t.item() == tok.eos_token_id: break
            result = ids[0, start:].tolist()
            return {'text': tok.decode(result, skip_special_tokens=True).strip(),
                    'ended': bool(result and result[-1] == tok.eos_token_id), 'tokens': result}
        train = [(n, r, v) for n in NAMES for r in ('drink', 'fruit') for v in REL[r][1]]
        cases = [('train', ('小林', 'drink', '豆浆')), ('train', ('小周', 'fruit', '苹果')),
                 ('held_name', (HELD, 'drink', '咖啡')), ('held_name', (HELD, 'fruit', '香蕉')),
                 ('held_value', ('小林', 'drink', '绿茶')), ('held_value', ('小周', 'fruit', '芒果')),
                 ('held_relation', (HELD, 'job', '医生')), ('held_relation', ('小林', 'job', '老师'))]
        def evaluate(stage):
            rows = []
            for split, pair in cases:
                n, r, v = pair; q = question(n, r)
                wrong = (n, 'fruit' if r != 'fruit' else 'drink', '苹果' if r != 'fruit' else '豆浆')
                for condition, state in [('none', None), ('correct', states[pair]), ('wrong_relation', states[wrong])]:
                    result = gen(q, state)
                    rows.append({'split': split, 'pair': pair, 'condition': condition, 'question': q,
                                 'target_in_text': v in result['text'], 'wrong_value_in_text': wrong[2] in result['text'], **result})
                save('progress.json', {'stage': stage, 'rows': rows})
            for q in NORMAL_HELD:
                for condition, state in [('none', None), ('unrelated', states[('小林', 'drink', '豆浆')])]:
                    rows.append({'split': 'normal_held', 'condition': condition, 'question': q, **gen(q, state)})
                save('progress.json', {'stage': stage, 'rows': rows})
            return rows
        with torch.no_grad():
            ids = prompt(question('小林', 'drink')); st = states[('小林', 'drink', '豆浆')]
            legacy = forward(ids, st, True); gated = forward(ids, st, False)
            equivalence = {'max_abs_logits': float((legacy - gated).abs().max()),
                           'same_argmax': bool(torch.equal(legacy.argmax(-1), gated.argmax(-1)))}
        before = evaluate('before')
        opt = torch.optim.AdamW(gate.parameters(), lr=LR, weight_decay=0)
        rng = random.Random(SEED); order = train.copy(); trace = []
        for step in range(STEPS):
            if step % len(order) == 0: rng.shuffle(order)
            pair = order[step % len(order)]; n, r, v = pair
            other_r = 'fruit' if r == 'drink' else 'drink'
            wrong = (n, other_r, rng.choice(REL[other_r][1]))
            opt.zero_grad(set_to_none=True)
            correct_loss = response_loss(question(n, r), states[pair], fact(*pair))
            wrong_loss = response_loss(question(n, r), states[wrong], '我不知道。')
            normal_q = NORMAL_TRAIN[step % len(NORMAL_TRAIN)]; ids = prompt(normal_q)
            with torch.no_grad(): teacher = forward(ids, None)[:, -1].detach()
            student = forward(ids, states[pair])[:, -1]
            kl = F.kl_div(F.log_softmax(student, -1), F.softmax(teacher, -1), reduction='batchmean')
            loss = correct_loss + wrong_loss + 0.5 * kl
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward(); grad = float(torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0, error_if_nonfinite=True)); opt.step()
            row = {'step': step + 1, 'correct_loss': float(correct_loss.detach()), 'wrong_loss': float(wrong_loss.detach()),
                   'kl': float(kl.detach()), 'gradient_norm': grad}
            trace.append(row)
            if step in (0, 15, 31, 47, 63):
                save('training-progress.json', {'trace': trace}); print(json.dumps(row), flush=True)
        torch.save({'gate': {k: v.detach().cpu() for k, v in gate.state_dict().items()}, 'seed': SEED,
                    'parent_sha256': hashes['candidate'], 'base_sha256': WEIGHT_SHA, 'offset': offset}, OUT / 'gate.pt')
        after = evaluate('after')
        for hook in hooks: hook.remove()
        final_hashes = {k: sha(v) for k, v in inputs.items()}
        result = {'status': 'research_only', 'seed': SEED, 'steps': STEPS, 'lr': LR, 'offset': offset,
                  'equivalence': equivalence, 'trace': trace, 'before': before, 'after': after,
                  'input_hashes': hashes, 'input_hashes_after': final_hashes, 'gate_sha256': sha(OUT / 'gate.pt'),
                  'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'limitations': ['Synthetic facts and manually selected state; no automatic routing or persistent individual.',
                                  'Two training relations and small exploratory held groups; no independent final acceptance.',
                                  'String containment is only a screening metric; inspect full outputs.',
                                  'KL compares final prompt token distribution under forced unrelated state.']}
        save('result.json', result)
        print(json.dumps({'completed': True, 'equivalence': equivalence, 'seconds': result['seconds'],
                          'source_unchanged': hashes == final_hashes}, ensure_ascii=False), flush=True)
    except BaseException:
        (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf8')
        raise


if __name__ == '__main__': main()
