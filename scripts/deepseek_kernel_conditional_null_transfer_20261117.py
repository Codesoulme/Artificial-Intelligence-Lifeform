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
from spora.research_conditional_null_residual import ConditionalNullResidual

SOURCE = ROOT / 'runs/deepseek-kernel-conditional-null-20261117'
OUT = ROOT / 'runs/deepseek-kernel-conditional-null-transfer-20261117'
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
    report['offset'] = json.loads((ROOT / 'runs/deepseek-kernel-full-answer-train-20261117/result.json').read_text(encoding='utf8'))['offset']
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
    null_residual = ConditionalNullResidual(768, rank=16).to('cuda')
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

    questions = [
        ('private', '小陈昨天晚上吃了什么？'),
        ('private', '我的朋友小李住在哪里？'),
        ('private', '张老师养的猫叫什么名字？'),
        ('private', '小王下周准备去哪个城市？'),
        ('private', '我最喜欢什么颜色？'),
        ('private', '小刘上次考试考了多少分？'),
        ('general', '太阳从哪个方向升起？'),
        ('general', '一周有多少天？'),
        ('general', '水结冰通常需要多少摄氏度？'),
        ('general', '人通常用什么器官呼吸？'),
        ('general', '北京是哪个国家的首都？'),
        ('general', '五加六等于多少？'),
    ]
    try:
        rows = []
        with torch.no_grad():
            for category, question in questions:
                for label, kv in (('off', None), ('null', 'null')):
                    output = generate(question, kv)
                    rows.append({'category': category, 'question': question, 'condition': label, **output})
                print(json.dumps({'category': category, 'question': question}, ensure_ascii=False), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    if any(sha(path) != hashes[name] for name, path in paths.items()):
        raise ValueError('Protected input changed')
    OUT.mkdir()
    with (OUT / 'result.json').open('x', encoding='utf8') as f:
        json.dump({'rows': rows, 'source_sha256': hashes, 'seconds': time.perf_counter() - started,
                   'limitations': ['Authored small sample, no automatic routing.', 'No retrieved or persisted memory.',
                                   'Null state manually forced for general as well as private questions.']},
                  f, ensure_ascii=False, indent=2)
    print(json.dumps({'cases': len(questions), 'outputs': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
