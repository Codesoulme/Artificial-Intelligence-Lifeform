"""Controlled dialogue-derived values through the isolated learned memory kernel."""
import hashlib
import json
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
from scripts.deepseek_spora_dialogue_values_memory_pilot_20261117 import selected_values, QUESTIONS
from scripts.prepare_dulemon_evidence_pairs_20261117 import OUT as DATA
from spora.research_memory_value_transform import MemoryValueTransform

SOURCE = ROOT / 'runs/deepseek-kernel-full-answer-train-20261117'
OUT = ROOT / 'runs/deepseek-kernel-natural-values-probe-20261117.json'
FACTS = ('小赵最喜欢的运动是{}。', '小赵最喜欢的酒是{}。', '小赵的职业是{}。')


def build_facts(values):
    if len(values) != len(FACTS) or any(not isinstance(v, str) or not v for v in values):
        raise ValueError('Expected one nonempty value per relation')
    return [form.format(value) for form, value in zip(FACTS, values)]


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    paths = {'base': MODEL / 'model.safetensors', 'candidate': SOURCE / 'candidate.pt',
             'report': SOURCE / 'result.json', 'dialogue_values': DATA}
    hashes = {name: sha(path) for name, path in paths.items()}
    report = json.loads(paths['report'].read_text(encoding='utf8'))
    if hashes['base'] != WEIGHT_SHA or hashes['candidate'] != report['candidate_sha256']:
        raise ValueError('Protected input mismatch')
    payload = torch.load(paths['candidate'], map_location='cpu', weights_only=True)
    if payload['base_sha256'] != WEIGHT_SHA or payload['layer'] != 5 or payload['head'] != 0:
        raise ValueError('Candidate mismatch')
    values = selected_values(json.loads(DATA.read_text(encoding='utf8')))
    facts = build_facts(values)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    kernel = LearnedKernel().to('cuda').eval()
    kernel.load_state_dict(payload['state'], strict=True)
    transform = MemoryValueTransform(128).to('cuda').eval()
    transform.load_state_dict(payload['value_transform'], strict=True)
    attn = model.model.layers[5].self_attn
    captured, active = {}, {'kv': None, 'positions': None, 'calls': 0}

    def record(name):
        return lambda mod, args, out: captured.__setitem__(name, out.detach())

    def inject(mod, args):
        if active['kv'] is None:
            return None
        hidden = args[0]
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
        ids = closed_think_inputs(tok, question + '请简洁回答。', 'cuda')['input_ids']
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
        with torch.inference_mode():
            states = [make_state(fact) for fact in facts]
            rows = []
            for i, question in enumerate(QUESTIONS):
                for label, state in (('off', None), ('own', states[i]), ('other', states[(i + 1) % len(states)])):
                    result = generate(question, state)
                    rows.append({'question_index': i, 'condition': label,
                        'target_value_in_output': values[i] in result['answer'], **result})
                    print(json.dumps({'question': i, 'condition': label}), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    if any(sha(path) != hashes[name] for name, path in paths.items()):
        raise ValueError('Protected input changed')
    result = {'rows': rows, 'source_sha256': hashes, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Three dialogue-derived values are from separate previously examined sessions.',
                  'Artificial name, relation text, and manually selected temporary states, no autonomous parsing.',
                  'Different memory architecture from existing persistent organism; no state saved or recovered.',
                  'String presence is not semantic correctness; other value from another relation is forced.',
                  'Outputs and dialogue-derived values remain on local disk.']}
    with OUT.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({'rows': len(rows), 'own_contains_target': sum(r['target_value_in_output'] for r in rows if r['condition'] == 'own')}), flush=True)


if __name__ == '__main__':
    main()
