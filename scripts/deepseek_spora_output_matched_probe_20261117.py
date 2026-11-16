"""Compare original and eight-value output heads on matched controlled states."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_addressed_read import closed_think_inputs
from spora.memory_output import MemoryOutputAdapter

SOURCE = ROOT / 'runs/deepseek-spora-memory-output-adapter-v2-20261111'
OUT = ROOT / 'runs/deepseek-spora-output-matched-probe-20261117.json'
VALUES = (0, 2, 4, 7)
PEOPLE = (1, 3)
ORDINARY = ('请只用一个数字回答：七减四等于几？', '请把“天气不错”改成疑问句。')


def inspect(chat, tokenizer, text, spec, logits, head):
    prompt = closed_think_inputs(tokenizer, text, 'cuda')
    target_device = next(chat.organism.parameters()).device
    with torch.inference_mode():
        raw = chat.decoder(**prompt, **spec).logits[:, -1, :].float()
        bias = torch.zeros_like(raw) if head is None else head(logits.to(target_device)).float()
        scored = raw + bias
        top_ids = scored[0].topk(5).indices.tolist()
        previous = chat.decoder.output_adapter
        chat.decoder.output_adapter = head
        try:
            generated = chat.decoder.generate(**prompt, **{**spec, 'use_cache': True},
                                               memory_output_logits=logits if head is not None else None,
                                               max_new_tokens=24, do_sample=False,
                                               pad_token_id=tokenizer.pad_token_id,
                                               eos_token_id=tokenizer.eos_token_id)
        finally:
            chat.decoder.output_adapter = previous
    tail = generated[0, prompt['input_ids'].shape[1]:].tolist()
    return {'text': tokenizer.decode(tail, skip_special_tokens=True).strip(),
            'token_ids': tail, 'ended': tokenizer.eos_token_id in tail,
            'top': [{'id': i, 'token': tokenizer.decode([i]), 'raw': float(raw[0, i]),
                     'bias': float(bias[0, i]), 'score': float(scored[0, i])} for i in top_ids]}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    adapter_path = SOURCE / 'adapter.pt'
    adapter_sha = p.sha(adapter_path)
    prior = json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))
    data = torch.load(adapter_path, map_location='cpu', weights_only=True)
    source_sha = p.sha(p.runtime.CHECKPOINT)
    if adapter_sha != prior['adapter_sha256'] or data['base_sha256'] != p.WEIGHT_SHA:
        raise ValueError('Adapter provenance mismatch')
    for index in (6, 12):
        if data['candidate_sha256'][str(index)] != p.sha(ROOT / 'runs/deepseek-spora-diverse-memory-replay-20261111' / f'layer-{index}.pt'):
            raise ValueError('Adapter replacement layer provenance mismatch')
    if p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Base provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    rows, ordinary = [], []
    with tempfile.TemporaryDirectory(prefix='spora-output-matched-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        source_digest = p.frozen_digest(chat.organism, chat.state)
        old = chat.decoder.output_adapter
        v2 = MemoryOutputAdapter(**data['config']).to('cuda').eval()
        v2.load_state_dict(data['state'], strict=True)
        device = next(chat.organism.parameters()).device
        for person in PEOPLE:
            key, name, _ = p.ENROLLMENTS[person]
            for value in VALUES:
                state = p.fresh_controlled_state(chat.organism)
                state, decision = p.observe_sourced(chat.organism, state, key, value, 'direct', 1.0)
                if not decision['current_write']:
                    raise ValueError('Controlled write rejected')
                before = p.frozen_digest(chat.organism, state)
                keys = torch.tensor([key], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = p.query(chat.organism, state, keys, torch.zeros_like(keys), torch.zeros_like(keys))
                spec = p._memory_args(chat.organism, state, {'key': key, 'intent': 'current', 'event_time': 0})
                question = f'依照已保存的饮品偏好，{name}偏爱哪一种？只写饮品名称。'
                variants = {label: inspect(chat, tokenizer, question, spec, logits, head)
                            for label, head in (('no_bias', None), ('original', old), ('v2', v2))}
                rows.append({'person': person, 'key': key, 'value': value, 'expected': p.WORDS[value],
                             'native_top': int(logits.argmax(-1)[0]), 'question': question, 'variants': variants})
                if p.frozen_digest(chat.organism, state) != before:
                    raise ValueError('Controlled memory mutated during generation')
                print(json.dumps({'person': person, 'value': value, 'complete': True}), flush=True)
        # Explicitly force the last case's memory into unrelated questions; this is not routing error frequency.
        for question in ORDINARY:
            prompt = closed_think_inputs(tokenizer, question, 'cuda')
            with torch.inference_mode():
                ids = chat.decoder.generate(**prompt, use_cache=True, max_new_tokens=24, do_sample=False,
                                            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            plain_ids = ids[0, prompt['input_ids'].shape[1]:].tolist()
            variants = {label: inspect(chat, tokenizer, question, spec, logits, head)
                        for label, head in (('no_bias', None), ('original', old), ('v2', v2))}
            ordinary.append({'question': question, 'memory_off': {'token_ids': plain_ids,
                              'text': tokenizer.decode(plain_ids, skip_special_tokens=True).strip()},
                             'forced_memory': variants})
        if p.frozen_digest(chat.organism, chat.state) != source_digest:
            raise ValueError('Source individual changed')
    if p.sha(p.runtime.CHECKPOINT) != source_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA or p.sha(adapter_path) != adapter_sha:
        raise ValueError('Protected input changed')
    report = {'rows': rows, 'ordinary': ordinary, 'adapter_sha256': adapter_sha,
              'checkpoint_sha256': source_sha, 'base_sha256': p.WEIGHT_SHA,
              'seconds': time.perf_counter()-start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Eight synthetic facts using two known people and four finite value classes already seen by v2.',
                              'One newly authored prompt is not proven disjoint from previous experiments.',
                              'Memory addresses and values are supplied by the test; no natural routing or persistence.',
                              'Forcing memory into ordinary questions is an intervention, not measured real-world routing frequency.',
                              'First-token scores and exact text are diagnostic, not semantic correctness certification.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'rows': len(rows), 'ordinary': len(ordinary)}), flush=True)


if __name__ == '__main__':
    main()
