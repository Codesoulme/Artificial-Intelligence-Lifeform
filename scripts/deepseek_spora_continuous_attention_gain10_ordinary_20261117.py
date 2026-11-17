"""Compare active irrelevant research memory against no memory on chat prompts."""
import json
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from scripts import deepseek_spora_bridge_reload_audit as reload
from scripts.deepseek_spora_continuous_attention_gain10_20261117 import pilot as s

p = s.p
SOURCE = ROOT / 'runs/deepseek-spora-continuous-attention-gain10-64-20261117'
OUT = ROOT / 'runs/deepseek-spora-continuous-attention-gain10-ordinary-20261117.json'
QUESTIONS = ('你好，请简单介绍你自己。', '请计算三加四等于多少。', '春天下雨时出门应注意什么？',
             '请用一句话解释什么是地图。', '用一句话鼓励一个学习新知识的人。', '路口红灯亮了应该怎么做？')


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = {'candidate': SOURCE/'candidate.pt', 'report': SOURCE/'result.json',
             'source': p.runtime.CHECKPOINT, 'base': p.MODEL/'model.safetensors'}
    before = {k: p.sha(v) for k, v in paths.items()}
    payload = torch.load(paths['candidate'], map_location='cpu', weights_only=True)
    report = json.loads(paths['report'].read_text(encoding='utf-8'))
    if (before['base'] != p.WEIGHT_SHA or report['candidate_sha256'] != before['candidate']
            or payload['source_sha256'] != before['source'] or payload['base_sha256'] != before['base']
            or payload['experiment_config'].get('cross_attention_representation') != 'continuous'
            or payload['experiment_config'].get('cross_attention_output_gain') != 10.0):
        raise ValueError('Candidate provenance mismatch')
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tok = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    with tempfile.TemporaryDirectory(prefix='spora-gain10-ordinary-') as folder, ExitStack() as stack:
        p.runtime.WORKING_CHECKPOINT = Path(folder)/'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        decoder, model = chat.decoder.eval(), chat.organism.eval()
        original = p.frozen_digest(model, chat.state)
        device = next(model.parameters()).device
        embedding = chat.base.model.embed_tokens.weight.detach()
        codec = s.ContinuousTextCodec(embedding.cpu(), memory_width=model.config.memory_width,
                                     slot_count=5, seed=29071, eos_token_id=tok.eos_token_id).to(device)
        codec.load_state_dict(payload['codec_state'], strict=True)
        bridge = s.TokenCodeDecoder(codec.slot_width, embedding.shape[1]).to(device)
        bridge.slot_count = 5
        bridge.load_state_dict(payload['bridge'], strict=True)
        decoder.reader.load_state_dict(payload['reader'], strict=True)
        for site in decoder.reader.native_sites:
            stack.enter_context(patch.object(site, 'slow_signal',
                                            side_effect=lambda x, query: torch.zeros_like(query)))
        for old in payload['adapters']:
            attn = decoder.base.model.layers[old['layer']].self_attn
            layer = s.MemorySemanticGatedLowRankLinear(getattr(attn, old['projection']),
                decoder._memory_context, model.config.memory_width, bridge, embedding.shape[1], 4)
            with torch.no_grad():
                for name in ('a', 'b', 'condition'):
                    getattr(layer, name).copy_(old[name])
            setattr(attn, old['projection'], layer)
        old = payload['direct_adapters'][0]
        attn = decoder.base.model.layers[20].self_attn
        direct = s.MemoryResidualOutput(attn.o_proj, decoder._memory_context, model.config.memory_width,
                                        bridge, embedding.shape[1], 4)
        direct.to_hidden.load_state_dict(old['to_hidden'], strict=True)
        with torch.no_grad():
            direct.gate.copy_(old['gate'])
        attn.o_proj = direct
        old_final = payload['final_adapters'][0]
        final = s.MemoryFinalHiddenResidual(decoder.base.model.norm, decoder._memory_context,
                                           model.config.memory_width, bridge, embedding.shape[1], 4)
        final.to_hidden.load_state_dict(old_final['to_hidden'], strict=True)
        with torch.no_grad():
            final.gate.copy_(old_final['gate'])
        decoder.base.model.norm = final
        old_cross = payload['cross_adapters'][0]
        attention = decoder.base.model.layers[old_cross['layer']].self_attn
        cross = reload.restore_cross_attention(old_cross, attention, decoder, model, codec, embedding,
                                                payload['experiment_config'])
        attention.o_proj = cross
        decoder.requires_grad_(False)
        identities = ('小赵',)
        values = ('奶昔', '淡粉色', '练瑜伽')
        states = []
        with torch.no_grad():
            for value in values:
                temp_fast, temp_velocity = model.initial_state(1)
                for site in temp_fast:
                    site[0] = torch.zeros_like(site[0])
                ids = [tok.encode(identities[0], add_special_tokens=False)[-1]] + tok.encode(value, add_special_tokens=False)
                encoded = codec.encode(torch.tensor(ids, device=device))
                target = torch.zeros((1, model.config.memory_width), device=device)
                target[0, :encoded.numel()] = encoded
                for i, site in enumerate(model.memories):
                    temp_fast[i], temp_velocity[i] = site.write_temporal(
                        temp_fast[i], temp_velocity[i], site.current_basis[3:4].to(device), target,
                        torch.ones(1, device=device), torch.zeros(1, dtype=torch.long, device=device), False)
                snapshot = p.fresh_controlled_state(model)
                snapshot.current = [x[0].detach() for x in temp_fast]
                states.append(snapshot)
        rows = []
        with torch.inference_mode():
            for question in QUESTIONS:
                prompt = p.closed_think_inputs(tok, question, 'cuda')
                variants = [('off', {})] + [(f'memory_{i}', p._memory_args(model, st,
                    {'key': 3, 'intent': 'current', 'event_time': 0})) for i, st in enumerate(states)]
                results = []
                for name, spec in variants:
                    generated = decoder.generate(**prompt, **{**spec, 'use_cache': True},
                        max_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id)
                    ids = generated[0, prompt['input_ids'].shape[1]:].tolist()
                    results.append({'condition': name, 'token_ids': ids,
                                    'answer': tok.decode(ids, skip_special_tokens=True).strip()})
                rows.append({'question': question, 'variants': results,
                             'changed_count': sum(x['token_ids'] != results[0]['token_ids'] for x in results[1:])})
                print(json.dumps({'completed': len(rows), 'changed': rows[-1]['changed_count']}), flush=True)
        if p.frozen_digest(model, chat.state) != original:
            raise ValueError('Original individual changed')
    if any(p.sha(v) != before[k] for k, v in paths.items()):
        raise ValueError('Protected source changed')
    result = {'rows': rows, 'forced_memory_conditions': len(QUESTIONS)*len(values),
              'changed_conditions': sum(x['changed_count'] for x in rows),
              'source_sha256': before, 'seconds': time.perf_counter()-started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Forced activation and explicit addresses, not an actual learned routing gate.',
                              'Six authored questions and three artificial memory values; no quality grading.',
                              'Any output change is a difference, not automatically an error.',
                              'No natural write, identity inference, long-term storage or deployment.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'changed_conditions': result['changed_conditions'], 'total': result['forced_memory_conditions']}))


if __name__ == '__main__':
    main()
