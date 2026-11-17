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
from spora.research_memory_path_ablation import ablate_memory_paths

p = s.p
SOURCE = ROOT / 'runs/deepseek-spora-continuous-attention-gain10-64-20261117'
OUT = ROOT / 'runs/deepseek-spora-continuous-attention-gain10-ordinary-20261117.json'
QUESTIONS = ('你好，请简单介绍你自己。', '请计算三加四等于多少。', '春天下雨时出门应注意什么？',
             '请用一句话解释什么是地图。', '用一句话鼓励一个学习新知识的人。', '路口红灯亮了应该怎么做？')
CROSS_ABLATION = False
OTHER_PATH_ABLATION = False
MEMORY_VALUES = ('奶昔', '淡粉色', '练瑜伽')
INCLUDE_EMPTY_MEMORY = False
ADDITIONAL_INPUTS = {}


def variant_summary(rows):
    total = changed = isolated = empty_total = empty_changed = 0
    for row in rows:
        off = row['variants'][0]['token_ids']
        for variant in row['variants'][1:]:
            if variant['condition'] == 'empty':
                empty_total += 1
                empty_changed += int(variant['token_ids'] != off)
                continue
            total += 1
            changed += int(variant['token_ids'] != off)
            if 'cross_off_token_ids' in variant:
                isolated += int(variant['token_ids'] != variant['cross_off_token_ids'])
    return {'forced_memory_conditions': total, 'changed_conditions': changed,
            'cross_ablation_changed_conditions': isolated if CROSS_ABLATION else None,
            'empty_conditions': empty_total, 'empty_changed_conditions': empty_changed}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = {'candidate': SOURCE/'candidate.pt', 'report': SOURCE/'result.json',
             'source': p.runtime.CHECKPOINT, 'base': p.MODEL/'model.safetensors', **ADDITIONAL_INPUTS}
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
        low_rank_modules = []
        for old in payload['adapters']:
            attn = decoder.base.model.layers[old['layer']].self_attn
            layer = s.MemorySemanticGatedLowRankLinear(getattr(attn, old['projection']),
                decoder._memory_context, model.config.memory_width, bridge, embedding.shape[1], 4)
            with torch.no_grad():
                for name in ('a', 'b', 'condition'):
                    getattr(layer, name).copy_(old[name])
            setattr(attn, old['projection'], layer)
            low_rank_modules.append(layer)
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
        values = MEMORY_VALUES
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
                for i, site in enumerate(model.memories):
                    recovered = site.retrieve([temp_fast[i][0]], site.current_basis[3:4].to(device))[0]
                    if codec.decode(recovered)['token_ids'] != ids + [tok.eos_token_id]:
                        raise ValueError('Controlled memory read does not match written tokens')
                snapshot = p.fresh_controlled_state(model)
                snapshot.current = [x[0].detach() for x in temp_fast]
                states.append(snapshot)
            empty_state = p.fresh_controlled_state(model)
            empty_fast, _ = model.initial_state(1)
            empty_state.current = [torch.zeros_like(x[0]) for x in empty_fast]
        rows = []
        state_digests = [p.frozen_digest(model, state) for state in states]
        empty_digest = p.frozen_digest(model, empty_state)
        with torch.inference_mode():
            for question in QUESTIONS:
                prompt = p.closed_think_inputs(tok, question, 'cuda')
                variants = [('off', {})]
                if INCLUDE_EMPTY_MEMORY:
                    variants.append(('empty', p._memory_args(model, empty_state,
                        {'key': 3, 'intent': 'current', 'event_time': 0})))
                variants += [(f'memory_{i}', p._memory_args(model, st,
                    {'key': 3, 'intent': 'current', 'event_time': 0})) for i, st in enumerate(states)]
                results = []
                for name, spec in variants:
                    generated = decoder.generate(**prompt, **{**spec, 'use_cache': True},
                        max_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id)
                    ids = generated[0, prompt['input_ids'].shape[1]:].tolist()
                    results.append({'condition': name, 'token_ids': ids,
                                    'answer': tok.decode(ids, skip_special_tokens=True).strip()})
                    if CROSS_ABLATION and name != 'off':
                        cross.enabled = False
                        try:
                            ablated = decoder.generate(**prompt, **{**spec, 'use_cache': True},
                                max_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id,
                                eos_token_id=tok.eos_token_id)
                        finally:
                            cross.enabled = True
                        off_ids = ablated[0, prompt['input_ids'].shape[1]:].tolist()
                        results[-1]['cross_off_token_ids'] = off_ids
                        results[-1]['cross_off_answer'] = tok.decode(off_ids, skip_special_tokens=True).strip()
                    if OTHER_PATH_ABLATION and name != 'off':
                        path_outputs = {}
                        for path in ('native', 'low_rank', 'direct', 'final', 'all'):
                            disabled = ({'native', 'low_rank', 'direct', 'final', 'cross'}
                                        if path == 'all' else {path})
                            with ablate_memory_paths(decoder.reader, low_rank_modules, [direct], [final],
                                                     disabled, cross=[cross]):
                                ablated = decoder.generate(**prompt, **{**spec, 'use_cache': True},
                                    max_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id,
                                    eos_token_id=tok.eos_token_id)
                            path_ids = ablated[0, prompt['input_ids'].shape[1]:].tolist()
                            path_outputs[path] = {'token_ids': path_ids,
                                                  'answer': tok.decode(path_ids, skip_special_tokens=True).strip()}
                        if path_outputs['all']['token_ids'] != results[0]['token_ids']:
                            raise ValueError('All-path ablation differs from no-memory generation')
                        results[-1]['path_ablations'] = path_outputs
                rows.append({'question': question, 'variants': results,
                             'changed_count': sum(x['token_ids'] != results[0]['token_ids'] for x in results[1:])})
                print(json.dumps({'completed': len(rows), 'changed': rows[-1]['changed_count']}), flush=True)
        if p.frozen_digest(model, chat.state) != original:
            raise ValueError('Original individual changed')
        if [p.frozen_digest(model, state) for state in states] != state_digests:
            raise ValueError('Experimental memory changed')
        if p.frozen_digest(model, empty_state) != empty_digest:
            raise ValueError('Empty memory changed')
    if any(p.sha(v) != before[k] for k, v in paths.items()):
        raise ValueError('Protected source changed')
    result = {'rows': rows, **variant_summary(rows), 'cross_ablation': CROSS_ABLATION,
              'other_path_ablation': OTHER_PATH_ABLATION, 'memory_values': list(values),
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
