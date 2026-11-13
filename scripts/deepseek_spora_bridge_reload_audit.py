"""Reload a fixed candidate and probe broader generation without training.

Legacy candidates lack codec metadata. This audit deliberately accepts only
the named five-slot, full-fact, linear-bridge experiment and checks token parity
against its saved generations before attempting any new development probes.
"""
import json
import tempfile
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from scripts import deepseek_spora_semantic_bridge_pilot as s

p = s.p
torch = p.torch
SOURCE = p.ROOT / 'runs/continuous-reader-all-facts-late-residual-strict-pilot-20261112'
OUT = p.ROOT / 'runs/continuous-reader-bridge-reload-audit-v1'
NAMES = ('小林', '小周', '小王', '小赵', '小吴', '小郑')
QUESTION = '请只说出{name}喜欢的水果名称。'
EXTRA_WORDS = ()
EXTRA_KEYS = (3,)
EXTRA_GROUP_PREFIX = 'unseen'
EXTRA_ONLY = False
VERIFY_CODEC_READS = False
PARITY_VALUE_INDICES = None
MIX_ABLATION = False
CROSS_ABLATION = False
PROBE_OBSERVER = None


def correct_answer(answer, name, value):
    # Accept only the requested bare fact or the supervised assertion, not a list.
    return answer in (value, f'{name}喜欢的水果是{value}。')


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir()
    started = time.perf_counter()
    torch.manual_seed(s.SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    protected = {path: p.sha(path) for path in
                 (SOURCE / 'candidate.pt', SOURCE / 'result.json',
                  p.runtime.CHECKPOINT, p.MODEL / 'model.safetensors')}
    payload = torch.load(SOURCE / 'candidate.pt', map_location='cpu', weights_only=True)
    original = json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))
    if (payload['source_sha256'] != protected[p.runtime.CHECKPOINT]
            or payload['base_sha256'] != p.WEIGHT_SHA
            or original['candidate_sha256'] != protected[SOURCE / 'candidate.pt']):
        raise ValueError('Candidate/source lineage mismatch')
    tok = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    rows = []
    with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
        p.runtime.WORKING_CHECKPOINT = Path(temp) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        decoder, model = chat.decoder.eval(), chat.organism.eval()
        original_digest = p.frozen_digest(model, chat.state)
        device = next(model.parameters()).device
        embedding = chat.base.model.embed_tokens.weight.detach()
        codec = s.ContinuousTextCodec(embedding.cpu(), memory_width=model.config.memory_width,
                                     slot_count=5, seed=29071, eos_token_id=tok.eos_token_id).to(device)
        bridge_class = payload.get('experiment_config', {}).get('bridge_class', 'TokenCodeDecoder')
        if bridge_class == 'TokenCodebookResidualDecoder':
            bridge = s.TokenCodebookResidualDecoder(codec, embedding).to(device)
        elif bridge_class == 'TokenCodeDecoder':
            bridge = s.TokenCodeDecoder(codec.slot_width, embedding.shape[1]).to(device)
        else:
            raise ValueError('Unsupported bridge format for reload audit')
        bridge.slot_count = 5
        bridge.load_state_dict(payload['bridge'], strict=True)

        def generate(question, spec):
            prompt = p.closed_think_inputs(tok, question, 'cuda')
            with torch.inference_mode():
                generated = decoder.generate(**prompt, **{**spec, 'use_cache': True},
                    max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id)
            tail = generated[0, prompt['input_ids'].shape[1]:].tolist()
            return tok.decode(tail, skip_special_tokens=True).strip(), tail

        ordinary = ('你好，请简单介绍自己。', '请写一句鼓励学习的话。', '一加一等于几？')
        ordinary_before = [generate(q, {}) for q in ordinary]
        decoder.reader.load_state_dict(payload['reader'], strict=True)
        for site in decoder.reader.native_sites:
            stack.enter_context(patch.object(site, 'slow_signal',
                                side_effect=lambda x, query: torch.zeros_like(query)))
        if [(x['layer'], x['projection']) for x in payload['adapters']] != [
                (18, 'q_proj'), (18, 'v_proj'), (20, 'q_proj'), (20, 'v_proj')]:
            raise ValueError('Unexpected adapter layout')
        for old in payload['adapters']:
            attention = decoder.base.model.layers[old['layer']].self_attn
            adapter = s.MemorySemanticGatedLowRankLinear(getattr(attention, old['projection']),
                decoder._memory_context, model.config.memory_width, bridge, embedding.shape[1], 4)
            with torch.no_grad():
                for name in ('a', 'b', 'condition'):
                    getattr(adapter, name).copy_(old[name])
            setattr(attention, old['projection'], adapter)
        if len(payload['direct_adapters']) != 1 or payload['direct_adapters'][0]['layer'] != 20:
            raise ValueError('Unexpected direct residual layout')
        old = payload['direct_adapters'][0]
        attention = decoder.base.model.layers[20].self_attn
        adapter = s.MemoryResidualOutput(attention.o_proj, decoder._memory_context,
            model.config.memory_width, bridge, embedding.shape[1], 4)
        adapter.to_hidden.load_state_dict(old['to_hidden'], strict=True)
        with torch.no_grad():
            adapter.gate.copy_(old['gate'])
        attention.o_proj = adapter
        cross_payload = payload.get('cross_adapters', ())
        cross_modules = []
        for old_cross in cross_payload:
            index = old_cross['layer']
            if old_cross['projection'] != 'o_proj' or index == 20:
                raise ValueError('Unexpected cross-attention location')
            target_attention = decoder.base.model.layers[index].self_attn
            cross = s.MemorySlotCrossAttention(target_attention.o_proj, decoder._memory_context,
                                                model.config.memory_width, codec, embedding, 4,
                                                old_cross['width'])
            cross.load_memory_state(old_cross['state'])
            target_attention.o_proj = cross
            cross_modules.append(cross)
        final_payload = payload.get('final_adapters', ())
        if final_payload:
            if len(final_payload) != 1:
                raise ValueError('Unexpected final residual layout')
            old_final = final_payload[0]
            final = s.MemoryFinalHiddenResidual(
                decoder.base.model.norm, decoder._memory_context, model.config.memory_width,
                bridge, embedding.shape[1], 4)
            final.to_hidden.load_state_dict(old_final['to_hidden'], strict=True)
            with torch.no_grad():
                final.gate.copy_(old_final['gate'])
            decoder.base.model.norm = final
        decoder.requires_grad_(False)
        bridge.requires_grad_(False)

        def state_for(events):
            state = p.fresh_controlled_state(model)
            fast, velocity = model.initial_state(1)
            for site in fast:
                site[0] = torch.zeros_like(site[0])
            with torch.no_grad():
                for key, value in events:
                    ids = [tok.encode(NAMES[key], add_special_tokens=False)[-1]]
                    ids += tok.encode(s.WORDS[value], add_special_tokens=False)
                    encoded = codec.encode(torch.tensor(ids, device=device))
                    target = torch.zeros((1, model.config.memory_width), device=device)
                    target[0, :encoded.numel()] = encoded
                    for i, site in enumerate(model.memories):
                        fast[i], velocity[i] = site.write_temporal(fast[i], velocity[i],
                            site.current_basis[key:key + 1].to(device), target,
                            torch.ones(1, device=device), torch.zeros(1, dtype=torch.long, device=device), False)
                state.current = [x[0].detach() for x in fast]
                if VERIFY_CODEC_READS and len(events) == 1:
                    key, value = events[0]
                    expected = [tok.encode(NAMES[key], add_special_tokens=False)[-1]]
                    expected += tok.encode(s.WORDS[value], add_special_tokens=False)
                    expected += [tok.eos_token_id]
                    reads = [site.retrieve([fast[i][0]], site.current_basis[key:key + 1].to(device))[0]
                             for i, site in enumerate(model.memories)]
                    actual = codec.decode(torch.stack(reads).mean(0))['token_ids']
                    if actual != expected:
                        raise ValueError('Controlled codec read did not recover written tokens')
            return state

        def probe(group, key, value, state, template=QUESTION, condition='memory'):
            digest = p.frozen_digest(model, state)
            spec = p._memory_args(model, state, {'key': key, 'intent': 'current', 'event_time': 0})
            if condition == 'off':
                spec = {}
            answer, tail = generate(template.format(name=NAMES[key]), spec)
            ablated = None
            if MIX_ABLATION:
                if not hasattr(bridge, 'mix'):
                    raise ValueError('Mix ablation requires the retrieval bridge')
                with torch.no_grad():
                    saved_mix = bridge.mix.detach().clone()
                    bridge.mix.zero_()
                try:
                    ablated = generate(template.format(name=NAMES[key]), spec)
                finally:
                    with torch.no_grad():
                        bridge.mix.copy_(saved_mix)
            cross_ablated = None
            if CROSS_ABLATION:
                if not cross_modules:
                    raise ValueError('Cross ablation requires memory cross-attention')
                original_enabled = [module.enabled for module in cross_modules]
                try:
                    for module in cross_modules:
                        module.enabled = False
                    cross_ablated = generate(template.format(name=NAMES[key]), spec)
                finally:
                    for module, enabled in zip(cross_modules, original_enabled):
                        module.enabled = enabled
            if PROBE_OBSERVER is not None:
                PROBE_OBSERVER(decoder=decoder, tokenizer=tok, spec=spec,
                               cross_modules=tuple(cross_modules), group=group,
                               name=NAMES[key], expected=s.WORDS[value],
                               question=template.format(name=NAMES[key]))
            if p.frozen_digest(model, state) != digest:
                raise ValueError('Evaluation mutated the memory state')
            row = dict(group=group, key=key, name=NAMES[key], expected=s.WORDS[value],
                       question=template.format(name=NAMES[key]), condition=condition,
                       answer=answer, token_ids=tail,
                       fact_correct=correct_answer(answer, NAMES[key], s.WORDS[value]))
            if ablated is not None:
                row.update(zero_mix_answer=ablated[0], zero_mix_token_ids=ablated[1],
                           zero_mix_fact_correct=correct_answer(ablated[0], NAMES[key], s.WORDS[value]))
            if cross_ablated is not None:
                row.update(cross_off_answer=cross_ablated[0], cross_off_token_ids=cross_ablated[1],
                           cross_off_fact_correct=correct_answer(cross_ablated[0], NAMES[key], s.WORDS[value]))
            rows.append(row)
            with (OUT / 'rows.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            print(json.dumps({'completed': len(rows), 'group': group,
                              'correct': row['fact_correct']}), flush=True)
            return row

        saved = {row['expected']: row for row in original['after']
                 if row['group'] == 'held_name' and row['condition'] == 'memory'
                 and row['question'] == QUESTION.format(name=NAMES[3])}
        parity_values = range(len(s.WORDS)) if PARITY_VALUE_INDICES is None else tuple(PARITY_VALUE_INDICES)
        if (not parity_values or any(type(value) is not int or not 0 <= value < len(s.WORDS)
                                     for value in parity_values) or len(set(parity_values)) != len(parity_values)):
            raise ValueError('Invalid parity values')
        for value in parity_values:
            word = s.WORDS[value]
            row = probe('reload_parity', 3, value, state_for([(3, value)]))
            if row['token_ids'] != saved[word]['token_ids']:
                raise ValueError(f'Reloaded generation mismatch for value {value}')
        if EXTRA_WORDS:
            base_count = len(s.WORDS)
            if set(EXTRA_WORDS) & set(s.WORDS) or len(set(EXTRA_WORDS)) != len(EXTRA_WORDS):
                raise ValueError('Extra values must be distinct from original training values')
            s.WORDS = tuple(s.WORDS) + tuple(EXTRA_WORDS)
            for key in EXTRA_KEYS:
                for offset in range(len(EXTRA_WORDS)):
                    first = base_count + offset
                    second = base_count + (offset + 1) % len(EXTRA_WORDS)
                    probe(EXTRA_GROUP_PREFIX + '_value', key, first, state_for([(key, first)]))
                    probe(EXTRA_GROUP_PREFIX + '_swap', key, second, state_for([(key, second)]))
        if not EXTRA_ONLY:
            for key in (4, 5):
                for value in (0, 1, 3, 9, 16, 17):
                    probe('new_identity', key, value, state_for([(key, value)]))
            for template in ('{name}爱吃什么水果？只回答名字。',
                             '查一下记忆，告诉我{name}最喜欢哪种水果。'):
                for value in (0, 1, 16, 17):
                    probe('new_question', 3, value, state_for([(3, value)]), template)
            assignment = [(0, 0), (1, 1), (2, 3), (3, 9), (4, 16), (5, 17)]
            together = state_for(assignment)
            for key, value in assignment:
                probe('multi_memory', key, value, together)
            corrected = state_for(assignment + [(3, 1)])
            for key, value in assignment:
                probe('after_correction', key, 1 if key == 3 else value, corrected)
            for key in (3, 4):
                for value in (1, 16):
                    for condition in ('empty', 'off'):
                        probe('control', key, value, state_for([]), condition=condition)
        ordinary_after = [generate(q, {}) for q in ordinary]
        if ordinary_before != ordinary_after:
            raise ValueError('Ordinary off-path generations changed on reload')
        if p.frozen_digest(model, chat.state) != original_digest:
            raise ValueError('Original individual changed')
    if any(p.sha(path) != digest for path, digest in protected.items()):
        raise ValueError('Protected source files changed')
    summary = defaultdict(lambda: {'correct': 0, 'total': 0})
    for row in rows:
        summary[row['group']]['correct'] += int(row['fact_correct'])
        summary[row['group']]['total'] += 1
    result = dict(summary=dict(summary), rows=rows, source_candidate_sha256=protected[SOURCE / 'candidate.pt'],
                  codec_contract={'slot_count': 5, 'seed': 29071, 'full_fact': True, 'identity': True},
                  parity_verified=len(parity_values), extra_words=list(EXTRA_WORDS), extra_keys=list(EXTRA_KEYS),
                  codec_reads_checked=VERIFY_CODEC_READS, mix_ablation=MIX_ABLATION, cross_ablation=CROSS_ABLATION,
                  ordinary_off_tokens_match=True, source_files_unchanged=True,
                  seconds=time.perf_counter() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  limitations=['Numeric addressing and controlled writes, no natural write routing.',
                               ('Authored development probes; consult the candidate configuration to determine trained versus held values.'
                                if EXTRA_WORDS else 'Authored development probes; all fruit values occurred in training.'),
                               'Strict phrase metric can reject valid paraphrases; raw outputs are retained.',
                               'Off path is the existing hybrid, not untouched original DeepSeek.',
                               'Memory states are recreated for this reader reload, not restored from a saved individual.'])
    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
