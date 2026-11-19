"""Compare equally sourced answer adapters on unseen addresses and conversations."""

import argparse
import json
import time
from itertools import chain
from pathlib import Path

from deepseek_aligned_memory_calibration import load_adapter
from deepseek_aligned_memory_dialogue import OUT as ADAPTER_RUN, RANK, answer, matrix_bias, score
from deepseek_memory_attribute_router import CLARIFY, MISSING
from deepseek_memory_attribute_span import Route, route_action
from deepseek_memory_dialogue_context import ACK, OUT as DIALOGUE_RUN, features_for, route
from deepseek_memory_field_decoding import FIELDS, OUT as FIELD_RUN, fit_field
from deepseek_memory_turn_priority import OUT as PRIORITY_RUN, combine, gate_probabilities
from deepseek_output_aligned_memory import (
    MODEL, ROOT, SEED, SLOTS, SOURCE, encode, fingerprint, fresh, load_integrated, read, sha, torch, write, write_bank,
)

OUT = ROOT / 'runs/deepseek-memory-field-transfer'
FACTS = {
    26: {'drink': ('栗子米浆', '青杏冰茶'), 'work': ('海堤测绘员', '织机检修师'),
         'place': ('杉浦水闸', '新荷档案馆')},
    47: {'drink': ('山楂苏打', '檀香奶茶'), 'work': ('窑炉技师', '索道安全员'),
         'place': ('芦岭钟楼', '柏泉旧剧场')},
}
QUESTIONS = (
    (26, 'drink', True, '齐宁在座谈上提起过她最中意的一款饮料。', '你能说出那款的名字吗？'),
    (47, 'work', True, '我翻了何程的资料，想核实他那阵子是靠什么营生的。', '查到了就说一下。'),
    (26, 'place', True, '齐宁的记录提到了她结束那段行程时落脚的地点。', '说来听听？'),
    (47, 'drink', False, '刚看完何程上回出门的路线。', '行程先放一边，何程平常点单最偏爱喝哪样？'),
    (26, 'work', False, '齐宁有次谈起自己常喝的东西。', '这次要填履历，齐宁当时做的是哪份工作？'),
    (47, 'place', False, '何程的工种已经核实了。', '接着核实外出记录，何程那天的目的地是哪儿？'),
    (47, 'time', True, '我想核对何程到底是哪天抵达那座城的。', '能翻到确切的那一项吗？'),
    (26, 'time', False, '齐宁走访的地方已经对上了。', '现在要填表，她又是在哪年哪月出发的？'),
    (47, 'amount', False, '何程以前常点喝的。', '请查那份记录里这杯饮料实际付了多少钱。'),
    (26, 'ordinary', False, '刚谈到齐宁从事的行业。', '先帮我写一句短一点的面试自我介绍。'),
)


@torch.inference_mode()
def build_banks(site, head, tokenizer, basis):
    width = site.current_basis.shape[-1]
    banks = {}
    for field in FIELDS:
        bank = torch.zeros(SLOTS, width, width)
        for version in (0, 1):
            for entity, group in FACTS.items():
                ids = [*encode(tokenizer, group[field][version]), tokenizer.eos_token_id]
                if not 3 <= len(ids) <= SLOTS or set(ids[:-1]) & set(tokenizer.all_special_ids):
                    raise ValueError(f'Fact does not fit bank: {entity}/{field}/{version}')
                target = torch.zeros(SLOTS, width)
                target[:len(ids), :RANK] = torch.nn.functional.normalize(head.weight[ids].float() @ basis, dim=-1).cpu()
                bank = write_bank(site, bank, site.current_basis[entity:entity + 1], target)
            banks[field, version] = bank.clone()
    return banks


def main(out):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if out.exists():
        raise FileExistsError(out)
    if torch.cuda.mem_get_info()[0] < 3.7 * 1024**3:
        raise RuntimeError('Insufficient free GPU memory')
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(82438)
    field_plan = read(FIELD_RUN / 'plan.json')
    priority_setup = read(PRIORITY_RUN / 'routes.json')
    paths = dict(shared=ADAPTER_RUN / 'candidate.pt', specialist=FIELD_RUN / 'field-adapters.pt',
                 dialogue=DIALOGUE_RUN / 'router.pt', priority=PRIORITY_RUN / 'routing.pt')
    expected_digests = dict(shared=read(ADAPTER_RUN / 'result.json')['candidate_sha256'])
    for run, names in ((FIELD_RUN, ('field-adapters.pt', 'plan.json')), (DIALOGUE_RUN, ('router.pt',)),
                       (PRIORITY_RUN, ('routing.pt', 'routes.json'))):
        digests = read(run / 'result.json')['artifact_sha256']
        for name in names:
            if sha(run / name) != digests[name]:
                raise ValueError(f'Source artifact changed: {name}')
    if sha(paths['shared']) != expected_digests['shared']:
        raise ValueError('Source adapter changed')
    protected = (Path(__file__), SOURCE, *paths.values(), FIELD_RUN / 'plan.json', PRIORITY_RUN / 'routes.json',
                 *(MODEL / name for name in ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json')))
    hashes = fingerprint(protected)
    examples = list(chain.from_iterable(field_plan['training'][field] for field in FIELDS))
    rows = [dict(key=f'held-{i}', entity=key, field=field, history=history, question=question, target_turn=int(not inherit))
            for i, (key, field, inherit, history, question) in enumerate(QUESTIONS)]
    values = {value for group in FACTS.values() for pair in group.values() for value in pair}
    if {r['question'] for r in rows} & {r['question'] for r, _, _ in examples}:
        raise ValueError('Evaluation questions occur in training')
    if any(value in text for value in values for text in (ACK, *(r['history'] + r['question'] for r in rows),
           *(r['question'] + r.get('history', '') + opening for r, _, opening in examples))):
        raise ValueError('A stored fact occurs in text')
    out.mkdir()
    write(out / 'plan.json', dict(protected=hashes, examples=examples, rows=rows, facts=FACTS,
          gate_cutoff=priority_setup['gate_cutoff'], cutoff=priority_setup['cutoffs']['priority']))

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
              dtype=torch.float16, attn_implementation='sdpa', device_map={'': 'cuda'}, low_cpu_mem_usage=True).eval()
    decoder, parent, state = load_integrated(SOURCE, base, MODEL, individual_device='cpu')
    del parent, state
    decoder.eval().requires_grad_(False)
    head = base.lm_head
    width = head.weight.shape[1]
    _, original, timing, columns = load_adapter(torch.load(paths['shared'], weights_only=True), width)
    shared, _ = fit_field(decoder, tokenizer, examples, columns)
    torch.save(shared.state_dict(), out / 'shared-style.pt')
    shared.load_state_dict(torch.load(out / 'shared-style.pt', weights_only=True))
    shared.to('cuda')
    specialist = torch.load(paths['specialist'], weights_only=True)
    field_styles = {}
    for field in FIELDS:
        style = torch.nn.Sequential(torch.nn.Linear(width, 24), torch.nn.Tanh(), torch.nn.Linear(24, len(columns)))
        style.load_state_dict(specialist[field]['style'])
        field_styles[field] = style.eval().requires_grad_(False).to('cuda')

    contextual, direct = (Route(width, localized=True) for _ in range(2))
    contextual.load_state_dict(torch.load(paths['dialogue'], weights_only=True))
    priority = torch.load(paths['priority'], weights_only=True)
    direct.load_state_dict(priority['direct'])
    gate = torch.nn.Linear(width, 1).eval()
    gate.load_state_dict(priority['inheritance'])
    full = {r['key']: features_for(decoder, tokenizer, r, True) for r in rows}
    current = {r['key']: features_for(decoder, tokenizer, r, False) for r in rows}
    choices = combine(route(contextual.eval(), rows, full), route(direct.eval(), rows, current),
                      gate_probabilities(gate, rows, current), priority_setup['gate_cutoff'])
    cutoff = priority_setup['cutoffs']['priority']
    write(out / 'routes.json', choices)

    organism, _ = fresh()
    site = organism.memories[0]
    basis = torch.linalg.qr(torch.randn(width, RANK, generator=torch.Generator().manual_seed(SEED))).Q.to('cuda')
    torch.save(build_banks(site, head, tokenizer, basis), out / 'banks.pt')
    banks = torch.load(out / 'banks.pt', weights_only=True)
    outputs, bias_cache, answer_cache = [], {}, {}
    print('Saved and reloaded candidate and banks', flush=True)
    with torch.inference_mode(), (out / 'progress.jsonl').open('x', encoding='utf8') as stream:
        for r in choices:
            key, entity, field = r['key'], r['entity'], r['field']
            auto_action = route_action(r, cutoff)
            trials = [('automatic', auto_action, r['selected'])]
            if field in FIELDS:
                trials.append(('known_field', 'read', field))
            prompt = (full if r['inherits'] else current)[key][3]
            for trial, action, selected in trials:
                for version in ((0, 1) if field in FIELDS else (1,)):
                    expected = FACTS[entity][field][version] if field in FIELDS else None
                    for mode, style in (('original', original), ('shared', shared), ('field', field_styles.get(selected))):
                        start, opening, suffix, tokens = None, '', '', []
                        if action == 'read':
                            bank_key = entity, selected, version
                            if bank_key not in bias_cache:
                                bias = matrix_bias(head, site, banks[selected, version], site.current_basis[entity:entity + 1], basis)
                                if bias is None:
                                    raise ValueError('Expected populated bank')
                                bias_cache[bank_key] = 2 * bias
                            cache_key = mode, key, bank_key
                            if cache_key not in answer_cache:
                                answer_cache[cache_key] = answer(decoder, tokenizer, prompt, style, columns, timing, bias_cache[bank_key])
                            response, trace = answer_cache[cache_key]
                            tokens = [step['token'] for step in trace]
                            start = next((i for i, step in enumerate(trace) if step['memory_position'] == 0), None)
                            if start is not None:
                                opening = tokenizer.decode(tokens[:start], skip_special_tokens=True).strip()
                                suffix = tokenizer.decode(tokens[start:], skip_special_tokens=True).strip()
                        elif action == 'ordinary':
                            cache_key = ('ordinary', key)
                            if cache_key not in answer_cache:
                                answer_cache[cache_key] = answer(decoder, tokenizer, prompt)
                            response, trace = answer_cache[cache_key]
                            tokens = [step['token'] for step in trace]
                        else:
                            response = MISSING if action == 'missing' else CLARIFY
                        result = dict(key=key, entity=entity, relation=field, read_relation=selected, trial=trial, mode=mode,
                                      version=version, expected=expected, action=action, start=start, opening=opening,
                                      fact_suffix=suffix, response=response, tokens=tokens)
                        result['strict'] = score(result, values)
                        result['stored_values_in_response'] = sorted(value for value in values if value in response)
                        outputs.append(result)
                        stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                    stream.flush()
            print('Evaluated', key, len(outputs), flush=True)
    if fingerprint(protected) != hashes:
        raise ValueError('An experiment input changed')
    write(out / 'result.json', dict(rows=outputs, elapsed_seconds=time.perf_counter() - started,
          peak_allocated_bytes=torch.cuda.max_memory_allocated(), artifact_sha256={name: sha(out / name) for name in
          ('plan.json', 'routes.json', 'banks.pt', 'shared-style.pt')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    main(parser.parse_args().out)
