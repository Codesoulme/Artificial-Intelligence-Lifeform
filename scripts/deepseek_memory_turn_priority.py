"""Test explicit current-turn priority against unchanged dialogue pooling."""

import argparse
import json
import time
from pathlib import Path

from deepseek_aligned_memory_calibration import NEW_FACTS, load_adapter
from deepseek_aligned_memory_dialogue import OUT as SOURCE_RUN, RANK, answer, matrix_bias, score
from deepseek_factored_relation_router import BANK_RUN
from deepseek_memory_attribute_router import CLARIFY, MISSING, SUPPORTED, choose_cutoff
from deepseek_memory_attribute_span import KEYS, TRAIN as SINGLE_TRAIN, Route, batch, examples, fit, route_action
from deepseek_memory_dialogue_context import ACK, CAL, OUT as DIALOGUE_RUN, SWITCH, TRAIN, features_for, route, samples
from deepseek_output_aligned_memory import MODEL, ROOT, SEED, SOURCE, fingerprint, fresh, load_integrated, read, sha, torch, write

OUT = ROOT / 'runs/deepseek-memory-turn-priority'
HELD_FOLLOWUPS = (
    ('这项具体写了什么？', ('这次我们说的是那个人最乐意喝的饮品。', '这个话题是那人的实际职务。',
     '重点是她回忆中去过的地方。', '这段讲的细节是到访的确切日期。',
     '这里说的是进入场馆花费的金额。', '这里说的重点是同行者的关系。')),
    ('具体内容呢？', ('我在找那位客人最喜爱的喝的东西。', '我在查那人靠哪项工作谋生。',
     '我在查那趟行程实际抵达哪里。', '我在回忆事情是哪天发生的。',
     '我在找饮品买了几瓶的记录。', '我在找对方为何走那条路线的记录。')),
)
HELD_SWITCHES = (
    ('刚刚正说她的职业经历。', '这回请查他当时偏爱喝哪种饮品。', 'drink'),
    ('刚刚在讲他们去过的地方。', '这回请查他当时偏爱喝哪种饮品。', 'drink'),
    ('此前的关注点是客人的饮料偏好。', '改问这趟见闻里人的工作岗位叫什么。', 'work'),
    ('此前的关注点是见面发生的日期。', '改问这趟见闻里人的工作岗位叫什么。', 'work'),
    ('先前的话题围绕饮品喜好。', '现在给我一句面试开场白。', 'ordinary'),
    ('先前的话题围绕他的工作。', '现在给我一句面试开场白。', 'ordinary'),
    ('前文谈的是实际到访的地点。', '现在查查他们是在星期几去的。', 'time'),
    ('前文谈的是平常爱喝的饮料。', '现在查查他们是在星期几去的。', 'time'),
)
GATE_CUTOFFS = (.35, .5, .65, .8, .9)


def pack(rows, features):
    return batch([dict(question=r['key']) for r in rows], features)


def train_inheritance(rows, features):
    first, _, _, _ = pack(rows, features)
    labels = torch.tensor([float(r['kind'] == 'follow') for r in rows])
    weight = torch.where(labels == 1, .5 / labels.mean(), .5 / (1 - labels.mean()))
    torch.manual_seed(82437)
    gate = torch.nn.Linear(first.shape[-1], 1)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=.025, weight_decay=.04)
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(gate(first).flatten(), labels, weight=weight)
        loss.backward()
        optimizer.step()
    return gate.eval().requires_grad_(False)


def gate_probabilities(gate, rows, features):
    tails = torch.stack([features[row['key']][0] for row in rows])
    with torch.no_grad():
        return gate(tails).flatten().sigmoid().tolist()


def combine(contextual, current, probabilities, cutoff):
    combined = []
    for old, new, probability in zip(contextual, current, probabilities, strict=True):
        inherit = probability >= cutoff
        source = old if inherit else new
        combined.append(dict(source, inherits=inherit, inheritance_probability=probability))
    return combined


def choose_inheritance(rows, probabilities):
    grid = []
    for cutoff in GATE_CUTOFFS:
        predictions = [p >= cutoff for p in probabilities]
        grid.append(dict(cutoff=cutoff, misses=sum(r['kind'] == 'follow' and not pred for r, pred in zip(rows, predictions)),
                         false_inheritance=sum(r['kind'] != 'follow' and pred for r, pred in zip(rows, predictions))))
    winner = min(grid, key=lambda r: (r['false_inheritance'] + r['misses'], r['false_inheritance'], -r['cutoff']))
    return winner['cutoff'], grid


def summarize(routes, generated, cutoffs):
    summary = {}
    for mode, group in routes.items():
        rows = [row for row in generated if row['mode'] == mode]
        followups = [row for row in group if row['kind'] == 'follow']
        switches = [row for row in group if row['kind'] == 'switch']
        supported = [row for row in group if row['field'] in SUPPORTED]
        pairs = [[row for row in rows if row['key'] == query['key']] for query in supported]
        summary[mode] = dict(
            cutoff=cutoffs[mode], followups=len(followups), followup_field_correct=sum(r['selected'] == r['field'] for r in followups),
            switches=len(switches), switch_field_correct=sum(r['selected'] == r['field'] for r in switches),
            switch_reads=sum(route_action(r, cutoffs[mode]) == 'read' and r['selected'] == r['field'] for r in switches),
            wrong_reads=sum(route_action(r, cutoffs[mode]) == 'read' and r['selected'] != r['field'] for r in group),
            wrong_injections=sum(r['start'] is not None and r['selected'] != r['field'] for r in rows),
            ordinary_retained=sum(r['field'] == 'ordinary' and r['action'] == 'ordinary' for r in rows),
            strict=sum(r['strict'] for r in rows), answers=2 * len(supported),
            pairs_correct=sum(len(pair) == 2 and all(r['strict'] for r in pair) for pair in pairs), pairs=len(pairs),
            followups_inherited=sum(r.get('inherits', True) for r in followups),
            switches_inherited=sum(r.get('inherits', True) for r in switches))
    return summary


def main(out):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if out.exists():
        raise FileExistsError(out)
    if torch.cuda.mem_get_info()[0] < 3.7 * 1024**3:
        raise RuntimeError('Insufficient free GPU memory')
    started = time.perf_counter()
    torch.set_num_threads(2)
    train = samples(TRAIN, SWITCH['train'])
    direct_train = [dict(key=f'direct-{i}', question=row['question'], field=row['field'], kind='direct')
                    for i, row in enumerate(examples(SINGLE_TRAIN))]
    calibration = samples(CAL, SWITCH['calibration'])
    held = samples(HELD_FOLLOWUPS, HELD_SWITCHES)
    train_questions = {row['question'] for row in train + direct_train}
    if train_questions & {row['question'] for row in calibration + held}:
        raise ValueError('Question overlap')
    candidate_path, banks_path, dialogue_path = SOURCE_RUN / 'candidate.pt', BANK_RUN / 'banks.pt', DIALOGUE_RUN / 'router.pt'
    for path, digest in ((candidate_path, read(SOURCE_RUN / 'result.json')['candidate_sha256']),
                         (banks_path, read(BANK_RUN / 'result.json')['banks_sha256']),
                         (dialogue_path, read(DIALOGUE_RUN / 'result.json')['artifact_sha256']['router.pt'])):
        if sha(path) != digest:
            raise ValueError('Source checkpoint changed')
    sources = (Path(__file__), SOURCE, candidate_path, banks_path, dialogue_path,
               *(MODEL / name for name in ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json')))
    hashes = fingerprint(sources)
    values = {v for facts in NEW_FACTS.values() for versions in facts.values() for v in versions}
    if any(v in row.get('history', '') + ACK + row['question'] for row in train + direct_train + calibration + held for v in values):
        raise ValueError('A stored fact occurs in the conversation')
    out.mkdir()
    write(out / 'plan.json', dict(train=train, direct_train=direct_train, calibration=calibration, held=held,
          sources=hashes, entity=22, acknowledgement=ACK, gate_grid=GATE_CUTOFFS, gain_multiplier=2.))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
              dtype=torch.float16, attn_implementation='sdpa', device_map={'': 'cuda'}, low_cpu_mem_usage=True).eval()
    decoder, parent, state = load_integrated(SOURCE, base, MODEL, individual_device='cpu')
    del parent, state
    decoder.eval().requires_grad_(False)
    current_cache = {}

    def current_features(rows):
        result = {}
        for row in rows:
            if row['question'] not in current_cache:
                current_cache[row['question']] = features_for(decoder, tokenizer, row, False)
            result[row['key']] = current_cache[row['question']]
        return result

    train_features = current_features(train + direct_train)
    inherited = train_inheritance(train + direct_train, train_features)
    explicit = [r for r in train if r['kind'] == 'switch'] + direct_train
    direct = fit(pack(explicit, train_features), torch.tensor([KEYS.index(r['field']) for r in explicit]), localized=True)
    saved = out / 'routing.pt'
    torch.save({'inheritance': inherited.state_dict(), 'direct': direct.state_dict()}, saved)
    checkpoint = torch.load(saved, weights_only=True)
    inherited.load_state_dict(checkpoint['inheritance'])
    direct.load_state_dict(checkpoint['direct'])
    width = base.lm_head.weight.shape[1]
    previous = Route(width, localized=True)
    previous.load_state_dict(torch.load(dialogue_path, weights_only=True))
    previous.eval().requires_grad_(False)
    cal_current = current_features(calibration)
    cal_full = {r['key']: features_for(decoder, tokenizer, r, True) for r in calibration}
    cal_base = route(previous, calibration, cal_full)
    cal_gate = gate_probabilities(inherited, calibration, cal_current)
    gate_cutoff, gate_grid = choose_inheritance(calibration, cal_gate)
    cal_chosen = combine(cal_base, route(direct, calibration, cal_current), cal_gate, gate_cutoff)
    cutoff_base, grid_base = choose_cutoff(cal_base)
    cutoff_priority, grid_priority = choose_cutoff(cal_chosen)
    cutoffs = {'contextual': cutoff_base, 'priority': cutoff_priority}
    held_current = current_features(held)
    held_full = {r['key']: features_for(decoder, tokenizer, r, True) for r in held}
    old_routes = route(previous, held, held_full)
    routes = {'contextual': old_routes,
              'priority': combine(old_routes, route(direct, held, held_current),
                                  gate_probabilities(inherited, held, held_current), gate_cutoff)}
    write(out / 'routes.json', dict(held=routes, cutoffs=cutoffs, gate_cutoff=gate_cutoff, gate_grid=gate_grid,
                                    calibration={'contextual': grid_base, 'priority': grid_priority}))
    print(json.dumps(dict(gate=gate_cutoff, cutoffs=cutoffs)), flush=True)

    _, style, timing, columns = load_adapter(torch.load(candidate_path, weights_only=True), width)
    organism, _ = fresh()
    site = organism.memories[0]
    banks = torch.load(banks_path, weights_only=True, map_location='cpu')
    basis = torch.linalg.qr(torch.randn(width, RANK, generator=torch.Generator().manual_seed(SEED))).Q.to('cuda')
    generated, cache, memory = [], {}, {}
    with torch.inference_mode(), (out / 'progress.jsonl').open('x', encoding='utf8') as log:
        for mode, group in routes.items():
            for row in group:
                field, selected = row['field'], row['selected']
                action = route_action(row, cutoffs[mode])
                use_current = mode == 'priority' and not row['inherits']
                prompt = (held_current if use_current else held_full)[row['key']][3]
                for version in ((0, 1) if field in SUPPORTED else (1,)):
                    expected = NEW_FACTS[22][field][version] if field in SUPPORTED else None
                    tokens, opening, suffix, start = [], '', '', None
                    if action in ('read', 'ordinary'):
                        key = (tuple(prompt), action, selected, version if action == 'read' else None)
                        if key not in cache:
                            if action == 'read':
                                bank_key = selected, version
                                if bank_key not in memory:
                                    bias = matrix_bias(base.lm_head, site, banks[bank_key], site.current_basis[22:23], basis)
                                    if bias is None:
                                        raise ValueError('Expected populated bank')
                                    memory[bank_key] = 2 * bias
                                cache[key] = answer(decoder, tokenizer, prompt, style, columns, timing, memory[bank_key])
                            else:
                                cache[key] = answer(decoder, tokenizer, prompt)
                        response, trace = cache[key]
                        tokens = [step['token'] for step in trace]
                        start = next((i for i, step in enumerate(trace) if step['memory_position'] == 0), None)
                        if start is not None:
                            opening = tokenizer.decode(tokens[:start], skip_special_tokens=True)
                            suffix = tokenizer.decode(tokens[start:], skip_special_tokens=True).strip()
                    else:
                        response = MISSING if action == 'missing' else CLARIFY
                    result = dict(mode=mode, key=row['key'], kind=row['kind'], field=field, relation=field,
                                  selected=selected, read_relation=selected, version=version, expected=expected, action=action,
                                  current_only=use_current, start=start, response=response, tokens=tokens,
                                  opening=opening, fact_suffix=suffix)
                    result['strict'] = score(result, values)
                    generated.append(result)
                    log.write(json.dumps(result, ensure_ascii=False) + '\n')
                log.flush()
            print('Generated', len(generated), flush=True)
    if fingerprint(sources) != hashes:
        raise ValueError('Experiment inputs changed')
    summary = summarize(routes, generated, cutoffs)
    write(out / 'result.json', dict(summary=summary, rows=generated, elapsed_seconds=time.perf_counter() - started,
          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
          artifact_sha256={name: sha(out / name) for name in ('plan.json', 'routes.json', 'routing.pt')}))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    main(parser.parse_args().out)
