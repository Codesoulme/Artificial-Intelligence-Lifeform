"""Measure relation abstention and read strength on separate dialogue fixtures."""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

from deepseek_aligned_memory_dialogue import (
    OUT as SOURCE_RUN, FACTS, HELD, RELATIONS, RANK, THRESHOLD, TRAIN, answer, matrix_bias,
    question_features, score,
)
from deepseek_output_aligned_memory import (
    MODEL, ROOT, SEED, SLOTS, SOURCE, encode, fingerprint, fresh, load_integrated, prefix,
    read, sha, torch, write, write_bank,
)

OUT = ROOT / 'runs/deepseek-aligned-memory-calibration'
CALIBRATION = (
    ('drink', '回想刚才那次经历，他平日爱点什么饮料？'),
    ('drink', '记忆中，此人偏好什么喝的？'),
    ('drink', '她上次提到自己喜欢哪样饮品？'),
    ('work', '从那段记录看，这人的工作是什么？'),
    ('work', '说说他当时靠什么职业谋生。'),
    ('work', '回忆一下，这位当事人是干什么的？'),
    ('place', '回想那次见闻，他去的是哪一处？'),
    ('place', '那一回，她到访的是哪个地点？'),
    ('place', '那段记录说他们到了哪儿？'),
    ('ordinary', '记录一下今天常见的茶与咖啡有哪些区别。'),
    ('ordinary', '什么饮料适合做成无糖版本？'),
    ('ordinary', '介绍不同职业选择的利弊。'),
    ('ordinary', '当事人通常需要在求职面试中准备什么？'),
    ('ordinary', '请推荐一个周末可到访的地方。'),
    ('ordinary', '如何把一段出游经历写成游记？'),
    ('ordinary', '描述一下天空为什么是蓝色的。'),
)
TEST = (
    ('drink', '按你的经历说说，这个人爱喝哪一样？'),
    ('drink', '档案中写到他最中意的饮料，你记得是什么吗？'),
    ('work', '当初记录的是哪一种职业？说清楚这个人做哪行。'),
    ('work', '回顾那次的事，这个人当时以什么为业？'),
    ('place', '那趟出行的目的地，在你记得的那次经历里是哪里？'),
    ('place', '复述一下当时这人实际去了哪儿。'),
    ('ordinary', '我想跟这位当事人聊聊饮料品牌，帮我想三个开场白。'),
    ('ordinary', '请写一段经历，描写一个人第一次到访陌生地方。'),
    ('ordinary', '比较手工职业和办公室工作的日常节奏。'),
    ('ordinary', '咖啡放凉后，口感通常会有什么变化？'),
    ('ordinary', '解释下怎样规划一周的工作。'),
    ('ordinary', '把十六除以四，再加三，是多少？'),
)
NEW_FACTS = {
    22: {'drink': ('桑葚燕麦冰饮', '桂花梅子茶'), 'work': ('珐琅器修复员', '海洋声学分析员'),
         'place': ('槐月旧粮仓', '沙屿风力观测站')},
    38: {'drink': ('陈皮葡萄气泡饮', '龙眼冻乌龙'), 'work': ('木船模型制作师', '山地电网巡检员'),
         'place': ('绛石峡谷驿站', '北栎旧渡口')},
}
THRESHOLDS = (.4, .5, .6, .65, .7, .75, .8, .85, .9, .95, 1.)


def select_threshold(calibration):
    grid = []
    for threshold in THRESHOLDS:
        successes = errors = 0
        for row in calibration:
            active = row['selected'] != 'ordinary' and row['confidence'] >= threshold
            successes += active and row['selected'] == row['relation']
            errors += active and row['selected'] != row['relation']
        grid.append({'threshold': threshold, 'correct_reads': successes, 'wrong_reads': errors})
    selected = max(grid, key=lambda row: (-row['wrong_reads'], row['correct_reads'], row['threshold']))
    return selected['threshold'], grid


def load_adapter(candidate, width):
    columns = torch.tensor(candidate['columns'], device='cuda')
    route = torch.nn.Linear(width, len(RELATIONS))
    style = torch.nn.Sequential(torch.nn.Linear(width, 24), torch.nn.Tanh(), torch.nn.Linear(24, len(columns)))
    gate = torch.nn.Linear(width, 1)
    for name, module in (('route', route), ('style', style), ('gate', gate)):
        module.load_state_dict(candidate[name])
        module.eval().requires_grad_(False).to('cuda')
    return route, style, gate, columns


def summarize(rows):
    result = {}
    for suite in sorted({row['suite'] for row in rows}):
        for mode in sorted({r['mode'] for r in rows if r['suite'] == suite}):
            group = [r for r in rows if r['suite'] == suite and r['mode'] == mode]
            facts = [r for r in group if r['expected'] is not None]
            controls = [r for r in group if r['expected'] is None]
            pairs = {}
            for row in facts:
                pairs.setdefault((row['question'], row['entity']), []).append(row)
            result[f'{suite}/{mode}'] = {
                'facts': len(facts), 'routed': sum(r['read_admitted'] for r in facts),
                'triggered': sum(r['start'] is not None for r in facts),
                'exact_suffix': sum(r['exact_suffix'] for r in facts), 'strict': sum(r['strict'] for r in facts),
                'complete_pairs': sum(len(pair) == 2 and all(r['strict'] for r in pair) for pair in pairs.values()),
                'pairs': sum(len(pair) == 2 for pair in pairs.values()), 'controls': len(controls),
                'control_triggered': sum(r['start'] is not None for r in controls),
                'control_baseline_equal': sum(r['reference_equal'] for r in controls),
            }
    return result


@torch.inference_mode()
def main(out):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if out.exists():
        raise FileExistsError(out)
    if torch.cuda.mem_get_info()[0] < 3.7 * 1024**3:
        raise RuntimeError('Insufficient free GPU memory')
    torch.set_num_threads(2)
    started = time.perf_counter()
    questions = [q for _, q in CALIBRATION + TEST]
    earlier_questions = {q for group in TRAIN.values() for q, _ in group} | {q for group in HELD.values() for q in group}
    if len(set(questions)) != len(questions) or set(questions) & earlier_questions:
        raise ValueError('Question overlap')
    source_result = read(SOURCE_RUN / 'result.json')
    candidate_path = SOURCE_RUN / 'candidate.pt'
    if sha(candidate_path) != source_result['candidate_sha256']:
        raise ValueError('Source candidate digest mismatch')
    paths = (SOURCE, candidate_path, Path(__file__), Path(sys.modules[answer.__module__].__file__),
             *(MODEL / f for f in ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json')))
    hashes = fingerprint(paths)
    out.mkdir()
    write(out / 'plan.json', {'calibration': CALIBRATION, 'test': TEST, 'facts': NEW_FACTS,
                              'threshold_grid': THRESHOLDS, 'gain_multipliers': (1., 2.),
                              'source_sha256': hashes, 'focus_address': 'provided'})
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, dtype=torch.float16,
        attn_implementation='sdpa', device_map={'': 'cuda'}, low_cpu_mem_usage=True).eval()
    decoder, parent, unused_state = load_integrated(SOURCE, base, MODEL, individual_device='cpu')
    del parent, unused_state
    decoder.eval().requires_grad_(False)
    model, _ = fresh()
    site = model.memories[0]
    width = model.config.memory_width
    del model, _
    gc.collect()
    torch.cuda.empty_cache()
    head = base.lm_head
    hidden_width = head.weight.shape[1]
    basis = torch.linalg.qr(torch.randn(hidden_width, RANK, generator=torch.Generator().manual_seed(SEED))).Q.to('cuda')
    candidate = torch.load(candidate_path, map_location='cpu', weights_only=True)
    classifier, style, gate, columns = load_adapter(candidate, hidden_width)

    def route(question):
        prompt = encode(tokenizer, prefix(tokenizer, question))
        probs = classifier(question_features(decoder, prompt)).softmax(dim=-1)
        return prompt, RELATIONS[int(probs.argmax())], float(probs.max())

    calibration = []
    for relation, question in CALIBRATION:
        _, selected, confidence = route(question)
        calibration.append(dict(relation=relation, question=question, selected=selected, confidence=confidence))
    threshold, grid = select_threshold(calibration)
    write(out / 'calibration.json', {'rows': calibration, 'threshold': threshold, 'grid': grid})
    print(json.dumps({'threshold': threshold, 'selection': [r for r in grid if r['threshold'] == threshold]}), flush=True)

    banks = {}
    for relation in RELATIONS[:-1]:
        bank = torch.zeros(SLOTS, width, width)
        for version in (0, 1):
            for entity, group in NEW_FACTS.items():
                ids = [*encode(tokenizer, group[relation][version]), tokenizer.eos_token_id]
                if not 3 <= len(ids) <= SLOTS or set(ids[:-1]) & set(tokenizer.all_special_ids):
                    raise ValueError(f'Unexpected fact token length: {entity}/{relation}/{version} ({len(ids)})')
                directions = torch.nn.functional.normalize(head.weight[ids].float() @ basis, dim=-1)
                target = torch.zeros_like(bank[:, :, 0])
                target[:len(ids), :RANK] = directions.cpu()
                bank = write_bank(site, bank, site.current_basis[entity:entity + 1], target)
            banks[relation, version] = bank.clone()
        banks[relation, 'cleared'] = write_bank(site, bank, site.current_basis[22:23], torch.zeros(SLOTS, width))
    torch.save(banks, out / 'banks.pt')

    all_values = {v for facts in (FACTS, NEW_FACTS) for group in facts.values() for versions in group.values() for v in versions}
    rows = []
    modes = (('original', THRESHOLD, 1.), ('original_double_gain', THRESHOLD, 2.),
             ('calibrated', threshold, 1.), ('calibrated_double_gain', threshold, 2.))
    with (out / 'progress.jsonl').open('x', encoding='utf-8') as stream:
        for relation, question in TEST:
            prompt, selected, confidence = route(question)
            _, base_trace = answer(decoder, tokenizer, prompt)
            base_tokens = [s['token'] for s in base_trace]
            if relation == 'ordinary':
                cases = [(22, 1, None)]
            else:
                cases = [(entity, version, facts[relation][version]) for entity, facts in NEW_FACTS.items() for version in (0, 1)]
                cases += [(22, 'cleared', None), (49, 1, None)]
            for entity, version, expected in cases:
                biases = None
                if selected != 'ordinary' and confidence >= min(threshold, THRESHOLD):
                    biases = matrix_bias(head, site, banks[selected, version], site.current_basis[entity:entity + 1], basis)
                for mode, cutoff, gain in modes:
                    active = biases is not None and confidence >= cutoff
                    trace = base_trace
                    if active:
                        _, trace = answer(decoder, tokenizer, prompt, style, columns, gate, biases * gain)
                    start = next((i for i, item in enumerate(trace) if item['memory_position'] == 0), None)
                    tokens = [item['token'] for item in trace]
                    opening = tokenizer.decode(tokens[:start], skip_special_tokens=True).strip() if start is not None else ''
                    suffix = tokenizer.decode(tokens[start:], skip_special_tokens=True).strip() if start is not None else ''
                    row = dict(suite='test', mode=mode, question=question, relation=relation, read_relation=selected,
                               confidence=confidence, entity=entity, version=version, expected=expected, start=start,
                               opening=opening, fact_suffix=suffix, read_admitted=active,
                               exact_suffix=expected is not None and suffix.rstrip('。.!！ ') == expected,
                               reference_equal=tokens == base_tokens, tokens=tokens)
                    row['strict'] = score(row, all_values)
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            print('Scored test rows', len(rows), flush=True)

    diagnostics = []
    for row in source_result['rows']:
        if row['expected'] is None or row['start'] is None or row['fact_suffix'].rstrip('。.!！ ') == row['expected']:
            continue
        relation, entity, version = row['relation'], row['entity'], row['version']
        prompt, _, _ = route(row['question'])
        ids = [*encode(tokenizer, row['expected']), tokenizer.eos_token_id]
        bank = candidate['banks'][f'{relation}:{version}']
        bias = matrix_bias(head, site, bank, site.current_basis[entity:entity + 1], basis)
        generated = [step['token'] for step in row['trace']]
        for slot, target in enumerate(ids):
            position = row['start'] + slot
            if generated[position] == target:
                continue
            previous = torch.tensor([prompt + generated[:position]], device='cuda')
            logits = decoder(input_ids=previous, use_cache=False).logits[0, -1].float()
            winner = int((logits + bias[slot]).argmax())
            if winner != generated[position]:
                raise ValueError('Prior divergence did not reproduce')
            diagnostics.append(dict(relation=relation, entity=entity, version=version, slot=slot,
                target=tokenizer.decode([target]), observed=tokenizer.decode([winner]),
                evidence_rank=1 + int((bias[slot] > bias[slot, target]).sum()),
                target_minus_observed_base=float(logits[target] - logits[winner]),
                target_minus_observed_memory=float(bias[slot, target] - bias[slot, winner]),
                double_gain_winner=tokenizer.decode([int((logits + 2 * bias[slot]).argmax())])))
            break
    if fingerprint(paths) != hashes:
        raise ValueError('A protected source changed')
    summary = summarize(rows)
    write(out / 'result.json', {'summary': summary, 'rows': rows, 'previous_error_diagnostics': diagnostics,
                               'banks_sha256': sha(out / 'banks.pt'), 'plan_sha256': sha(out / 'plan.json'),
                               'calibration_sha256': sha(out / 'calibration.json'), 'threshold': threshold,
                               'elapsed_seconds': time.perf_counter() - started,
                               'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
    print(json.dumps({'summary': summary, 'previous_error_diagnostics': diagnostics}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    main(parser.parse_args().out)
