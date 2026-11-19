"""Evaluate schema-aware routing and availability checks over frozen memory banks."""

import argparse
import json
import time
from pathlib import Path

from deepseek_aligned_memory_calibration import NEW_FACTS, load_adapter
from deepseek_aligned_memory_dialogue import (
    OUT as SOURCE_RUN, RANK, RELATIONS, answer, matrix_bias, question_features, score,
)
from deepseek_factored_relation_router import BANK_RUN, TRAIN, examples, predict
from deepseek_output_aligned_memory import (
    MODEL, ROOT, SEED, SOURCE, encode, fingerprint, fresh, load_integrated, prefix, read,
    sha, torch, write,
)

OUT = ROOT / 'runs/deepseek-memory-attribute-router'
PRIOR = ROOT / 'runs/deepseek-factored-relation-router-support'
FIELDS = {
    'drink': '回忆一个人偏爱的饮品是什么，答案是饮品名称',
    'work': '回忆一个人实际从事什么职业，答案是职业或岗位名称',
    'place': '回忆一次实际到访的地点，答案是地点名称',
    'time': '回忆发生日期、时间、时长或频率',
    'amount': '回忆价格、数量、容量、距离或其他数值',
    'other': '回忆其他具体事实，如衣着、原因、关系、交通方式，或同时问多个事实',
    'ordinary': '普通解释、建议、创作或假设任务，无需回忆实际发生的事实',
}
CAL = (
    ('drink', '她当时提过最爱哪种饮料吗？'),
    ('drink', '按之前存的经历，他平常最偏爱喝什么？'),
    ('work', '回头说一下那个人担任的岗位叫什么。'),
    ('work', '你记着的那次见闻里，他以哪一行为业？'),
    ('place', '他们那一趟真正去了哪处地方？'),
    ('place', '按那段回忆报一下她的到访地。'),
    ('time', '从旧记录看，那回她持续在那里待了多久？'),
    ('amount', '回看记录，那杯饮品买成多少元？'),
    ('other', '那次同行的人跟当事人是什么关系？'),
    ('ordinary', '给一个故事角色想一个有趣的职业。'),
    ('ordinary', '请建议三个比较适合雨天到访的地方。'),
    ('ordinary', '解释如何在家调出比较好喝的饮料。'),
)
TEST = (
    ('drink', '那份经历里，这位客人说自己最偏爱什么喝的？'),
    ('work', '回想那次提到的情况，此人具体是做哪行的？'),
    ('place', '按照你记下的那件事，他当时到的是什么地方？'),
    ('time', '当事人去那个地方是星期几？按当时记下的信息回答。'),
    ('time', '翻看那段经历，这个人在哪一年开始做那份工作？'),
    ('amount', '他说喜欢的那种饮品在记录里标了多少毫升？'),
    ('amount', '根据那次记录，他去的地方要付多少门票钱？'),
    ('other', '请回忆他当时穿的外套是什么颜色。'),
    ('other', '那个人上回到了目的地，他是坐什么车去的？'),
    ('ordinary', '假设这个人去了一处陌生地点，帮我续写两句。'),
    ('ordinary', '帮我拟一句询问顾客饮品喜好的开场白。'),
    ('ordinary', '简述职业选择时应考虑的三个因素。'),
    ('ordinary', '十七减去九等于几？'),
)
RECHECK = (
    ('time', '你还记得那次见面是哪一天吗？'),
    ('amount', '请说出那个人喝下的饮料有多少毫升。'),
    ('amount', '按那趟经历的记载，出发地距到访地有多少公里？'),
)
SUPPORTED = frozenset(RELATIONS[:-1])
MISSING = '我没有找到这项记录。'
CLARIFY = '我还不能确定你要查询哪项记录，请具体说明。'
GRID = (.2, .3, .4, .5, .6, .7, .8, .9, 1.)
ATTRIBUTE_TRAIN = {
    'time': ('你记着的那次品尝是在几点？', '从记录看，他何时开始上那份班？',
             '那位当事人多久到访那里一次？', '当时她在那个地方待了几天？', '记录中这件事发生在什么时候？'),
    'amount': ('他为最爱喝的那杯付了多少钱？', '那份职业的工资水平是多少？',
               '她喝的饮品有多大容量？', '到那个景点路程多远？', '那次购买的饮料一共有几瓶？'),
    'other': ('他当时为何选那一种喝的？', '是谁陪她到访那个地方的？',
              '他去那里是乘坐哪种交通工具？', '当时他在咖啡店穿了什么衣服？', '到访地和当事人有何渊源？'),
}
FRESH = (
    ('drink', '请从那个旧档案里说出她最爱喝的东西。'),
    ('work', '依照前头记下的内容，当事人当时的岗位是哪一个？'),
    ('place', '那次真实出行里，这个人最终在哪一处落脚？'),
    ('time', '档案里提到那个人是哪天抵达那里的？'),
    ('time', '那回他在那个岗位上一共做了多久？'),
    ('amount', '他那次点自己爱喝的饮料，杯子装了多少液体？'),
    ('amount', '根据那次见闻，那个地方每人收多少入场费？'),
    ('other', '那份记录中，介绍这位当事人去那里的人是谁？'),
    ('other', '当时有什么原因让他选中那款饮品？'),
    ('ordinary', '写一段虚构经历，人物的职业和饮料由你设定。'),
    ('ordinary', '请介绍选择出游目的地的一般方法。'),
    ('ordinary', '说明人在高温下如何补充水分。'),
    ('ordinary', '把十四乘二后减去五。'),
)


def classify(decoder, tokenizer, question):
    keys = tuple(FIELDS)
    scores = []
    for order in (keys, keys[3:] + keys[:3]):
        choices = dict(zip('ABCDEFG', order))
        listing = '\n'.join(f'{letter}. {FIELDS[field]}' for letter, field in choices.items())
        instruction = ('识别用户请求需要回答的属性。以用户真正索要的信息为准。'
                       '不回答原问题，只输出一个选项字母。\n' + listing
                       + '\n用户请求：' + json.dumps(question, ensure_ascii=False))
        ids = encode(tokenizer, prefix(tokenizer, instruction))
        inputs = torch.tensor([ids], device='cuda')
        labels = [encode(tokenizer, letter) for letter in choices]
        if any(len(label) != 1 for label in labels):
            raise ValueError('Choice labels must tokenize to one token')
        logits = decoder(input_ids=inputs, use_cache=False).logits[0, -1].float()
        option_scores = logits[[label[0] for label in labels]].log_softmax(dim=0).tolist()
        scores.append(dict(zip(order, option_scores)))
    mean_scores = torch.tensor([sum(row[key] for row in scores) / len(scores) for key in keys])
    probabilities = dict(zip(keys, mean_scores.softmax(dim=0).tolist()))
    selected = max(probabilities, key=probabilities.get)
    return selected, probabilities[selected], probabilities, [max(row, key=row.get) for row in scores]


def train_attributes(decoder, tokenizer):
    items = examples(TRAIN) + [(field, q) for field, questions in ATTRIBUTE_TRAIN.items() for q in questions]
    keys = tuple(FIELDS)
    vectors = torch.stack([question_features(decoder, encode(tokenizer, prefix(tokenizer, q))).cpu()
                           for _, q in items])
    targets = torch.tensor([keys.index(field) for field, _ in items])
    with torch.inference_mode(False), torch.enable_grad():
        # Copy features created in inference mode before training the head.
        vectors = vectors.clone()
        targets = targets.clone()
        torch.manual_seed(82429)
        head = torch.nn.Linear(vectors.shape[1], len(keys))
        optimizer = torch.optim.AdamW(head.parameters(), lr=.03, weight_decay=.04)
        for _ in range(400):
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(head(vectors), targets).backward()
            optimizer.step()
    return head.eval().requires_grad_(False)


def choose_cutoff(rows):
    grid = []
    for cutoff in GRID:
        admitted = [row for row in rows if row['confidence'] >= cutoff and row['order_agrees']]
        grid.append(dict(cutoff=cutoff,
                         wrong_reads=sum(r['selected'] in SUPPORTED and r['selected'] != r['field'] for r in admitted),
                         correct_labels=sum(r['selected'] == r['field'] for r in admitted)))
    chosen = max(grid, key=lambda r: (-r['wrong_reads'], r['correct_labels'], r['cutoff']))
    return chosen['cutoff'], grid


def summarize(rows):
    summary = {}
    for suite in ('held', 'previous_errors', 'missing_address'):
        for mode in ('prior', 'schema'):
            group = [r for r in rows if r['suite'] == suite and r['mode'] == mode]
            present = [r for r in group if r['expected'] is not None]
            missing = [r for r in group if r['expected'] is None and r['field'] != 'ordinary']
            ordinary = [r for r in group if r['field'] == 'ordinary']
            summary[f'{suite}/{mode}'] = {
                'present': len(present), 'strict': sum(r['strict'] for r in present),
                'complete_pairs': sum(len(pair) == 2 and all(r['strict'] for r in pair) for pair in
                                      ([r for r in present if r['question'] == q] for q in {r['question'] for r in present})),
                'missing': len(missing), 'explicit_missing': sum(r['action'] == 'missing' for r in missing),
                'missing_injected': sum(r['start'] is not None for r in missing),
                'ordinary': len(ordinary), 'ordinary_baseline_equal': sum(r['baseline_equal'] for r in ordinary),
                'clarifications': sum(r['action'] == 'clarify' for r in group),
            }
    return summary


@torch.inference_mode()
def main(out, trained=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(2)
    started = time.perf_counter()
    held = FRESH if trained else TEST
    all_questions = CAL + held + RECHECK
    train_questions = {q for _, q in examples(TRAIN)} | {q for group in ATTRIBUTE_TRAIN.values() for q in group}
    if (len({question for _, question in all_questions}) != len(all_questions)
            or train_questions.intersection(q for _, q in all_questions)):
        raise ValueError('Duplicate or overlapping questions')
    candidate_path = SOURCE_RUN / 'candidate.pt'
    banks_path = BANK_RUN / 'banks.pt'
    prior_path = PRIOR / 'routes.pt'
    for path, run, digest_field in ((candidate_path, SOURCE_RUN, 'candidate_sha256'),
                                    (banks_path, BANK_RUN, 'banks_sha256'), (prior_path, PRIOR, 'routes_sha256')):
        if sha(path) != read(run / 'result.json')[digest_field]:
            raise ValueError('Checkpoint hash mismatch: ' + str(path))
    prior_result = read(PRIOR / 'result.json')
    prior_cutoff = prior_result['summary']['factored']['threshold']
    sources = (Path(__file__), SOURCE, candidate_path, banks_path, prior_path,
               *(MODEL / name for name in ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json')))
    hashes = fingerprint(sources)
    out.mkdir()
    write(out / 'plan.json', {'fields': FIELDS, 'calibration': CAL, 'held': held, 'previous_errors': RECHECK,
                              'trained': trained, 'attribute_training': ATTRIBUTE_TRAIN if trained else None,
                              'source_sha256': hashes, 'prior_cutoff': prior_cutoff, 'cutoff_grid': GRID,
                              'focus_entity': 22, 'matrix_gain_multiplier': 2.})
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, dtype=torch.float16,
        attn_implementation='sdpa', device_map={'': 'cuda'}, low_cpu_mem_usage=True).eval()
    decoder, parent, old_state = load_integrated(SOURCE, base, MODEL, individual_device='cpu')
    del parent, old_state
    decoder.eval().requires_grad_(False)
    attribute_head = train_attributes(decoder, tokenizer) if trained else None
    if trained:
        torch.save(attribute_head.state_dict(), out / 'attribute-head.pt')
        attribute_head.load_state_dict(torch.load(out / 'attribute-head.pt', weights_only=True))
    fields = {}
    for field, question in all_questions:
        if attribute_head is None:
            selected, confidence, probabilities, votes = classify(decoder, tokenizer, question)
        else:
            vector = question_features(decoder, encode(tokenizer, prefix(tokenizer, question))).cpu()
            probabilities = dict(zip(FIELDS, attribute_head(vector).softmax(dim=0).tolist()))
            selected = max(probabilities, key=probabilities.get)
            confidence = probabilities[selected]
            votes = [selected]
        fields[question] = dict(question=question, field=field, selected=selected, confidence=confidence,
                               probabilities=probabilities, votes=votes, order_agrees=all(v == selected for v in votes))
    cutoff, grid = choose_cutoff([fields[q] for _, q in CAL])
    write(out / 'classification.json', {'rows': list(fields.values()), 'cutoff': cutoff, 'grid': grid})
    print(json.dumps({'cutoff': cutoff, 'calibration': next(r for r in grid if r['cutoff'] == cutoff)}), flush=True)

    width = base.lm_head.weight.shape[1]
    _, style, gate, columns = load_adapter(torch.load(candidate_path, weights_only=True), width)
    prior = torch.nn.Linear(width, len(RELATIONS))
    prior.load_state_dict(torch.load(prior_path, weights_only=True)['factored'])
    prior.eval().requires_grad_(False)
    organism, _ = fresh()
    site = organism.memories[0]
    banks = torch.load(banks_path, weights_only=True, map_location='cpu')
    basis = torch.linalg.qr(torch.randn(width, RANK, generator=torch.Generator().manual_seed(SEED))).Q.to('cuda')
    all_values = {value for entity in NEW_FACTS.values() for versions in entity.values() for value in versions}
    cases = [(suite, field, question, 22, version) for suite, questions in (('held', held), ('previous_errors', RECHECK))
             for field, question in questions for version in ((0, 1) if field in SUPPORTED else (1,))]
    cases.extend(('missing_address', field, question, entity, version) for field, question in held if field in SUPPORTED
                 for entity, version in ((22, 'cleared'), (49, 1)))
    rows, questions, memory = [], {}, {}
    with (out / 'progress.jsonl').open('x', encoding='utf8') as stream:
        for suite, field, question, entity, version in cases:
            if question not in questions:
                prompt = encode(tokenizer, prefix(tokenizer, question))
                pred, confidence, _ = predict(prior, question_features(decoder, prompt).cpu(), factored=True)
                baseline_text, baseline_trace = answer(decoder, tokenizer, prompt)
                questions[question] = (prompt, pred, confidence, baseline_text, baseline_trace)
            prompt, pred, prior_confidence, baseline_text, baseline_trace = questions[question]
            expected = NEW_FACTS[entity][field][version] if field in SUPPORTED and suite == 'held' else None
            baseline_tokens = [s['token'] for s in baseline_trace]
            for mode in ('prior', 'schema'):
                classification = fields[question]
                selected = pred if mode == 'prior' else classification['selected']
                response, trace, action = baseline_text, baseline_trace, 'default'
                if mode == 'schema' and (classification['confidence'] < cutoff or not classification['order_agrees']):
                    response, trace, action = CLARIFY, [], 'clarify'
                elif selected in SUPPORTED and (mode == 'schema' or prior_confidence >= prior_cutoff):
                    key = (selected, entity, version)
                    if key not in memory:
                        bias = matrix_bias(base.lm_head, site, banks[selected, version],
                                           site.current_basis[entity:entity + 1], basis)
                        memory[key] = None if bias is None else 2 * bias
                    if memory[key] is not None:
                        response, trace = answer(decoder, tokenizer, prompt, style, columns, gate, memory[key])
                        action = 'read'
                    elif mode == 'schema':
                        response, trace, action = MISSING, [], 'missing'
                elif mode == 'schema' and selected != 'ordinary':
                    response, trace, action = MISSING, [], 'missing'
                tokens = [s['token'] for s in trace]
                start = next((i for i, s in enumerate(trace) if s['memory_position'] == 0), None)
                result = dict(suite=suite, mode=mode, question=question, field=field, relation=field,
                              read_relation=selected, entity=entity, version=version, expected=expected,
                              response=response, start=start, action=action, tokens=tokens,
                              opening=tokenizer.decode(tokens[:start], skip_special_tokens=True).strip() if start is not None else '',
                              fact_suffix=tokenizer.decode(tokens[start:], skip_special_tokens=True).strip() if start is not None else '',
                              baseline_equal=response == baseline_text and tokens == baseline_tokens)
                result['strict'] = score(result, all_values)
                rows.append(result)
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            stream.flush()
            if len(rows) % 8 == 0:
                print('Evaluated', len(rows), flush=True)
    if fingerprint(sources) != hashes:
        raise ValueError('Experiment inputs changed')
    summary = summarize(rows)
    write(out / 'result.json', {'rows': rows, 'summary': summary, 'elapsed_seconds': time.perf_counter() - started,
                               'plan_sha256': sha(out / 'plan.json'), 'classification_sha256': sha(out / 'classification.json'),
                               'head_sha256': sha(out / 'attribute-head.pt') if trained else None,
                               'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--trained', action='store_true')
    args = parser.parse_args()
    main(args.out, args.trained)
