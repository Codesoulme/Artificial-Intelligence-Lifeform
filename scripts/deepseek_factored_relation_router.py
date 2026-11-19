"""Matched flat and factored routing with grouped contrast pairs and real generation."""

import argparse
import json
import time
from pathlib import Path

from deepseek_aligned_memory_calibration import NEW_FACTS, load_adapter, select_threshold
from deepseek_aligned_memory_dialogue import (
    OUT as SOURCE_RUN, RELATIONS, RANK, answer, matrix_bias, question_features, score,
)
from deepseek_output_aligned_memory import (
    MODEL, ROOT, SEED, SOURCE, encode, fingerprint, fresh, load_integrated,
    prefix, read, sha, torch, write,
)

OUT = ROOT / 'runs/deepseek-factored-relation-router'
BANK_RUN = ROOT / 'runs/deepseek-aligned-memory-calibration-factorial'
TRAIN = (
    ('drink', '回忆上回的聚餐，他选了哪种喝的？', '下回聚餐该给客人准备哪种喝的？'),
    ('drink', '那份档案里，她平日爱点什么饮料？', '那份档案的饮料栏应该怎么填写？'),
    ('drink', '记得活动结束时，对方说自己爱喝什么吗？', '活动结束后，一般喝什么比较解渴？'),
    ('drink', '请复述这人当时说的饮品喜好。', '请编写一句介绍饮品喜好的示例。'),
    ('drink', '他最中意的是哪一种喝的？', '儿童聚会最适合哪一种喝的？'),
    ('work', '他说自己在那一时期靠哪门手艺谋生？', '哪些手艺比较适合自学后谋生？'),
    ('work', '回顾登记内容，她从事哪类职业？', '登记职业信息有哪些常见分类？'),
    ('work', '记忆里那个人的工作岗位是什么？', '写一段工作岗位介绍作为范例。'),
    ('work', '前面提到他在做哪一行？', '想改行的话，应该先做哪些准备？'),
    ('work', '用一句话说他的职业。', '用一句话说说职业规划的意义。'),
    ('place', '说说那一天那个人去了何处。', '说说适合那一天出游的几个去处。'),
    ('place', '她笔记里写下的到访地是哪儿？', '推荐几种写到访地点的笔记格式。'),
    ('place', '那趟路他们走到了哪个地方？', '他们可以挑哪个地方作为下次旅行的终点？'),
    ('place', '以往那次事件发生在他去了什么地方之后？', '写一段他去了某个地方之后发生的虚构故事。'),
    ('place', '回头看那张记录，他实际到过哪里？', '请查一查网上哪里适合现在到访。'),
)
CALIBRATION = (
    ('drink', '从旧档案找出当事人偏爱的喝法。', '帮当事人设计一杯符合个人偏好的饮料。'),
    ('drink', '有人提过他最想喝什么，你还记得吗？', '如何让聚会客人自己选择想喝什么？'),
    ('work', '告诉我此前那个人是以哪种工作为生的。', '讨论一个人适合以哪种工作为生。'),
    ('work', '查一下那回登记的职业叫什么。', '查一下有哪些常用职业登记表。'),
    ('place', '先前那一次他最终抵达了哪个地点？', '安排一条路线，让他最终抵达一个适合野餐的地点。'),
    ('place', '想想那回见闻中，当事者去了哪里。', '想想能去哪里收集适合写游记的见闻。'),
)
HELD = (
    ('drink', '别猜，按你记下的内容把那位客人的饮品偏好说成完整一句。', '别猜，帮我写一句询问客人饮品偏好的话。'),
    ('drink', '整理那次旧事时，这个人点名爱喝的是哪样东西？', '整理一份点单指南，推荐几种可以点来喝的东西。'),
    ('work', '沿用刚才那段经历里的身份，他当时任职做什么？', '为刚才那个人虚构一个工作身份，写进故事。'),
    ('work', '那份经历里的人凭什么职业挣钱？请答一个整句。', '请说一句话，解释怎样通过职业技能挣钱。'),
    ('place', '回到我们谈的那段事，实际落脚的地点是哪处？', '如果重新安排那趟出行，推荐一处适合落脚的地点。'),
    ('place', '那件事发生时，那位当事人到访了哪个地方来着？', '帮我写一句询问当事人是否到访过某地的话。'),
)
UNSUPPORTED = (
    '这份记录里当事人的年龄是多少？',
    '你还记得那次见面是哪一天吗？',
    '请查记录，这个人当时穿了什么颜色的衣服？',
    '按当时的记录，那处地点的门票多少钱？',
)
SUPPORT_TRAIN = (
    '那杯茶在那次聚餐时花了几块钱？',
    '记录里写了她喝那杯饮料时用的是哪种杯子吗？',
    '那人从事这份工作领多少薪水？',
    '他是何时开始从事那项工作的？',
    '记得那处到访景点的营业时间吗？',
    '当事人当时是搭乘什么交通工具到目的地的？',
    '请复述那天这个人说话时穿的鞋子是什么颜色。',
    '记录里有当事人的电话号码吗？',
)
SUPPORT_CALIBRATION = (
    '上回那位客人在自己喜欢的饮料里加了几勺糖？',
    '那次见面的时候，这位当事人有多少年工龄？',
    '那回抵达目的地以后，他们又逗留了多久？',
)
FRESH_HELD = (
    ('drink', '他喝东西最对胃口的是哪种？照那段回忆说明。', '帮我问他喝什么最对胃口。'),
    ('work', '在旧事里的那个人，平时具体是做哪行的？', '解释一下怎么选择适合自己的行业。'),
    ('place', '那回他们实际到访的地方叫什么？', '给他们的下一次到访推荐一个地方。'),
)
FRESH_UNSUPPORTED = (
    '请说出那个人喝下的饮料有多少毫升。',
    '之前说的那份职业，一周工作多少小时？',
    '按那趟经历的记载，出发地距到访地有多少公里？',
    '那次记录中的人和同伴是什么亲属关系？',
)


def examples(pairs):
    return [(relation, positive) for relation, positive, _ in pairs] + [
        ('ordinary', negative) for _, _, negative in pairs]


def fit(vectors, labels, factored):
    torch.manual_seed(82421)
    head = torch.nn.Linear(vectors.shape[1], len(RELATIONS))
    optimizer = torch.optim.AdamW(head.parameters(), lr=.03, weight_decay=.04)
    needs_memory = labels != len(RELATIONS) - 1
    for _ in range(400):
        scores = head(vectors)
        if factored:
            intent = torch.nn.functional.binary_cross_entropy_with_logits(scores[:, 3], needs_memory.float())
            relation = torch.nn.functional.cross_entropy(scores[needs_memory, :3], labels[needs_memory])
            loss = intent + relation * needs_memory.float().mean()
        else:
            loss = torch.nn.functional.cross_entropy(scores, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return head.eval().requires_grad_(False)


def predict(head, vector, factored):
    logits = head(vector)
    if factored:
        intent = logits[3].sigmoid()
        probabilities = torch.cat((intent * logits[:3].softmax(dim=0), (1 - intent).reshape(1)))
    else:
        probabilities = logits.softmax(dim=0)
    return RELATIONS[int(probabilities.argmax())], float(probabilities.max()), probabilities.tolist()


def summarize(routes, generated, thresholds):
    result = {}
    for name, cutoff in thresholds.items():
        held = [row for row in routes[name] if row['split'] == 'held']
        positive = [row for row in held if row['relation'] != 'ordinary']
        negative = [row for row in held if row['relation'] == 'ordinary']
        accepts = lambda row: row['selected'] != 'ordinary' and row['confidence'] >= cutoff
        rows = [row for row in generated if row['mode'] == name]
        answers = [row for row in rows if row['expected'] is not None]
        controls = [row for row in rows if row['expected'] is None]
        pairs = [[row for row in answers if row['question'] == query['question']] for query in positive]
        result[name] = {
            'threshold': cutoff, 'correct_routes': sum(accepts(row) and row['selected'] == row['relation'] for row in positive),
            'wrong_relation_routes': sum(accepts(row) and row['selected'] != row['relation'] for row in positive),
            'positive_questions': len(positive), 'false_reads': sum(accepts(row) for row in negative),
            'negative_questions': len(negative), 'strict_answers': sum(row['strict'] for row in answers),
            'answers': len(answers), 'strict_pairs': sum(len(pair) == 2 and all(row['strict'] for row in pair) for pair in pairs),
            'pairs': len(pairs), 'control_injections': sum(row['start'] is not None for row in controls),
            'control_token_equal': sum(row['baseline_equal'] for row in controls), 'controls': len(controls),
        }
    return result


def main(out, support_negatives=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if out.exists():
        raise FileExistsError(out)
    if torch.cuda.mem_get_info()[0] < 3.7 * 1024**3:
        raise RuntimeError('Insufficient free GPU memory')
    started = time.perf_counter()
    torch.set_num_threads(2)
    splits = {'train': examples(TRAIN), 'calibration': examples(CALIBRATION),
              'held': examples(HELD) + [('ordinary', question) for question in UNSUPPORTED]}
    if support_negatives:
        splits['train'] += [('ordinary', question) for question in SUPPORT_TRAIN]
        splits['calibration'] += [('ordinary', question) for question in SUPPORT_CALIBRATION]
        splits['held'] += examples(FRESH_HELD) + [('ordinary', question) for question in FRESH_UNSUPPORTED]
    questions = [question for group in splits.values() for _, question in group]
    if len(set(questions)) != len(questions):
        raise ValueError('Overlapping questions')
    candidate_path = SOURCE_RUN / 'candidate.pt'
    banks_path = BANK_RUN / 'banks.pt'
    if sha(candidate_path) != read(SOURCE_RUN / 'result.json')['candidate_sha256']:
        raise ValueError('Candidate digest mismatch')
    if sha(banks_path) != read(BANK_RUN / 'result.json')['banks_sha256']:
        raise ValueError('Bank digest mismatch')
    sources = (SOURCE, candidate_path, banks_path, Path(__file__), MODEL / 'model.safetensors',
               MODEL / 'tokenizer.json', MODEL / 'tokenizer_config.json', MODEL / 'config.json')
    hashes = fingerprint(sources)
    out.mkdir()
    write(out / 'plan.json', {'splits': splits, 'sources': hashes, 'decoder_gain_multiplier': 2.0,
                              'support_negatives': support_negatives,
                              'cutoff_selection': 'minimize wrong reads then maximize correct reads on calibration',
                              'focus_address': '22 by default, 38 for the second question per relation'})
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, dtype=torch.float16,
        attn_implementation='sdpa', device_map={'': 'cuda'}, low_cpu_mem_usage=True).eval()
    decoder, parent, old_state = load_integrated(SOURCE, base, MODEL, individual_device='cpu')
    del parent, old_state
    decoder.eval().requires_grad_(False)
    prompts = {question: encode(tokenizer, prefix(tokenizer, question)) for question in questions}
    features = {question: question_features(decoder, prompts[question]).cpu().clone() for question in questions}
    vectors = torch.stack([features[question] for _, question in splits['train']])
    labels = torch.tensor([RELATIONS.index(relation) for relation, _ in splits['train']])
    flat = fit(vectors, labels, factored=False)
    factored = fit(vectors, labels, factored=True)
    torch.save({'flat': flat.state_dict(), 'factored': factored.state_dict()}, out / 'routes.pt')
    saved = torch.load(out / 'routes.pt', weights_only=True)
    flat.load_state_dict(saved['flat'])
    factored.load_state_dict(saved['factored'])
    routes, thresholds, grids = {}, {}, {}
    for name, head in (('flat', flat), ('factored', factored)):
        routes[name] = []
        for split in ('calibration', 'held'):
            for relation, question in splits[split]:
                selected, confidence, probabilities = predict(head, features[question], name == 'factored')
                routes[name].append(dict(split=split, question=question, relation=relation, selected=selected,
                                         confidence=confidence, probabilities=probabilities))
        thresholds[name], grids[name] = select_threshold([r for r in routes[name] if r['split'] == 'calibration'])
    write(out / 'routing.json', {'routes': routes, 'thresholds': thresholds, 'grids': grids})
    print(json.dumps({'thresholds': thresholds}), flush=True)

    width = base.lm_head.weight.shape[1]
    _, style, gate, columns = load_adapter(torch.load(candidate_path, weights_only=True), width)
    organism, _ = fresh()
    site = organism.memories[0]
    banks = torch.load(banks_path, map_location='cpu', weights_only=True)
    basis = torch.linalg.qr(torch.randn(width, RANK, generator=torch.Generator().manual_seed(SEED))).Q.to('cuda')
    values = {value for facts in NEW_FACTS.values() for versions in facts.values() for value in versions}
    generation, cache, biases = [], {}, {}
    second_questions = {question for index, (_, question, _) in enumerate(HELD) if index % 2 == 1}
    with torch.inference_mode(), (out / 'progress.jsonl').open('x', encoding='utf8') as stream:
        for name, rows in routes.items():
            for row in rows:
                if row['split'] != 'held':
                    continue
                question, relation, selected = row['question'], row['relation'], row['selected']
                prompt = prompts[question]
                entity = 38 if question in second_questions else 22
                if question not in cache:
                    _, cache[question] = answer(decoder, tokenizer, prompt)
                baseline = cache[question]
                active = selected != 'ordinary' and row['confidence'] >= thresholds[name]
                for version in ((0, 1) if relation != 'ordinary' else (1,)):
                    expected = NEW_FACTS[entity][relation][version] if relation != 'ordinary' else None
                    options = None
                    if active:
                        key = (selected, entity, version)
                        if key not in biases:
                            memory_scores = matrix_bias(base.lm_head, site, banks[selected, version],
                                                        site.current_basis[entity:entity + 1], basis)
                            biases[key] = None if memory_scores is None else 2 * memory_scores
                        options = biases[key]
                    trace = baseline
                    if options is not None:
                        _, trace = answer(decoder, tokenizer, prompt, style, columns, gate, options)
                    tokens = [step['token'] for step in trace]
                    start = next((index for index, step in enumerate(trace) if step['memory_position'] == 0), None)
                    opening = tokenizer.decode(tokens[:start], skip_special_tokens=True).strip() if start is not None else ''
                    suffix = tokenizer.decode(tokens[start:], skip_special_tokens=True).strip() if start is not None else ''
                    result = dict(mode=name, question=question, relation=relation, entity=entity, version=version,
                                  read_relation=selected, expected=expected, accepted=active, tokens=tokens,
                                  opening=opening, fact_suffix=suffix, start=start,
                                  response=tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                                  baseline_equal=tokens == [step['token'] for step in baseline])
                    result['strict'] = score(result, values)
                    generation.append(result)
                    stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                stream.flush()
            print('Generated', len(generation), flush=True)
    if fingerprint(sources) != hashes:
        raise ValueError('Source changed during experiment')
    summary = summarize(routes, generation, thresholds)
    cohorts = {'previous': {question for _, question in examples(HELD)} | set(UNSUPPORTED)}
    if support_negatives:
        cohorts['fresh'] = {question for _, question in examples(FRESH_HELD)} | set(FRESH_UNSUPPORTED)
    grouped_summary = {
        cohort: summarize({mode: [row for row in entries if row['question'] in questions]
                           for mode, entries in routes.items()},
                          [row for row in generation if row['question'] in questions], thresholds)
        for cohort, questions in cohorts.items()
    }
    write(out / 'result.json', {'summary': summary, 'rows': generation, 'elapsed_seconds': time.perf_counter() - started,
                               'cohorts': grouped_summary,
                               'routes_sha256': sha(out / 'routes.pt'), 'routing_sha256': sha(out / 'routing.json'),
                               'plan_sha256': sha(out / 'plan.json'), 'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
    print(json.dumps(grouped_summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--support-negatives', action='store_true')
    args = parser.parse_args()
    main(args.out, args.support_negatives)
