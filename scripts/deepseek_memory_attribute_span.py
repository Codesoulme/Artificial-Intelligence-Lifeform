"""Compare question-tail routing with learned attribute localization."""

import argparse
import json
import time
from pathlib import Path

from deepseek_aligned_memory_calibration import NEW_FACTS, load_adapter
from deepseek_aligned_memory_dialogue import OUT as SOURCE_RUN, RANK, answer, matrix_bias, score
from deepseek_factored_relation_router import BANK_RUN
from deepseek_memory_attribute_router import CLARIFY, FIELDS, MISSING, SUPPORTED, choose_cutoff
from deepseek_output_aligned_memory import MODEL, ROOT, SEED, SOURCE, fingerprint, fresh, load_integrated
from deepseek_output_aligned_memory import prefix, read, sha, torch, write

OUT = ROOT / 'runs/deepseek-memory-attribute-span'
KEYS = tuple(FIELDS)
# Every stem is tested with different requested attributes, including a non-recall request.
TRAIN = (
    ('看完那份旧档案，我想知道', ('他爱喝的饮品名', '他实际的职位', '他去的地方叫什么',
      '他动身是哪一天', '那杯饮料价值多少', '谁陪他同行', '如何把这类档案归类')),
    ('围绕那次遇见的人，请说说', ('她最常喝哪种饮料', '她当时做什么工作', '她到访的是哪里',
      '她离开时是几点', '那趟行程花了多少钱', '她穿的衣服是什么样', '如何虚构一个类似人物')),
    ('当时那件事，还有个疑问。', ('他偏爱的饮料是哪款', '他的职业是什么', '他最终去了什么地点',
      '事情发生在何时', '那里离出发点多远', '他为何做出那个决定', '请解释怎么写一份游记')),
    ('说到当事人的记录，请说明', ('当事人喜欢喝哪样东西', '当事人以什么工作为生', '当事人到过哪个地点',
      '当事人何时到那里', '对方喝了多少毫升', '两个人是什么关系', '如何整理人物档案')),
    ('回到刚才谈的那个人。', ('这个人平时最爱喝什么', '这个人具体干哪行', '这个人那次到的地方是哪',
      '这人工作了多久', '那份工作的报酬是多少', '那时他坐什么车', '帮我写一句评价他人职业的示例')),
    ('刚才提到那段往事，', ('里面说她喜欢哪种饮品', '里面记了她是什么职业', '里面的实际到访地是哪',
      '她在那里待了几天', '那里的门票多少钱', '他和同伴怎么认识的', '给我一段虚构的续写')),
    ('那次交流让人好奇，', ('他最想点的饮品是什么', '他当时担任哪个岗位', '他们那天去了哪里',
      '他是哪一年开始工作的', '这人点了几杯', '他为何选择那个目的地', '谈谈如何规划一次旅行')),
)
CAL = (
    ('对那位客人的经历我再确认一下，', ('最合他口味的饮料是啥', '他靠哪门工作谋生', '那趟出行最终到哪儿',
      '到那里时是几点钟', '去那里一共走了多少公里', '他同谁一起去的', '能否给我三条接待建议')),
    ('这段旧记录里涉及吃喝和出行，我要问的是', ('那人喜欢喝的东西叫什么', '那个人具体的任职名称',
      '她实际去过的那处地点', '那次访问持续了多长时间', '这杯饮品能装多少', '她过去是因为什么原因到访',
      '从零开始写一篇出游故事的方法')),
    ('把前面那个人说清楚。', ('说下他最中意的饮料', '说下他平时做的职业', '说下那回去的是哪个地方',
      '说下是哪一天去的', '说下在那里付了多少费用', '说下到访时穿了什么', '请编一小段问候语')),
)
HELD = (
    ('那位曾经提到饮料、工作和行程的客人，我现在想确认', ('哪款喝的最得他心意', '他任的到底是什么职',
      '他的行程终点地名', '他抵达目的地那天是星期几', '那处地点要交几元入场费',
      '谁向他介绍了那里', '怎样组织一次让人愉快的聚会')),
    ('从我们说起的那次相遇出发，再说说', ('他喝的东西里最中意哪样', '他平日靠做哪行挣钱',
      '他们当时具体在何处落脚', '相遇是哪年哪月', '他在餐馆的花销数额',
      '对方当时身穿哪种外套', '如何为故事想象一个新的结尾')),
    ('旧档案里的那个人，此处我关心的是', ('他钟爱的饮料是哪类', '他具体做的营生',
      '他的实际到访地点', '他是在几月开始那份工作的', '那杯东西标了几毫升',
      '同行者的身份', '档案格式应该如何设计')),
    ('关于前面那次实际经历，请回答，', ('他最偏好的可饮用东西叫什么', '他实际执业的岗位名称',
      '他抵达的那处地方叫什么', '从动身到抵达历经几小时', '两地间距是几里',
      '他是出于何种缘由而去', '推荐一条适合下个假期的旅游路线')),
)


def examples(groups):
    rows = []
    for group, (stem, slots) in enumerate(groups):
        if len(slots) != len(KEYS):
            raise ValueError('Each group must contain every field exactly once')
        for key, span in zip(KEYS, slots):
            rows.append(dict(field=key, question=stem + span + '？', span=span,
                             span_start=len(stem), group=group))
    return rows


@torch.no_grad()
def extract(decoder, tokenizer, row):
    text = prefix(tokenizer, row['question'])
    question_start = text.index(row['question'])
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    tokens = torch.tensor([encoded['input_ids']], device='cuda')
    output = decoder(input_ids=tokens, output_hidden_states=True, use_cache=False)
    hidden = torch.nn.functional.normalize(output.hidden_states[-1][0].float(), dim=-1).cpu()
    question_end = question_start + len(row['question'])
    span_start = question_start + row['span_start']
    span_end = span_start + len(row['span'])
    indices, target = [], []
    for i, (start, end) in enumerate(encoded['offset_mapping']):
        if start < question_end and end > question_start and end > start:
            indices.append(i)
            target.append(start < span_end and end > span_start)
    if not target or not any(target):
        raise ValueError('Tokenization did not map the labeled question region')
    offsets = [(max(0, encoded['offset_mapping'][i][0] - question_start),
                min(len(row['question']), encoded['offset_mapping'][i][1] - question_start)) for i in indices]
    return hidden[-1], hidden[indices], torch.tensor(target), encoded['input_ids'], offsets


class Route(torch.nn.Module):
    def __init__(self, width, localized):
        super().__init__()
        self.field = torch.nn.Linear(width, len(KEYS))
        self.locator = torch.nn.Linear(width, 1) if localized else None

    def forward(self, tail, tokens, valid):
        if self.locator is None:
            return self.field(tail), None
        attention = self.locator(tokens).squeeze(-1).masked_fill(~valid, -torch.inf).log_softmax(dim=-1)
        pooled = (attention.exp().unsqueeze(-1) * tokens).sum(dim=1)
        return self.field(pooled), attention


def batch(rows, features):
    entries = [features[r['question']] for r in rows]
    tails = torch.stack([x[0] for x in entries])
    tokens = torch.nn.utils.rnn.pad_sequence([x[1] for x in entries], batch_first=True)
    target = torch.nn.utils.rnn.pad_sequence([x[2] for x in entries], batch_first=True)
    lengths = torch.tensor([len(x[1]) for x in entries])
    valid = torch.arange(tokens.shape[1])[None, :] < lengths[:, None]
    return tails, tokens, valid, target


def fit(data, targets, localized):
    tail, tokens, valid, spans = data
    torch.manual_seed(82431)
    model = Route(tail.shape[-1], localized)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.025, weight_decay=.04)
    for _ in range(400):
        logits, attention = model(tail, tokens, valid)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        if attention is not None:
            log_span_mass = torch.logsumexp(attention.masked_fill(~spans, -torch.inf), dim=1)
            loss -= .5 * log_span_mass.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval().requires_grad_(False)


@torch.no_grad()
def predictions(model, rows, features):
    tail, tokens, valid, _ = batch(rows, features)
    logits, attention = model(tail, tokens, valid)
    result = []
    for index, (row, probs) in enumerate(zip(rows, logits.softmax(dim=-1))):
        selected = KEYS[int(probs.argmax())]
        offsets = features[row['question']][4]
        focus = offsets[int(attention[index].argmax())] if attention is not None else None
        result.append(dict(**row, selected=selected, confidence=float(probs.max()), order_agrees=True,
                           probabilities=probs.tolist(), focus=None if focus is None else
                           row['question'][focus[0]:focus[1]],
                           focus_hits_span=(focus is not None and focus[0] < row['span_start'] + len(row['span'])
                                            and focus[1] > row['span_start'])))
    return result


def route_action(row, cutoff):
    if row['confidence'] < cutoff:
        return 'clarify'
    return 'read' if row['selected'] in SUPPORTED else 'ordinary' if row['selected'] == 'ordinary' else 'missing'


def metrics(routes, rows, cutoff):
    supported = [r for r in routes if r['field'] in SUPPORTED]
    controls = [r for r in routes if r['field'] not in SUPPORTED]
    present = [r for r in rows if r['expected'] is not None]
    missing = [r for r in rows if r['expected'] is None and r['field'] != 'ordinary']
    ordinary = [r for r in rows if r['field'] == 'ordinary']
    pairs = [[r for r in present if r['question'] == query['question']] for query in supported]
    return dict(cutoff=cutoff, labels_correct=sum(r['selected'] == r['field'] for r in routes),
                total=len(routes), focus_hits_span=sum(r['focus_hits_span'] for r in routes),
                correct_reads=sum(route_action(r, cutoff) == 'read' and r['selected'] == r['field'] for r in supported),
                supported=len(supported), false_reads=sum(route_action(r, cutoff) == 'read' for r in controls),
                controls=len(controls), strict=sum(r['strict'] for r in present), present=len(present),
                pairs_correct=sum(len(pair) == 2 and all(r['strict'] for r in pair) for pair in pairs),
                missing_injected=sum(r['start'] is not None for r in missing),
                missing_explicit=sum(r['action'] == 'missing' for r in missing), missing=len(missing),
                ordinary_unchanged=sum(r['action'] == 'ordinary' for r in ordinary), ordinary=len(ordinary))


def main(out):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if out.exists():
        raise FileExistsError(out)
    if torch.cuda.mem_get_info()[0] < 3.7 * 1024**3:
        raise RuntimeError('Insufficient free GPU memory')
    torch.set_num_threads(2)
    started = time.perf_counter()
    splits = {name: examples(groups) for name, groups in (('train', TRAIN), ('calibration', CAL), ('held', HELD))}
    rows = sum(splits.values(), [])
    if len({r['question'] for r in rows}) != len(rows):
        raise ValueError('Duplicate question across splits')
    candidate_path, banks_path = SOURCE_RUN / 'candidate.pt', BANK_RUN / 'banks.pt'
    if sha(candidate_path) != read(SOURCE_RUN / 'result.json')['candidate_sha256']:
        raise ValueError('Adapter changed')
    if sha(banks_path) != read(BANK_RUN / 'result.json')['banks_sha256']:
        raise ValueError('Banks changed')
    sources = (Path(__file__), SOURCE, candidate_path, banks_path, MODEL / 'model.safetensors',
               MODEL / 'config.json', MODEL / 'tokenizer.json', MODEL / 'tokenizer_config.json')
    hashes = fingerprint(sources)
    out.mkdir()
    write(out / 'plan.json', {'source_sha256': hashes, 'splits': splits, 'focus': 22,
                              'gain_multiplier': 2., 'keys': KEYS, 'seed': 82431})
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
              dtype=torch.float16, attn_implementation='sdpa', device_map={'': 'cuda'}, low_cpu_mem_usage=True).eval()
    decoder, parent, state = load_integrated(SOURCE, base, MODEL, individual_device='cpu')
    del parent, state
    decoder.eval().requires_grad_(False)
    features = {row['question']: extract(decoder, tokenizer, row) for row in rows}
    data = batch(splits['train'], features)
    labels = torch.tensor([KEYS.index(r['field']) for r in splits['train']])
    models = {name: fit(data, labels, localized) for name, localized in (('tail', False), ('localized', True))}
    torch.save({name: model.state_dict() for name, model in models.items()}, out / 'routing.pt')
    saved = torch.load(out / 'routing.pt', weights_only=True)
    routes, thresholds, grids = {}, {}, {}
    for name, model in models.items():
        model.load_state_dict(saved[name])
        thresholds[name], grids[name] = choose_cutoff(predictions(model, splits['calibration'], features))
        routes[name] = predictions(model, splits['held'], features)
    write(out / 'routes.json', dict(held=routes, thresholds=thresholds, grids=grids))
    print(json.dumps(thresholds), flush=True)

    width = base.lm_head.weight.shape[1]
    _, style, gate, columns = load_adapter(torch.load(candidate_path, weights_only=True), width)
    organism, _ = fresh()
    site = organism.memories[0]
    banks = torch.load(banks_path, weights_only=True, map_location='cpu')
    basis = torch.linalg.qr(torch.randn(width, RANK, generator=torch.Generator().manual_seed(SEED))).Q.to('cuda')
    values = {v for entity in NEW_FACTS.values() for pair in entity.values() for v in pair}
    outputs, cache, memory = [], {}, {}
    with torch.inference_mode(), (out / 'progress.jsonl').open('x', encoding='utf8') as log:
        for mode, questions in routes.items():
            for row in questions:
                field, question, selected = row['field'], row['question'], row['selected']
                prompt = features[question][3]
                action = route_action(row, thresholds[mode])
                for version in ((0, 1) if field in SUPPORTED else (1,)):
                    expected = NEW_FACTS[22][field][version] if field in SUPPORTED else None
                    start = None
                    tokens = []
                    if action in ('read', 'ordinary'):
                        key = (question, action, selected, version if action == 'read' else None)
                        if key not in cache:
                            if action == 'read':
                                if (selected, version) not in memory:
                                    bias = matrix_bias(base.lm_head, site, banks[selected, version],
                                                       site.current_basis[22:23], basis)
                                    if bias is None:
                                        raise ValueError('Expected populated bank')
                                    memory[selected, version] = 2 * bias
                                cache[key] = answer(decoder, tokenizer, prompt, style, columns, gate, memory[selected, version])
                            else:
                                cache[key] = answer(decoder, tokenizer, prompt)
                        response, trace = cache[key]
                        tokens = [step['token'] for step in trace]
                        start = next((i for i, step in enumerate(trace) if step['memory_position'] == 0), None)
                    else:
                        response = MISSING if action == 'missing' else CLARIFY
                    result = dict(mode=mode, question=question, field=field, relation=field, read_relation=selected,
                                  version=version, expected=expected, action=action, start=start, response=response,
                                  tokens=tokens, opening=tokenizer.decode(tokens[:start], skip_special_tokens=True) if start is not None else '',
                                  fact_suffix=tokenizer.decode(tokens[start:], skip_special_tokens=True).strip() if start is not None else '')
                    result['strict'] = score(result, values)
                    outputs.append(result)
                    log.write(json.dumps(result, ensure_ascii=False) + '\n')
                log.flush()
            print('Generated', len(outputs), flush=True)
    if fingerprint(sources) != hashes:
        raise ValueError('Experiment inputs changed')
    summary = {name: metrics(routes[name], [r for r in outputs if r['mode'] == name], thresholds[name]) for name in routes}
    write(out / 'result.json', dict(rows=outputs, summary=summary, elapsed_seconds=time.perf_counter() - started,
          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
          artifact_sha256={name: sha(out / name) for name in ('plan.json', 'routes.json', 'routing.pt')}))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    main(parser.parse_args().out)
