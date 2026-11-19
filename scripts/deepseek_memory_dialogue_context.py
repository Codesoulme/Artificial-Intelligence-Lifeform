"""Test context-dependent follow-ups against frozen and dialogue-trained routing."""

import argparse
import json
import time
from pathlib import Path

from deepseek_aligned_memory_calibration import NEW_FACTS, load_adapter
from deepseek_aligned_memory_dialogue import OUT as SOURCE_RUN, RANK, answer, matrix_bias, score
from deepseek_factored_relation_router import BANK_RUN
from deepseek_memory_attribute_router import CLARIFY, MISSING, SUPPORTED, choose_cutoff
from deepseek_memory_attribute_span import KEYS, OUT as SPAN_RUN, Route, batch, fit, route_action
from deepseek_output_aligned_memory import MODEL, ROOT, SEED, SOURCE, fingerprint, fresh, load_integrated, read, sha, torch, write

OUT = ROOT / 'runs/deepseek-memory-dialogue-context'
FIELDS = ('drink', 'work', 'place', 'time', 'amount', 'other')
# Each topic group uses an identical follow-up across fields. The neutral
# assistant turn introduces neither the target relation nor an answer value.
ACK = '好的，你可以接着问。'
TRAIN = (
    ('这件事里具体是什么？', ('刚才说的是那位客人爱点的饮品。', '刚才说的是他在做的工作。', '刚才说的是那趟行程的去处。',
     '刚才说的是见面发生在哪一天。', '刚才说的是饮料花费的金额。', '刚才说的是他的同行者。')),
    ('那你把答案告诉我。', ('我想谈这人平日爱喝什么。', '我想谈这人的职业名称。', '我想谈他到访的地点名称。',
     '我想谈这件事具体发生时刻。', '我想谈那杯东西有多少毫升。', '我想谈对方的衣着颜色。')),
    ('请从记录里说出来。', ('这部分关于当事人的饮品偏好。', '这部分关于那位来客的工作岗位。', '这部分关于他去了哪个地方。',
     '这部分关于他开始工作的年月。', '这部分关于地点收的门票价格。', '这部分关于去那里的原因。')),
    ('然后呢，具体是哪一个？', ('前面提到了她喜欢哪种喝的。', '前面提到了她靠哪门专业谋生。', '前面提到了她最后去了哪里。',
     '前面提到了她去了几天。', '前面提到了她喝了几杯。', '前面提到了她乘坐的交通工具。')),
    ('把那项再说明白一点。', ('要弄清楚他爱喝的东西叫什么。', '要弄清楚那人从事什么职业。', '要弄清楚当时去过的那处地点。',
     '要弄清楚他抵达时是几点钟。', '要弄清楚出发地和到访地的距离。', '要弄清楚两位当事人是什么关系。')),
    ('你还记得是什么吗？', ('那份经历提到了她最喜欢的饮料。', '那份经历提到了她的任职方向。', '那份经历提到了她真实走访的地点。',
     '那份经历提到了她在当地待了多久。', '那份经历提到了她买东西花了多少钱。', '那份经历提到了谁带她去那里。')),
)
CAL = (
    ('请把这个细节说给我。', ('我们说到客人的饮料喜好。', '我们说到那个人的职业身份。', '我们说到记录中的访问地点。',
     '我们说到那天碰面的日期。', '我们说到那杯饮料的价格。', '我们说到当事人选择那里的缘由。')),
    ('嗯，那项是怎么记的？', ('这次想确认他钟爱的饮品叫什么。', '这次想确认他的岗位名称。', '这次想确认她那回到过的地方。',
     '这次想确认故事发生在哪个月。', '这次想确认进场一共付了多少。', '这次想确认她当时穿着什么。')),
)
HELD = (
    ('那一项的答案是什么？', ('那位客人最常点的喝的值得记下来。', '我们接着聊那个人是干哪一行的。', '刚刚的话题是那一趟最后落脚的地方。',
     '我在回想抵达时是哪一日。', '那款饮品的杯子装了多少是这里的重点。', '我们谈的是与他同行者的身份。')),
    ('能再告诉我这个细节吗？', ('把焦点放在她一向偏爱的饮料上。', '现在要回顾她谋生时具体做的事。', '说的是那一回她实际到访的地名。',
     '我只关心她什么时候开始在那里工作。', '这里谈的是入场需要付多少费用。', '我们只想知道她那天坐什么车过去。')),
    ('你记下来的内容具体是什么？', ('接下来只想查他点名最喜欢的可饮用东西。', '这一段问那位当事人任什么职。', '我说的是他的实际行程终点。',
     '话题是从出发到抵达用了多少小时。', '当前关心两处地点隔了多少公里。', '先把他为何要去那里的问题说清楚。')),
)
SWITCH = {'train': (
    ('刚才说的是他最喜欢的饮品。', '换个话题，请写一句问候语。', 'ordinary'),
    ('刚才说的是她的工作岗位。', '不过我现在问的是她去了哪里。', 'place'),
    ('前面聊了到过的地方。', '这次只想确认他爱喝什么东西。', 'drink'),
    ('前面说他要回忆见面日期。', '换个话题，怎样写一份游记？', 'ordinary'),
    ('他最喜欢的饮料被记下来了。', '现在我想知道他以什么职业谋生。', 'work'),
    ('我们在谈她到了哪个目的地。', '请编一个行程作为示例。', 'ordinary'),
    ('他的工作岗位是上一个话题。', '我改问，他那时做了几年？', 'time'),
    ('前面谈的是客人喝的茶。', '帮我推荐适合冬天喝的饮料。', 'ordinary'),
), 'calibration': (
    ('这里说的是工作岗位。', '现在帮我写一段求职开场白。', 'ordinary'),
    ('刚才谈的是他的饮品喜好。', '我想改问他到访过的地方。', 'place'),
), 'held': (
    ('到过的地方是刚才要聊的话题。', '先不说那次了，给我推荐一种写游记的方法。', 'ordinary'),
    ('刚才在聊当事人的职业。', '现在离开那份档案，说说一般怎样准备面试。', 'ordinary'),
    ('要聊的原本是他钟爱的饮料。', '改问那趟经历中他实际抵达的地点。', 'place'),
    ('前面在说她那次去了哪里。', '我现在想问她当时做哪一行。', 'work'),
)}


def samples(groups, switches):
    rows = []
    for index, (question, topics) in enumerate(groups):
        for field, history in zip(FIELDS, topics, strict=True):
            rows.append(dict(key=f'follow-{index}-{field}', kind='follow', field=field,
                             history=history, question=question, target_turn=0))
    for index, (history, question, field) in enumerate(switches):
        rows.append(dict(key=f'switch-{index}', kind='switch', field=field,
                         history=history, question=question, target_turn=1))
    return rows


@torch.no_grad()
def features_for(decoder, tokenizer, row, include_history):
    users = [row['history'], row['question']] if include_history else [row['question']]
    messages = [{'role': 'user', 'content': users[0]}]
    if include_history:
        messages.extend(({'role': 'assistant', 'content': ACK}, {'role': 'user', 'content': users[1]}))
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not rendered.rstrip().endswith('<think>'):
        raise ValueError('Unexpected generation template')
    rendered += '</think>\n\n'
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded['input_ids']
    result = decoder(input_ids=torch.tensor([ids], device='cuda'), use_cache=False, output_hidden_states=True)
    hidden = torch.nn.functional.normalize(result.hidden_states[-1][0].float(), dim=-1).cpu()
    positions, labels, turns = [], [], []
    cursor = 0
    for turn, user in enumerate(users):
        start = rendered.index(user, cursor)
        end = start + len(user)
        for index, (left, right) in enumerate(encoded['offset_mapping']):
            if left < end and right > start and right > left:
                positions.append(index)
                labels.append(not include_history or row['target_turn'] == turn)
                turns.append(turn)
        cursor = end
    if not any(labels):
        raise ValueError('Target turn has no tokens')
    return hidden[-1], hidden[positions], torch.tensor(labels), ids, turns


def route(model, rows, features):
    aliases = [dict(question=row['key']) for row in rows]
    tail, tokens, valid, _ = batch(aliases, features)
    with torch.no_grad():
        logits, attention = model(tail, tokens, valid)
    result = []
    for row, probs, weights in zip(rows, logits.softmax(dim=-1), attention):
        turn = features[row['key']][4][int(weights.argmax())]
        result.append(dict(**row, selected=KEYS[int(probs.argmax())], confidence=float(probs.max()),
                           order_agrees=True, attended_turn=turn, probabilities=probs.tolist()))
    return result


def summarize(routes, rows, cutoff):
    recall = [r for r in routes if r['field'] in SUPPORTED]
    controls = [r for r in routes if r['field'] not in SUPPORTED]
    pairs = [[r for r in rows if r['key'] == route_row['key']] for route_row in recall]
    return dict(cutoff=cutoff, field_correct=sum(r['field'] == r['selected'] for r in routes), questions=len(routes),
                correct_reads=sum(route_action(r, cutoff) == 'read' and r['selected'] == r['field'] for r in recall),
                recall=len(recall), wrong_reads=sum(route_action(r, cutoff) == 'read' and r['selected'] != r['field'] for r in routes),
                controls=len(controls), control_injections=sum(r['field'] not in SUPPORTED and r['start'] is not None for r in rows),
                ordinary_retained=sum(r['field'] == 'ordinary' and r['action'] == 'ordinary' for r in rows),
                missing_explicit=sum(r['field'] not in SUPPORTED and r['field'] != 'ordinary' and r['action'] == 'missing' for r in rows),
                strict=sum(r['strict'] for r in rows), answers=sum(r['field'] in SUPPORTED for r in rows),
                strict_pairs=sum(len(pair) == 2 and all(r['strict'] for r in pair) for pair in pairs))


def main(out):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if out.exists():
        raise FileExistsError(out)
    if torch.cuda.mem_get_info()[0] < 3.7 * 1024**3:
        raise RuntimeError('Insufficient free GPU memory')
    started = time.perf_counter()
    torch.set_num_threads(2)
    splits = {name: samples(groups, SWITCH[name]) for name, groups in (('train', TRAIN), ('calibration', CAL), ('held', HELD))}
    texts = [{(r['history'], r['question']) for r in rows} for rows in splits.values()]
    if any(a & b for i, a in enumerate(texts) for b in texts[i + 1:]):
        raise ValueError('Dialogue overlap across splits')
    candidate_path, banks_path, span_path = SOURCE_RUN / 'candidate.pt', BANK_RUN / 'banks.pt', SPAN_RUN / 'routing.pt'
    for path, expected in ((candidate_path, read(SOURCE_RUN / 'result.json')['candidate_sha256']),
                           (banks_path, read(BANK_RUN / 'result.json')['banks_sha256']),
                           (span_path, read(SPAN_RUN / 'result.json')['artifact_sha256']['routing.pt'])):
        if sha(path) != expected:
            raise ValueError('Source checkpoint changed')
    sources = (Path(__file__), SOURCE, candidate_path, banks_path, span_path,
               *(MODEL / name for name in ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json')))
    hashes = fingerprint(sources)
    values = {value for facts in NEW_FACTS.values() for pair in facts.values() for value in pair}
    if any(value in r['history'] + ACK + r['question'] for rows in splits.values() for r in rows for value in values):
        raise ValueError('A fact value appears in the conversation')
    out.mkdir()
    write(out / 'plan.json', dict(splits=splits, acknowledgement=ACK, source_sha256=hashes, entity=22, gain_multiplier=2.))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, dtype=torch.float16,
        attn_implementation='sdpa', device_map={'': 'cuda'}, low_cpu_mem_usage=True).eval()
    decoder, parent, state = load_integrated(SOURCE, base, MODEL, individual_device='cpu')
    del parent, state
    decoder.eval().requires_grad_(False)
    features = {split: {r['key']: features_for(decoder, tokenizer, r, True) for r in rows} for split, rows in splits.items()}
    no_history_cache = {}
    no_history = {}
    for row in splits['held']:
        if row['question'] not in no_history_cache:
            no_history_cache[row['question']] = features_for(decoder, tokenizer, row, False)
        no_history[row['key']] = no_history_cache[row['question']]
    train = splits['train']
    data = batch([dict(question=r['key']) for r in train], features['train'])
    learned = fit(data, torch.tensor([KEYS.index(r['field']) for r in train]), localized=True)
    torch.save(learned.state_dict(), out / 'router.pt')
    learned.load_state_dict(torch.load(out / 'router.pt', weights_only=True))
    frozen = Route(data[0].shape[-1], localized=True)
    frozen.load_state_dict(torch.load(span_path, weights_only=True)['localized'])
    frozen.eval().requires_grad_(False)
    modes = {'frozen': frozen, 'dialogue': learned}
    routes, cutoffs, grids = {}, {}, {}
    for name, model in modes.items():
        cutoffs[name], grids[name] = choose_cutoff(route(model, splits['calibration'], features['calibration']))
        routes[name] = route(model, splits['held'], features['held'])
    without = route(learned, splits['held'], no_history)
    write(out / 'routes.json', dict(held=routes, removed_history=without, cutoffs=cutoffs, calibration_grids=grids))
    print(json.dumps(cutoffs), flush=True)

    width = base.lm_head.weight.shape[1]
    _, style, gate, columns = load_adapter(torch.load(candidate_path, weights_only=True), width)
    organism, _ = fresh()
    site = organism.memories[0]
    banks = torch.load(banks_path, weights_only=True, map_location='cpu')
    basis = torch.linalg.qr(torch.randn(width, RANK, generator=torch.Generator().manual_seed(SEED))).Q.to('cuda')
    outputs, cache, memory = [], {}, {}
    with torch.inference_mode(), (out / 'progress.jsonl').open('x', encoding='utf8') as log:
        for mode, rows in routes.items():
            for row in rows:
                field, selected = row['field'], row['selected']
                action = route_action(row, cutoffs[mode])
                ids = features['held'][row['key']][3]
                for version in ((0, 1) if field in SUPPORTED else (1,)):
                    expected = NEW_FACTS[22][field][version] if field in SUPPORTED else None
                    tokens, opening, suffix, start = [], '', '', None
                    if action in ('read', 'ordinary'):
                        cache_key = (row['key'], action, selected, version if action == 'read' else None)
                        if cache_key not in cache:
                            if action == 'read':
                                mem_key = (selected, version)
                                if mem_key not in memory:
                                    bias = matrix_bias(base.lm_head, site, banks[mem_key], site.current_basis[22:23], basis)
                                    if bias is None:
                                        raise ValueError('Expected populated matrix')
                                    memory[mem_key] = 2 * bias
                                cache[cache_key] = answer(decoder, tokenizer, ids, style, columns, gate, memory[mem_key])
                            else:
                                cache[cache_key] = answer(decoder, tokenizer, ids)
                        response, trace = cache[cache_key]
                        tokens = [step['token'] for step in trace]
                        start = next((i for i, step in enumerate(trace) if step['memory_position'] == 0), None)
                        if start is not None:
                            opening = tokenizer.decode(tokens[:start], skip_special_tokens=True)
                            suffix = tokenizer.decode(tokens[start:], skip_special_tokens=True).strip()
                    else:
                        response = MISSING if action == 'missing' else CLARIFY
                    generated = dict(mode=mode, key=row['key'], kind=row['kind'], field=field, relation=field,
                                     read_relation=selected, version=version, expected=expected, action=action,
                                     start=start, response=response, tokens=tokens, opening=opening, fact_suffix=suffix)
                    generated['strict'] = score(generated, values)
                    outputs.append(generated)
                    log.write(json.dumps(generated, ensure_ascii=False) + '\n')
                log.flush()
            print('Generated', len(outputs), flush=True)
    if fingerprint(sources) != hashes:
        raise ValueError('Experiment inputs changed')
    summary = {name: summarize(rows, [r for r in outputs if r['mode'] == name], cutoffs[name]) for name, rows in routes.items()}
    follows = [r for r in without if r['kind'] == 'follow']
    summary['removed_history'] = dict(followups=len(follows), reads=sum(route_action(r, cutoffs['dialogue']) == 'read' for r in follows),
                                      distinct_routes=len({(r['question'], r['selected']) for r in follows}))
    write(out / 'result.json', dict(rows=outputs, summary=summary, elapsed_seconds=time.perf_counter() - started,
          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
          artifact_sha256={name: sha(out / name) for name in ('plan.json', 'router.pt', 'routes.json')}))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    main(parser.parse_args().out)
