"""Compare token-level and whole-value memory actions on the same supervision."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_fast_weight_lexical_write import base, compress_dimension, SOURCE, tokens
from scripts.audit_spora_version_coverage import covered_reply, write_fresh
from scripts.audit_spora_versioned_memory_generation import frozen_features, MEMORY_FRACTION, NEUTRAL, read_versions, TRAIN_NAMES
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_ttt_read_adapter_pilot import generate
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_versioned_memory_pointer import VersionMemoryPointer, VersionReplyActions


SEED = 32601
LIMIT = 24
TRAINING = {
    'drinks': ('爱喝的饮品', '喝的是', ('奶茶', '绿豆汤', '冰气泡茶', '乌龙茶')),
    'places': ('订下的见面地点', '见面约在', ('东门外', '高台亭', '阶梯教室', '青柳桥')),
    'items': ('看中的物品', '挑的是', ('毛线帽', '灰色雨伞', '明信片', '蓝色书签')),
}
FORMS = (('只报{name}{field}的名称。', '', ''),
         ('{name}{field}是什么？简短给出名字。', '', '。'),
         ('回忆一下{name}{field}是哪一个，用完整的话说。', '我记得是', '。'),
         ('你能用一句话说清楚{name}{field}是什么吗？', '记下的是', '。'),
         ('核对一下{name}{field}，请写成一句话。', '{stem}', '。'))
EVENTS = {
    'drinks': (('罗凌川', '橘皮红茶', '芋泥豆奶'), ('谭问溪', '桂花乌龙', '紫米米浆')),
    'places': (('贺澄宁', '松风书屋', '白石码头'), ('蔡心栩', '江畔剧场', '鹿鸣园')),
    'items': (('裴彦', '藤纹茶盘', '帆布笔袋'), ('蒋培旭', '铜扣书夹', '方格便签')),
    'books': (('周菀晴', '星海拾光', '山城旧信'), ('梁子牧', '石阶回声', '晚潮来信')),
}
FACTS = {'drinks': '这次{name}选择喝{value}。', 'places': '与{name}会面约在{value}。',
         'items': '这次{name}带走的是{value}。', 'books': '{name}借阅的书是{value}。'}
QUESTIONS = {
    'drinks': ('拿{name}的饮料记录来说，给我品名就好。', '记得{name}那杯常点什么吗？请成句作答。'),
    'places': ('和{name}碰头的地方叫啥？只回答地点名。', '你还记得和{name}说好在哪儿碰面吗？请回答完整。'),
    'items': ('{name}选过哪件用品？单报商品名。', '你帮我回想一下{name}选的是哪件用品，写成一句话。'),
    'books': ('从记录查一下{name}借的书叫什么，答书名即可。', '你能用整句话告诉我{name}借了哪本书吗？'),
}
GRAMMAR = {'drinks': ('喝的是', '记录里喝的是'), 'places': ('见面约在',),
           'items': ('挑的是',), 'books': ('借的是', '借阅的是')}


def training_rows(tokenizer):
    rows = []
    for field, stem, values in TRAINING.values():
        for index, name in enumerate(TRAIN_NAMES):
            for question, before, after in FORMS:
                for value in (values[index % len(values)], values[(index + 1) % len(values)]):
                    before = before.format(stem=stem)
                    start, memory, end = (tokens(tokenizer, text) for text in (before, value, after))
                    tail = [*start, *memory, *end, tokenizer.eos_token_id]
                    if not 1 <= len(memory) <= 4 or tokenizer.decode(tail[:-1]) != before + value + after:
                        raise ValueError('Teaching reply has an incompatible memory span')
                    actions = [*(('text', token) for token in start),
                               *(('copy', index) for index in range(len(memory))),
                               *(('text', token) for token in (*end, tokenizer.eos_token_id))]
                    rows.append((question.format(name=name, field=field), tail, actions, len(memory)))
    return rows


def action_targets(rows, sizes, scaffold):
    piecewise, atomic = [], []
    position = 0
    for (_, _, actions, width), size in zip(rows, sizes, strict=True):
        copied = False
        for offset, (kind, value) in enumerate(actions):
            if kind == 'text':
                piecewise.append((position + offset, width, 4 + scaffold[value]))
                atomic.append((position + offset, copied, 1 + scaffold[value]))
            else:
                piecewise.append((position + offset, width, value))
                if value == 0:
                    atomic.append((position + offset, False, 0))
                copied = True
        position += size
    return {'coverage': torch.tensor(piecewise), 'atomic': torch.tensor(atomic)}


def fit(head, features, targets, steps, mode):
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    generator = torch.Generator().manual_seed(SEED)
    for step in range(steps):
        index, state, labels = targets[torch.randperm(len(targets), generator=generator)[:64]].T
        scores = head(features[index], state.bool() if mode == 'atomic' else state)
        loss = F.cross_entropy(scores, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in (0, steps // 2, steps - 1):
            print(mode, 'fit', step + 1, float(loss.detach()), flush=True)


@torch.inference_mode()
def atomic_reply(model, tokenizer, head, question, memory):
    if not len(memory):
        return {**generate(model, tokenizer, question), 'copy_at': None}
    inputs = closed_think_inputs(tokenizer, question, memory.device)['input_ids']
    eos = tokenizer.eos_token_id
    pieces = []
    past, copy_at = None, None
    while len(pieces) < LIMIT:
        out = model.model(input_ids=inputs, past_key_values=past, use_cache=True)
        past = out.past_key_values
        hidden = out.last_hidden_state[0, -1]
        chosen = head.choose(hidden, model.lm_head(hidden), copied=copy_at is not None,
                             room_for_copy=len(memory) <= LIMIT - len(pieces), memory_fraction=MEMORY_FRACTION)
        if chosen is None:
            copy_at = len(pieces)
            pieces.extend(memory.tolist())
            inputs = memory.unsqueeze(0)
        else:
            pieces.append(chosen)
            if chosen == eos:
                break
            inputs = torch.tensor([[chosen]], device=memory.device)
    return {'answer': tokenizer.decode(pieces, skip_special_tokens=True).strip(), 'token_ids': pieces, 'copy_at': copy_at}


def measure(output, expected, other, domain, short, eos):
    text = output['answer'].rstrip('。')
    prefix = ('我记得是', '记下的是', '记录中是', *GRAMMAR[domain])
    ended = bool(output['token_ids'] and output['token_ids'][-1] == eos)
    full = expected in text and other not in text
    exact = text == expected if short else any(text == stem + expected for stem in prefix)
    return {'value_and_stop': ended and full, 'strict': ended and exact, 'value_once': text.count(expected) == 1}


def scenarios(prior):
    for index, row in enumerate(row for row in prior['cases'] if row['domain'] != 'regression'):
        yield ('regression', row['domain'], row['name'], row['question'], not index % 2,
               [answer['value'] for answer in row['answers']], [answer['coverage'] for answer in row['answers']])
    for domain, events in EVENTS.items():
        for name, old, current in events:
            for index, template in enumerate(QUESTIONS[domain]):
                yield ('fresh', domain, name, template.format(name=name), index == 0, [old, current], None)


def assess(model, tokenizer, heads, versions, prior):
    rows = []
    for split, domain, name, question, short, expected, earlier in scenarios(prior):
        states = versions[name]
        if [tokenizer.decode(state.tolist()) for state in states] != expected:
            raise ValueError(f'The saved versions decoded incorrectly for {name}')
        answers = []
        for index, state in enumerate(states):
            modes = {'coverage': covered_reply(model, tokenizer, heads['coverage'], question, state),
                     'atomic': atomic_reply(model, tokenizer, heads['atomic'], question, state)}
            if earlier is not None:
                modes['prior_coverage'] = earlier[index]
            scores = {mode: measure(output, expected[index], expected[1 - index], domain,
                                    short, tokenizer.eos_token_id) for mode, output in modes.items()}
            answers.append({'value': expected[index], 'outputs': modes, 'scores': scores})
        rows.append({'split': split, 'domain': domain, 'name': name, 'question': question,
                     'short': bool(short), 'answers': answers})
        if len(rows) % 4 == 0:
            print('compared', len(rows), flush=True)
    neutral = []
    empty = torch.empty(0, dtype=torch.long, device='cuda')
    for question in NEUTRAL:
        output = atomic_reply(model, tokenizer, heads['atomic'], question, empty)
        if output.pop('copy_at') is not None or output != generate(model, tokenizer, question):
            raise ValueError('The inactive copy action changed frozen dialogue')
        neutral.append({'question': question, 'output': output})
    return rows, neutral


def summarize(rows):
    groups = {}
    for row in rows:
        split = groups.setdefault(row['split'], {})
        for mode in row['answers'][0]['scores']:
            counts = split.setdefault(mode, Counter())
            counts['pairs'] += 1
            counts['answers'] += 2
            for metric in ('value_and_stop', 'strict', 'value_once'):
                matches = [a['scores'][mode][metric] for a in row['answers']]
                counts[metric] += sum(matches)
                counts[f'pair_{metric}'] += all(matches)
    return groups


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    model_path = base.MODEL / 'model.safetensors'
    earlier_path, prior_path = (args.coverage / name for name in ('versions.pt', 'result.json'))
    adapter = SOURCE / 'adapters.pt'
    paths = (model_path, earlier_path, prior_path, adapter, base.MODEL / 'tokenizer.json', base.MODEL / 'tokenizer_config.json')
    hashes = {str(path): base.sha(path) for path in paths}
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    if (hashes[str(model_path)] != base.WEIGHT_SHA or prior['version_sha256'] != hashes[str(earlier_path)]
            or any(prior['sources'][str(path)] != hashes[str(path)] for path in paths[3:])):
        raise ValueError('Input artifacts differ from the coverage audit')
    result_path, fresh_path, heads_path = (args.output / name for name in ('result.json', 'versions.pt', 'heads.pt'))
    expected = json.loads(result_path.read_text(encoding='utf-8')) if args.verify else None
    if expected is not None and (hashes != expected['sources'] or base.sha(fresh_path) != expected['version_sha256']
                                 or base.sha(heads_path) != expected['heads_sha256']):
        raise ValueError('Reload artifacts changed')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed()
    print('Loaded frozen DeepSeek', flush=True)
    embeddings = model.get_input_embeddings().weight.detach()
    columns = compress_dimension(embeddings.shape[1]).to(embeddings.device)
    codebook = F.normalize(embeddings.index_select(1, columns).float(), dim=-1)
    if not args.verify:
        settings = torch.load(adapter, map_location='cpu', weights_only=True)
        atomic_save(write_fresh(model, tokenizer, columns, layer=settings['layers'][0], eta=settings['eta'],
                                ridge=settings['ridge'], events=EVENTS, facts=FACTS), fresh_path)
    previous = torch.load(earlier_path, map_location='cpu', weights_only=True)
    fresh = torch.load(fresh_path, map_location='cpu', weights_only=True)
    seen = set(TRAIN_NAMES) | set(previous)
    seen_values = {a['value'] for row in prior['cases'] for a in row['answers']} | {
        value for _, _, values in TRAINING.values() for value in values}
    if seen & set(fresh) or any(value in seen_values for rows in EVENTS.values() for _, old, new in rows for value in (old, new)):
        raise ValueError('The fresh split overlaps the training or previous evaluation facts')
    versions = read_versions({**previous, **fresh}, codebook)
    del codebook
    rows = training_rows(tokenizer)
    scaffold = sorted({value for _, _, actions, _ in rows for kind, value in actions if kind == 'text'})
    torch.manual_seed(SEED)
    heads = {'coverage': VersionMemoryPointer(model.config.hidden_size, scaffold),
             'atomic': VersionReplyActions(model.config.hidden_size, scaffold)}
    if args.verify:
        saved = torch.load(heads_path, map_location='cpu', weights_only=True)
        for mode, head in heads.items():
            head.load_state_dict(saved[mode])
    else:
        features, sizes = frozen_features(model, tokenizer, rows)
        features = features.clone()
        targets = action_targets(rows, sizes, {token: index for index, token in enumerate(scaffold)})
        for mode, head in heads.items():
            fit(head, features, targets[mode], args.steps, mode)
        atomic_save({mode: head.state_dict() for mode, head in heads.items()}, heads_path)
    for head in heads.values():
        head.eval().to('cuda')
    outputs, neutral = assess(model, tokenizer, heads, versions, prior)
    if any(base.sha(path) != hashes[str(path)] for path in paths):
        raise ValueError('A protected source changed')
    result = {'sources': hashes, 'version_sha256': base.sha(fresh_path), 'heads_sha256': base.sha(heads_path),
              'steps_per_head': expected['steps_per_head'] if expected else args.steps, 'training_rows': len(rows),
              'summary': summarize(outputs), 'cases': outputs, 'neutral': neutral}
    if expected is not None and any(expected[key] != value for key, value in result.items()):
        raise ValueError('Whole-value replay changed one or more answers')
    print(result['summary'], flush=True)
    store_json(args.output / ('reload.json' if args.verify else 'result.json'),
               {**result, 'seconds': perf_counter() - started, 'peak_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('coverage', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--steps', type=int, default=180)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
