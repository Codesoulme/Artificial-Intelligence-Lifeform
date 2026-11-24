"""Train and test the native decoder's whole-field memory actions."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import random
import re
from string import Formatter
from time import perf_counter

from torch.nn import functional as F

from scripts.audit_spora_reply_coverage import fresh_cases, measures
from scripts.benchmark_spora_recurrent_dialogue import reply, settings_for
from scripts.train_spora_memory_conditioned_reply import (
    DATA, HELD, MODEL, SCENARIOS, SOURCE, TRANSFER_FIELDS, AutoTokenizer, alternate, capture, dialogue_data, evaluate,
    extend_positions, frozen_digest, grade, install, load_full_streamed, load_integrated_runtime,
    load_recurrent_replacements, prompt_ids, read, recall_prompts, records, sha, state_for, store_json, tokens,
    torch, visible_history,
)
from spora.language_training import atomic_save
from spora.research_dialogue_working_state import FIELDS, WorkingTurnState
from spora.research_recurrent_decode import DynamicCache
from spora.research_span_action import SpanActionHead, available_fields, compile_actions, emit_action


SEED = 26028
FRESH = SOURCE.parent / 'working-state-span-transfer-v1.json'
FIELD_NAMES = {
    'day': ('哪天', '日期'), 'time': ('几点', '时间'), 'place': ('在哪儿', '地点'),
    'gear': ('要带什么', '携带物'), 'person': ('由谁负责', '姓名'),
    'quantity': ('数量是多少', '数量'), 'direction': ('怎么走', '方向'),
}
ORDINARY_TRAIN = (
    '请用一句话介绍橘子的味道。', '纸上的水彩怎样晒干？', '解释火车为什么要在铁轨上走。',
    '八加六等于多少？', '月亮为什么看起来会变形？', '把“你来了”翻译成英文。',
    '给我两个不辣的便当配菜。', '不考虑那个安排。什么是质数？',
    '夏天喝温水有什么好处？', '给一个给旧电池分类的方法。', '哪天是星期三的后一日？',
    '图书馆为什么需要保持安静？', '日落后天空为什么还有亮光？', '谁负责给新生登记一般由学校怎么安排？',
    '怎么礼貌地问别人迎新活动几点开始？', '讲一下怎么在陌生城市找路。', '各举一个液体和固体的例子。',
    '一盒有六根笔，抽出两根后剩多少？', '怎么给会议纪要加上标题？', '用怎样的语气向客户说明延误更合适？',
    '不必沿用刚才的地点，请介绍一种常见树木。', '手套和清洁剂一般应该怎么存放？',
    '形容暴雨将至的天空，用七八个字。', '试着写一句问路的通用句子。',
)


def plan(tokenizer, question, history, template, values, state, vocabulary_size):
    segments = []
    for literal, name, _, _ in Formatter().parse(template):
        segments.append((None, tokens(tokenizer, literal)))
        if name:
            segments.append((FIELDS.index(name), tokens(tokenizer, values[name])))
    surface, target = compile_actions(segments, state, vocabulary_size, tokenizer.eos_token_id)
    if tokenizer.decode(surface) != template.format_map(values):
        raise ValueError('Action expansion changed the annotated answer')
    return prompt_ids(tokenizer, history, question), surface, target, state


def plans_for(tokenizer, vocabulary_size, count):
    source = read(DATA)
    rng = random.Random(SEED)
    plans = []
    for index, original in enumerate(dialogue_data(source, held=False, count=count)):
        other = alternate(original, source, rng)
        history = [] if index % 3 == 0 else visible_history((original, other)[index % 2])
        template = source['kinds'][original['kind']]['answer']
        for case in (original, other):
            values = case['current']
            plans.append(plan(tokenizer, case['question'], history, template, values,
                              state_for(tokenizer, values), vocabulary_size))
    training = source['values']['train']
    for index in range(count * 6):
        requested = rng.sample(FIELDS, 1 + index % 3)
        values = {field: rng.choice(training[field]) for field in FIELDS}
        if 'time' in requested and 'day' not in requested and index % 2:
            values['time'] = rng.choice(training['day']) + values['time']
        extra = rng.choice([field for field in FIELDS if field not in requested])
        fields = requested + ([extra] if index % 3 else [])
        first = rng.choice(requested)
        replacement = rng.choice([value for value in training[first] if value != values[first]])
        names = [FIELD_NAMES[field][index % 2] for field in requested]
        if 'gear' in requested and index % 5 == 0:
            names[requested.index('gear')] = '节目曲目'
        question = (f'把当前安排的{"、".join(names)}发我，越简洁越好。',
                    f'我想确认{"和".join(names)}，请把最新情况写好。',
                    f'给我一句能转发的通知，提到{"、".join(names)}。')[index % 3]
        fields_in_answer = ['{' + field + '}' for field in requested]
        template = ('，'.join(fields_in_answer) + '。' if index % 2 else
                    '，'.join(name + '：' + slot for name, slot in zip(names, fields_in_answer, strict=True)) + '。')
        for state_values in (values, {**values, first: replacement}):
            state = state_for(tokenizer, {field: state_values[field] for field in fields})
            plans.append(plan(tokenizer, question, [], template, state_values, state, vocabulary_size))
    return plans


@torch.no_grad()
def training_features(decoder, tokenizer, plans):
    layer = decoder.base.model.layers[-1]
    requests = dict.fromkeys((tuple(prompt + answer), len(answer) + 1) for prompt, answer, _, _ in plans)
    print(f'Capturing {len(requests)} frozen dialogue sequences', flush=True)
    full = dict(zip(requests, capture(decoder, layer, requests), strict=True))
    positives = []
    for prompt, answer, targets, state in plans:
        raw = full[tuple(prompt + answer), len(answer) + 1]
        selected = raw[torch.tensor([target.offset for target in targets])]
        positives.append((selected, targets, state))
    negative = capture(decoder, layer, [(prompt_ids(tokenizer, [], q), 1) for q in ORDINARY_TRAIN])
    return positives, torch.cat(negative)


def fit(base, head, examples, negatives, steps):
    mode_vocab = base.config.vocab_size
    rng = random.Random(SEED)
    order = list(range(len(examples)))
    optimizer = torch.optim.AdamW(head.parameters(), lr=8e-4)
    trace, window = [], []
    negatives = base.model.norm(negatives.to(base.device)).detach()
    for step in range(steps):
        if step % len(order) == 0:
            rng.shuffle(order)
        raw, targets, state = examples[order[step % len(order)]]
        hidden = base.model.norm(raw.to(base.device))
        keys = head.field_keys(state, base.get_input_embeddings())
        allowed = available_fields(state, targets, mode_vocab, base.device)
        scores = head(hidden, keys, allowed, base.lm_head)
        target = torch.tensor([t.action for t in targets], device=base.device)
        selected = scores.gather(-1, target[:, None]).squeeze(-1)
        weights = torch.where(target >= mode_vocab, 2., 1.)
        weights[-1] = 2.
        route_loss = F.softplus(-head.route_logit(hidden[0])) + F.softplus(
            head.route_logit(negatives[step % len(negatives)])
        )
        loss = -(weights * selected).sum() / weights.sum() + .35 * route_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        window.append(float(loss.detach()))
        if (step + 1) % 96 == 0 or step + 1 == steps:
            item = {'step': step + 1, 'loss': sum(window) / len(window)}
            trace.append(item)
            print(item, flush=True)
            window.clear()
    return trace


def backbone_step(base, ids, cache):
    return base.model(input_ids=torch.tensor([ids], device=base.device), past_key_values=cache,
                      use_cache=True).last_hidden_state[0, -1]


@torch.inference_mode()
def generate(decoder, tokenizer, head, prompt, state, *, limit=48, should_read=None):
    base = decoder.base
    budget_vocab = base.config.vocab_size
    cache = DynamicCache(config=base.config)
    hidden = backbone_step(base, prompt, cache)
    remaining = {field for field, value in enumerate(state.slots) if value}
    route = bool(remaining and (head.route_logit(hidden) >= 0. if should_read is None else should_read(hidden, state)))
    keys = head.field_keys(state, base.get_input_embeddings()) if route else None
    surface, reads, eos = [], [], False
    while len(surface) < limit:
        budget = limit - len(surface)
        if route:
            allowed = torch.tensor([i in remaining and len(span) <= budget
                                    for i, span in enumerate(state.slots)], device=base.device)
            action = int(head(hidden, keys, allowed, base.lm_head).argmax())
        else:
            action = int(base.lm_head(hidden).argmax())
        if action == tokenizer.eos_token_id:
            eos = True
            break
        chunk = emit_action(action, state, remaining, budget, budget_vocab)
        if action >= budget_vocab:
            reads.append(FIELDS[action - budget_vocab])
        surface.extend(chunk)
        if len(surface) < limit:
            hidden = backbone_step(base, chunk, cache)
    return {'answer': tokenizer.decode(surface, skip_special_tokens=True).strip(), 'prompt_tokens': len(prompt),
            'generated_tokens': len(surface) + int(eos), 'eos': eos, 'route': route, 'reads': reads}


@torch.inference_mode()
def verify_chunked_cache(decoder, prompt, span):
    base = decoder.base
    chunked, sequential = DynamicCache(config=base.config), DynamicCache(config=base.config)
    backbone_step(base, prompt, chunked)
    backbone_step(base, prompt, sequential)
    chunk_hidden = backbone_step(base, span, chunked)
    for token in span:
        step_hidden = backbone_step(base, (token,), sequential)
    expected = len(prompt) + len(span)
    if any(layer.get_seq_length() != expected for cache in (chunked, sequential) for layer in cache.layers):
        raise ValueError('Whole-span decoding did not append all positions exactly once')
    logits_a, logits_b = base.lm_head(chunk_hidden).float(), base.lm_head(step_hidden).float()
    if int(logits_a.argmax()) != int(logits_b.argmax()):
        raise ValueError('Chunked and token-wise references choose different next tokens')
    return {'prompt': len(prompt), 'span': len(span), 'max_logit_delta': float((logits_a - logits_b).abs().max())}


def scored_cases(decoder, tokenizer, head, held, scopes):
    rows = []
    for i, row in enumerate(held):
        values = row['values']
        other = held[i ^ 1]
        replacement = other['values']['new']
        if other['case'] != row['case'] or replacement in values.values():
            raise ValueError('The paired state must change exactly one field')
        outputs = {}
        for scope in scopes:
            history = row['history'] if scope == 'long' else []
            prompt = prompt_ids(tokenizer, history, row['question'])
            answers = {}
            for variant, update, foil in (('matched', values['new'], replacement),
                                           ('swapped', replacement, values['new'])):
                a, b = TRANSFER_FIELDS[row['case']]
                state = state_for(tokenizer, {a: update, b: values['keep']})
                output = generate(decoder, tokenizer, head, prompt, state)
                forbidden = (values['old'], foil)
                answer = output['answer']
                answers[variant] = {**output, 'new': update in answer, 'keep': values['keep'] in answer,
                                    'foil': any(value in answer for value in forbidden),
                                    'passed': grade(output, [re.escape(update), re.escape(values['keep'])],
                                                    [re.escape(value) for value in forbidden])}
            outputs[scope] = answers
        rows.append({'case': row['case'], 'variant': row['variant'], 'values': values, 'replacement': replacement,
                     'outputs': outputs})
    return rows


def ordinary_probe(decoder, tokenizer, head, questions, state):
    results = []
    for question in questions:
        prompt = prompt_ids(tokenizer, [], question)
        native = reply(decoder, tokenizer, [], question, **settings_for(decoder, tokenizer, 48))
        native.pop('seconds')
        empty = generate(decoder, tokenizer, head, prompt, WorkingTurnState())
        if any(empty[field] != value for field, value in native.items()):
            raise ValueError('No-state action decoding changed native greedy dialogue')
        with_state = generate(decoder, tokenizer, head, prompt, state)
        results.append({'question': question, 'native': native, 'with_state': with_state,
                        'unchanged': all(with_state[field] == value for field, value in native.items())})
    return results


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    inputs = (args.checkpoint, DATA, SOURCE, SCENARIOS, FRESH, args.reference,
              *(folder / name for folder in args.candidates for name in ('attention.pt', 'comparison.json')))
    protected = {path: sha(path) for path in inputs}
    source_digests = {str(path): digest for path, digest in protected.items()}
    reference = read(args.reference)
    baseline_digests = {Path(path).resolve(): digest for path, digest in reference['sources'].items()}
    if any(baseline_digests.get(path.resolve()) != digest for path, digest in protected.items()
           if path != args.reference):
        raise ValueError('The baseline was evaluated on different model or prompt inputs')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    source = read(SOURCE)
    scenarios = {**source, 'recall': read(SCENARIOS)['recall']}
    held = [row for row in recall_prompts(tokenizer, scenarios, extended=True) if row['length'] == 'extended']
    fresh = fresh_cases(FRESH)
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    torch.cuda.empty_cache()
    base.lm_head.to(base.device)
    individual = frozen_digest(runtime.organism, runtime.state)
    if individual != reference['individual_sha256']:
        raise ValueError('The individual differs from the previous audit')
    candidates = load_recurrent_replacements(decoder, args.candidates, protected[args.checkpoint])
    head = SpanActionHead(base.config.hidden_size).to(base.device)
    weight_path = args.output / 'actions.pt'
    if args.verify:
        prior = read(args.output / 'result.json')
        if (prior['sources'] != source_digests or prior['individual_sha256'] != individual
                or sha(weight_path) != prior['sha256']):
            raise ValueError('Reload input or candidate digest differs')
        head.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True))
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        install(stack, candidates)
        cache_check = verify_chunked_cache(decoder, prompt_ids(tokenizer, held[0]['history'], held[0]['question']),
                                          tokens(tokenizer, held[0]['values']['new']))
        if not args.verify:
            plans = plans_for(tokenizer, base.config.vocab_size, args.count)
            positives, negatives = training_features(decoder, tokenizer, plans)
            trace = fit(base, head, positives, negatives, args.steps)
            atomic_save({key: value.cpu() for key, value in head.state_dict().items()}, weight_path)
        head.eval().requires_grad_(False)
        condition = {}
        for name, prompts, scopes in (('reused', held, ('long', 'state_only')), ('fresh', fresh, ('state_only',))):
            rows = scored_cases(decoder, tokenizer, head, prompts, scopes)
            condition[name] = {'rows': rows, 'summary': measures(rows)}
            print(name, condition[name]['summary'], flush=True)
        ordinary = ordinary_probe(decoder, tokenizer, head, [case['turns'][0][0] for case in source['open']],
                                  state_for(tokenizer, {'place': fresh[0]['values']['keep']}))
        memory = evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))
    if (memory != reference['memory'] or frozen_digest(runtime.organism, runtime.state) != individual
            or any(sha(path) != digest for path, digest in protected.items())):
        raise ValueError('Experiment input or persistent memory changed')
    core = {'conditions': condition, 'ordinary': ordinary, 'memory': memory, 'cache_check': cache_check}
    elapsed = perf_counter() - started
    if args.verify:
        if any(prior[key] != value for key, value in core.items()):
            raise ValueError('The action adapter did not reproduce its outputs')
        store_json(args.output / 'reload.json', {'seconds': elapsed, 'sources': source_digests, 'sha256': sha(weight_path)})
    else:
        store_json(args.output / 'result.json', {**core, 'sources': source_digests, 'individual_sha256': individual,
                                                'sha256': sha(weight_path), 'trace': trace, 'seconds': elapsed,
                                                'training_examples': len(plans), 'steps': args.steps})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('candidates', type=Path, nargs=2)
    parser.add_argument('--steps', type=int, default=768)
    parser.add_argument('--count', type=int, default=20, help='Training pairs per base task family')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.steps < 1 or args.count < 3 or (not args.verify and args.output.exists()):
        parser.error('Choose a positive budget and a new output directory')
    main(args)
