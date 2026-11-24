"""Compare paired revision intent and field routing on the frozen SPORA backbone."""

import argparse
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
import random
from time import perf_counter

from torch.nn import functional as F

from scripts.audit_spora_clause_intent import clause_features
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.prepare_spora_revision_samples import dialogue_samples
from scripts.train_spora_observed_episodes import (
    DATA, DESIGN, EPISODES, EXTRA_NAMES, FIELDS, HELD, MODEL, OBSERVATIONS, ROLE_NAMES, AutoTokenizer,
    MissingFieldActions, ObservationReadGate, PairedFieldScopeHead, SpanActionHead, TokenFieldAccessHead,
    audit, episodes, evaluate, extend_positions, frozen_digest, held_episodes, install, load_full_streamed,
    load_integrated_runtime, load_recurrent_replacements, read, records, scope_texts, sha, store_json, summarize,
    text_features, torch, training_data, training_examples,
)
from spora.language_training import atomic_save
from spora.research_clause_intent import ClauseIntent
from spora.research_dialogue_working_state import JointTurnValueWriter, characters


SEED = 26176
INTENTS = ('confirmed', 'pending', 'retain')
MODES = ('independent', 'contrast')
TRAIN = (
    ('{field}有新决定，{old}不用了，正式定成{value}。',
     '{field}有人建议由{old}变成{value}，还得最后批准。',
     '{field}仍用{old}，不执行换到{value}的建议。'),
    ('把{field}由{old}改作{value}，我已经核实，可以提交。',
     '把{field}由{old}改作{value}是草稿，先等核实。',
     '撤下把{field}由{old}改作{value}的那条通知。'),
    ('{value}已经选定，作为新的{field}替换{old}。',
     '{value}暂作{field}的备选，用不用它替换{old}还未决定。',
     '{value}不作为新的{field}，维持{old}。'),
    ('原来的{field}是{old}。现在用{value}，这项决定已生效。',
     '原来的{field}是{old}。可能用{value}，之后再定。',
     '原来的{field}是{old}。有人提到{value}，我决定保持原样。'),
    ('刚定下{field}要更新。请写{value}，旧记录{old}失效。',
     '正在讨论{field}要不要更新，{value}和{old}都还在考虑。',
     '关于{field}不做更新。把考虑过的{value}搁置，保留{old}。'),
    ('{field}的新值是{value}，替换{old}的决定请执行。',
     '讨论里提到{field}的新值是{value}，是否替换{old}仍待确认。',
     '听说要把{field}改到{value}，我撤回提案，还是{old}。'),
)
CALIBRATION = (
    ('核对后认定{field}应从{old}更为{value}，据此登记。',
     '正在核对{field}可否从{old}更为{value}，暂时不登记。',
     '核对结果是{field}不从{old}更为{value}，照现有记录。'),
    ('{field}现在确认采用{value}，原来的{old}请更新。',
     '{field}现在提议采用{value}，原来的{old}保持到确认时。',
     '{field}原来是{old}，那个采用{value}的安排已取消。'),
)
TEST = (
    ('原单上的{field}是{old}。我刚拍板，最终按{value}办。',
     '原单上的{field}是{old}。我还没拍板，{value}只是其中一个方案。',
     '原单上的{field}是{old}。之前说的{value}收回，最终不动。'),
    ('关于{field}，这次别照着{old}做了，最后核实为{value}。',
     '关于{field}，要不要把{old}换成{value}，我需要先问一下。',
     '关于{field}，之前提的{value}取消处理，照原来{old}执行。'),
)


def examples(rows, topics, labels, patterns):
    groups = []
    for i, row in enumerate(rows):
        domain = row['topic']
        field, value = next(iter(row['change'].items()))
        spoken = labels[domain][topics[domain].index(field)]
        prefix = ('', f'说的是{spoken}这一项。', '我补充一下这份安排。')[i % 3]
        lines = [prefix + pattern.format(field=spoken, value=value, old=row['initial'][field])
                 for pattern in patterns[i % len(patterns)]]
        if len(set(lines)) != 3 or any(line.count(value) != 1 or line.count(row['initial'][field]) != 1
                                        for line in lines):
            raise ValueError('Matched scenarios must contain the same two values exactly once')
        groups.append({'field': field, 'candidate': value, 'old': row['initial'][field], 'texts': lines})
    return groups


def training_groups(source, specimens, fixture, names, *, calibration=False):
    values = source['values']['held' if calibration else 'train']
    rows, _ = dialogue_samples(specimens, fixture, names, values, calibration=calibration)
    domains = {row['case']: row['fields'] for row in specimens}
    counts = 6 if calibration else 24
    rows = [{**row, 'topic': spec['case']} for spec, chunk in zip(specimens, range(0, len(rows), counts), strict=True)
            for row in rows[chunk:chunk + counts]]
    return examples(rows, domains, names, CALIBRATION if calibration else TRAIN)


def group_texts(groups):
    return {text for group in groups for text in group['texts']}


def features_for(featurize, train, calibration, held, raw, width):
    unique = sorted(group_texts(train) | group_texts(calibration) | group_texts(held) | set(raw))
    hidden, masks = clause_features(featurize, [{'text': text} for text in unique], width)
    return {text: (vector, mask) for text, vector, mask in zip(unique, hidden, masks, strict=True)}


def packed(features, items):
    hidden, mask = zip(*(features[text] for text in items), strict=True)
    return torch.stack(hidden), torch.stack(mask)


def fit(mode, groups, features, width):
    torch.manual_seed(SEED)
    model = ClauseIntent(width, recurrent=True, classes=len(INTENTS) + len(FIELDS))
    batches = torch.Generator().manual_seed(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.02)
    truth = torch.arange(3).repeat(16)
    fields = torch.tensor([FIELDS.index(row['field']) for row in groups])
    for epoch in range(12):
        for indices in torch.randperm(len(groups), generator=batches).split(16):
            text = [line for i in indices.tolist() for line in groups[i]['texts']]
            logits = model(*packed(features, text))
            intent, field = logits.split((len(INTENTS), len(FIELDS)), dim=-1)
            score = intent[:, 0] - intent[:, 1:].logsumexp(dim=-1)
            score = score.reshape(-1, 3)
            loss = (F.cross_entropy(intent, truth[:len(text)])
                    + F.cross_entropy(field, fields[indices].repeat_interleave(3)))
            if mode == 'contrast':
                loss = loss + .5 * F.softplus(1. - score[:, :1] + score[:, 1:]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if epoch in (3, 7, 11):
            print(mode, 'epoch', epoch + 1, round(float(loss.detach()), 5), flush=True)
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def predictions(model, features, texts):
    intent, field = model(*packed(features, texts)).split((len(INTENTS), len(FIELDS)), dim=-1)
    score = intent[:, 0] - intent[:, 1:].logsumexp(dim=-1)
    return [{'score': float(margin), 'intent': INTENTS[int(action)], 'field': FIELDS[int(slot)]}
            for margin, action, slot in zip(score, intent.argmax(dim=-1), field.argmax(dim=-1), strict=True)]


def assess(model, features, groups, floor=None):
    answers = predictions(model, features, [line for group in groups for line in group['texts']])
    rows = [{**group, 'answers': answers[i * 3:i * 3 + 3]} for i, group in enumerate(groups)]
    if floor is None:
        floor = max(answer['score'] for row in rows for answer in row['answers'][1:])
    result = Counter()
    for row in rows:
        first, pending, keep = row['answers']
        result['ordered'] += first['score'] > max(pending['score'], keep['score'])
        for intent, answer in zip(INTENTS, row['answers'], strict=True):
            answer['allowed'] = answer['score'] > floor
            result[intent + '_classified'] += answer['intent'] == intent
            result[intent + '_write'] += answer['allowed']
            result[intent + '_field'] += answer['field'] == row['field']
        result['all_three_gates'] += first['allowed'] and not pending['allowed'] and not keep['allowed']
        result['exact_confirmed_field'] += first['allowed'] and first['field'] == row['field']
    return {'floor': floor, 'summary': dict(result), 'rows': rows}


@torch.inference_mode()
def value_span(writer, features, text, field):
    device = writer.embedding.weight.device
    scores = writer(characters([text]).to(device), features([text]).to(device))[0, FIELDS.index(field), 1:]
    start, end = divmod(int(scores.argmax()), len(text))
    return text[start:end + 1]


def apply_to_turns(model, encoded, writer, featurize, cases, original, floor):
    scored = predictions(model, encoded, [case['update'] for case in cases])
    writes, trace = {}, []
    for case, prediction in zip(cases, scored, strict=True):
        key = case['key']
        old = original[key]
        slot, expected = next(iter(case['change'].items()))
        proposed = value_span(writer, featurize, case['update'], prediction['field'])
        allow = prediction['score'] > floor
        update = {prediction['field']: proposed} if allow else {}
        writes[key] = [old[0], update, *old[2:]]
        trace.append({'key': key, 'prediction': prediction, 'allowed': allow, 'proposed': proposed, 'write': update,
                      'correct_field': prediction['field'] == slot, 'expected': expected,
                      'exact': update == case['change'],
                      'oracle_span': value_span(writer, featurize, case['update'], slot)})
    return writes, trace


def check_matched_writes(report, writer, featurize):
    outcomes = []
    metrics = Counter()
    for row in report['rows']:
        written = []
        for intent, text, forecast in zip(INTENTS, row['texts'], row['answers'], strict=True):
            patch = {}
            if forecast['allowed']:
                field = forecast['field']
                patch[field] = value_span(writer, featurize, text, field)
            written.append(patch)
            metrics[intent + '_wrote'] += bool(patch)
        positive = written[0] == {row['field']: row['candidate']}
        metrics['exact_confirmed'] += positive
        metrics['exact_three_writes'] += positive and not written[1] and not written[2]
        outcomes.append({'field': row['field'], 'expected': row['candidate'], 'writes': written})
    return {'summary': dict(metrics), 'rows': outcomes}


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=args.verify or args.resume_heads)
    study = args.study
    prior = read(study / 'result.json')
    observed = args.observed
    episode_result = read(observed / 'result.json')
    original_paths = {Path(name).resolve(): digest for name, digest in prior['sources'].items()}
    if sha(args.checkpoint) != original_paths.get(args.checkpoint.resolve()):
        raise ValueError('The base differs from the previous revision comparison')
    paths = [args.checkpoint, DATA, DESIGN, EPISODES, OBSERVATIONS, study / 'result.json',
             study / 'revision-writers.pt', observed / 'result.json', observed / 'observation-weights.pt',
             *(directory / name for directory, name in
               ((args.actions, 'actions.pt'), (args.gate, 'gate.pt'), (args.typed, 'typed-fields.pt'),
                (args.scope, 'scope.pt'))),
             *(layer / name for layer in args.layers for name in ('attention.pt', 'comparison.json'))]
    digests = {str(path): sha(path) for path in paths}
    for path in paths:
        if path.parent != study and digests[str(path)] != original_paths.get(path.resolve()):
            raise ValueError(f'Frozen input changed: {path.name}')
    if (digests[str(study / 'revision-writers.pt')] != prior['weights_sha256']
            or digests[str(observed / 'observation-weights.pt')] != episode_result['checkpoint_sha256']):
        raise ValueError('A frozen write component changed')
    checkpoint = args.output / 'intent-heads.pt'
    if args.resume_heads and (args.output / 'result.json').exists():
        raise FileExistsError('This run has already finished')
    if args.verify:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        expected = read(args.output / 'result.json')
        if sha(checkpoint) != expected['weights_sha256'] or digests != expected['sources']:
            raise ValueError('Replay sources differ')
    else:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True) if args.resume_heads else {}
    samples = read(EPISODES)['cases']
    source = read(DATA)
    fixture = read(OBSERVATIONS)['topics']
    topics, _ = training_data(read(DESIGN))
    names = {name: (*ROLE_NAMES[name], EXTRA_NAMES[name]) for name in EXTRA_NAMES}
    fields = {row['case']: row['fields'] for row in samples}
    cases = held_episodes({'cases': samples})
    train = training_groups(source, samples, fixture, names)
    calibration = training_groups(source, samples, fixture, names, calibration=True)
    held = examples(cases, fields, names, TEST)
    raw = [case['update'] for case in cases]
    if (group_texts(train) & (group_texts(calibration) | group_texts(held) | set(raw))
            or group_texts(calibration) & (group_texts(held) | set(raw))):
        raise ValueError('Intent splits must not contain repeated sentences')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    if individual != prior['individual_sha256'] or individual != episode_result['individual_sha256']:
        raise ValueError('The experiment requires the same individual')
    extend_positions(decoder, rotary, 2048)
    width = base.config.hidden_size
    writer = JointTurnValueWriter(64, feature_width=width).to(base.device)
    writer.load_state_dict(torch.load(study / 'revision-writers.pt', map_location='cpu', weights_only=True)['joint']['weights'])
    writer.eval().requires_grad_(False)
    char_features = FrozenTurnFeatures(decoder, tokenizer, depth=12)
    vote_features = FrozenTurnFeatures(decoder, tokenizer, depth=20)
    encoded = features_for(vote_features, train, calibration, held, raw, width)
    outcomes, votes = {}, {}
    for mode in MODES:
        if args.verify or args.resume_heads:
            model = ClauseIntent(width, recurrent=True, classes=len(INTENTS) + len(FIELDS))
            model.load_state_dict(saved[mode])
            model.eval().requires_grad_(False)
        else:
            model = fit(mode, train, encoded, width)
            saved[mode] = model.state_dict()
        compared = assess(model, encoded, calibration)
        test = assess(model, encoded, held, compared['floor'])
        writes, trace = apply_to_turns(model, encoded, writer, char_features, cases,
                                       prior['arms']['joint']['votes'], compared['floor'])
        votes[mode] = writes
        outcomes[mode] = {'calibration': compared, 'test': test, 'original': trace,
                          'matched_writes': check_matched_writes(test, writer, char_features)}
        print(mode, test['summary'], 'original exact', sum(row['exact'] for row in trace), flush=True)
    if not (args.verify or args.resume_heads):
        atomic_save(saved, checkpoint)
    torch.cuda.empty_cache()
    base.lm_head.to(base.device)
    layers = load_recurrent_replacements(decoder, args.layers, digests[str(args.checkpoint)])
    head = SpanActionHead(width).to(base.device)
    head.load_state_dict(torch.load(args.actions / 'actions.pt', map_location='cpu', weights_only=True))
    head.eval().requires_grad_(False)
    plan = TokenFieldAccessHead(2 * width).to(base.device)
    plan.load_state_dict(torch.load(observed / 'observation-weights.pt', map_location='cpu', weights_only=True)['planner'])
    plan.eval().requires_grad_(False)
    adapter = MissingFieldActions(width).to(base.device)
    adapter.load_state_dict(torch.load(args.typed / 'typed-fields.pt', map_location='cpu', weights_only=True)['realizer'])
    adapter.eval().requires_grad_(False)
    gate_saved = torch.load(args.gate / 'gate.pt', map_location='cpu', weights_only=True)
    gate = ObservationReadGate(3 * width)
    gate.load_state_dict(gate_saved['weights'])
    gate.eval().requires_grad_(False)
    config = torch.load(args.scope / 'scope.pt', map_location='cpu', weights_only=True)
    scope = {'model': PairedFieldScopeHead(3 * width), 'cutoff': config['cutoff']}
    scope['model'].load_state_dict(config['state'])
    scope['model'].eval().requires_grad_(False)
    _, _, prototypes = training_examples(source, samples, fixture, topics)
    questions = {case[key] for case in cases for key in ('all', 'limited')}
    inputs = questions | {case['first'] for case in cases} | {text for group in prototypes.values() for text in group}
    inputs |= scope_texts({(domain, q) for domain in topics for q in questions})
    state_file = args.output / 'working-state.pt'
    if args.verify and sha(state_file) != expected['states_sha256']:
        raise ValueError('The stored state changed before replay')
    states = torch.load(state_file, map_location='cpu', weights_only=True) if args.verify else {}
    with ExitStack() as stack:
        install(stack, layers)
        semantic = text_features(base, tokenizer, inputs, keep_tokens=questions)
        for mode in MODES:
            snapshots = episodes(cases, votes[mode], tokenizer, semantic[0], prototypes, topics)
            if args.verify:
                for key in snapshots:
                    for version in snapshots[key]:
                        actual, frozen = snapshots[key][version], states[mode][key][version]
                        if (not torch.equal(actual['key'], frozen['key'])
                                or any(actual[k] != frozen[k] for k in ('slots', 'topic', 'revision'))):
                            raise ValueError('A write did not reproduce its saved state')
            else:
                states[mode] = snapshots
            rows = audit(decoder, tokenizer, head, adapter, scope, plan, gate, semantic, states[mode], cases,
                         topics, gate_saved['thresholds'])
            outcomes[mode]['read'] = {'summary': summarize(rows), 'rows': rows}
            print(mode, 'read', outcomes[mode]['read']['summary'], flush=True)
        memory = evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))
    if memory != prior['memory'] or frozen_digest(runtime.organism, runtime.state) != individual:
        raise ValueError('The durable replay or individual changed')
    if any(sha(Path(path)) != digest for path, digest in digests.items()):
        raise ValueError('A frozen source changed during the comparison')
    if not args.verify:
        atomic_save(states, state_file)
    result = {'sources': digests, 'weights_sha256': sha(checkpoint), 'states_sha256': sha(state_file),
              'individual_sha256': individual, 'memory': memory, 'arms': outcomes, 'training': len(train)}
    if args.verify:
        if any(expected[key] != value for key, value in result.items()):
            raise ValueError('Independent reload did not reproduce')
        store_json(args.output / 'reload.json', {'weights_sha256': sha(checkpoint), 'seconds': perf_counter() - started})
    else:
        store_json(args.output / 'result.json', {**result, 'seconds': perf_counter() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('checkpoint', 'study', 'observed', 'actions', 'gate', 'typed', 'scope', 'output'):
        parser.add_argument(argument, type=Path)
    parser.add_argument('layers', nargs=2, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify', action='store_true')
    mode.add_argument('--resume-heads', action='store_true', help='Finish an interrupted run with saved intent heads')
    main(parser.parse_args())
