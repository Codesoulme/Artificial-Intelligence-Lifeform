"""Train independent draft matching and action confirmation on frozen SPORA features."""

import argparse
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.train_spora_revision_intent import (
    DATA, DESIGN, EPISODES, EXTRA_NAMES, HELD, MODEL, OBSERVATIONS, ROLE_NAMES, AutoTokenizer, atomic_save, evaluate,
    extend_positions, frozen_digest, held_episodes, install, load_full_streamed, load_integrated_runtime,
    load_recurrent_replacements, read, records, sha, store_json,
)
from scripts.train_spora_revision_reference import (
    CAL_CUES, HELD_CUES, TRAIN_CUES, episodes, evaluation_cases, training_cases,
)
from scripts.train_spora_observed_episodes import encode_changes, state_for, state_values
from spora.research_revision_draft import RevisionDecision, RevisionDraft
from spora.research_revision_reference import DraftReferenceMatcher


SEED = 26220
UNSEEN_TOPIC = 'film_notice'
FIELD_TEXT = {'time': '日期和钟点', 'place': '场地或地址', 'person': '负责人员',
              'quantity': '数目与单位', 'gear': '用品或作品名称'}
TRAIN_ALIASES = {
    'time': ('何时开始', '到场钟点', '约定的日期时段', '选哪天哪一刻'),
    'place': ('要去的地址', '碰头的场所', '落脚点', '具体在哪个地方'),
    'person': ('参与者的姓名', '找谁办理', '承办人选', '由哪位接手'),
    'quantity': ('究竟有几件', '实际份数', '按多少来准备', '报上的总件数'),
    'gear': ('需要准备的材料', '所选用品', '打算使用的物品', '选中的内容名称'),
}
HELD_ALIASES = {
    'time': ('几号到了再开始', '安排在什么钟点'), 'place': ('地图上该找哪处', '应当到哪个场所'),
    'person': ('具体托付给谁', '当班的是什么人'), 'quantity': ('合起来有多少份', '究竟该清点出几件'),
    'gear': ('计入清单的具体项目', '究竟选哪样东西'),
}
BRIDGES = {
    'train': ('这里要讨论的是{cue}。', '注意，我在核对{cue}。', '{cue}这一项是我现在说的事。'),
    'calibration': ('咱们谈的是{cue}这个细节。',),
    'held': ('这回只说清{cue}。', '我想对应到{cue}这一栏。'),
}
ACTIONS = {
    'train': (('确认执行这项调整。', '好，这个改动可以正式生效。', '我同意变更，请登记。',
               '等到了批准，照商量好的办。', '这个已经定下来了，替换原记录。'),
              ('我确认收到了消息，再等会儿。', '这个先挂在讨论稿上。', '可以考虑，目前还没拍板。',
               '确认过内容了，但不要提交。', '原来的作数，收回这次调整。', '等一下，我还要再问。')),
    'calibration': (('核准了，按刚才的变更落实。', '决定已通过，可以改。'),
                    ('确认收到，变更先暂缓。', '上面提到的改动不用办了。')),
    'held': (('行，就照刚提到的安排吧。', '这项现在可以落到正式记录。', '最后拍板，按这个新方案办。'),
             ('我同意先讨论，暂时别动记录。', '收到，不过上面提的变更作废。', '这还只是个意见，晚点再答复。')),
}


def aliases(split, field):
    if split == 'train':
        return (TRAIN_CUES[field], *TRAIN_ALIASES[field])
    if split == 'calibration':
        return (CAL_CUES[field],)
    return (HELD_CUES[field], *HELD_ALIASES[field])


def build_cases(cases, names, split, *, original=False):
    rows = episodes(cases, names, split, {'train': TRAIN_CUES, 'calibration': CAL_CUES, 'held': HELD_CUES}[split])
    case_map = {row['key']: row for row in cases}
    variants = 5 if split == 'train' else (1 if split == 'calibration' else 3)
    result = []
    for i, row in enumerate(rows):
        case = case_map[row['key']]
        options = row['options']
        descriptions = tuple(f'{names[case["topic"]][opt["field"]]}，{FIELD_TEXT[opt["field"]]}' for opt in options)
        for variant in range(1 if original else variants):
            target = row['target']
            bridge = row['bridge']
            if not original and target is not None:
                cue = aliases(split, options[target]['field'])[variant]
                bridge = BRIDGES[split][(i // 6 + variant) % len(BRIDGES[split])].format(cue=cue)
            if not original and target is None and (i // 6 + variant) % 2:
                bridge = f'这两项都还要商量，{descriptions[0]}和{descriptions[1]}先放一起。'
            actions = ((row['last'], True),) if original else tuple(
                (samples[(i // 6 + variant) % len(samples)], allowed)
                for samples, allowed in zip(ACTIONS[split], (True, False), strict=True))
            for utterance, confirm in actions:
                result.append({'key': row['key'], 'topic': case['topic'], 'bridge': bridge, 'utterance': utterance,
                               'descriptions': descriptions, 'options': options, 'target': target, 'confirm': confirm})
    return result


def encode_texts(featurizer, texts):
    features = {}
    for start in range(0, len(texts), 48):
        batch = texts[start:start + 48]
        hidden = featurizer(batch)
        for i, text in enumerate(batch):
            features[text] = hidden[i, :len(text)].float().mean(dim=0)
    return features


def encode(featurizer, sets, *, stored=None):
    stored = {} if stored is None else stored
    texts = {text for rows in sets.values() for row in rows
             for text in (row['bridge'], row['utterance'], *row['descriptions'])}
    return stored | encode_texts(featurizer, sorted(texts - stored.keys()))


def tensors(rows, features):
    bridges = torch.stack([features[row['bridge']] for row in rows])
    drafts = torch.stack([torch.stack([features[text] for text in row['descriptions']]) for row in rows])
    actions = torch.stack([features[row['utterance']] for row in rows])
    choices = torch.tensor([2 if row['target'] is None else row['target'] for row in rows])
    commits = torch.tensor([int(row['confirm']) for row in rows])
    return bridges, drafts, actions, choices, commits


def train_heads(reference, action, data):
    bridges, drafts, actions, choices, commits = data
    optim = torch.optim.AdamW([*reference.parameters(), *action.parameters()], lr=.002)
    rng = torch.Generator().manual_seed(SEED)
    for epoch in range(10):
        for idx in torch.randperm(len(choices), generator=rng).split(128):
            scores = reference(bridges[idx], drafts[idx])
            command = action(F.normalize(actions[idx], dim=-1))
            loss = F.cross_entropy(scores, choices[idx]) + F.cross_entropy(command, commits[idx])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
        print('epoch', epoch + 1, round(float(loss.detach()), 5), flush=True)


@torch.inference_mode()
def predictions(reference, action, data):
    bridges, drafts, actions, _, _ = data
    logits = reference(bridges, drafts)
    command = action(F.normalize(actions, dim=-1))
    best, pick = logits[:, :2].max(1)
    rival = logits[:, :2].min(1).values
    gap = best - torch.maximum(logits[:, 2], rival)
    return list(zip(pick.tolist(), gap.tolist(), (command[:, 1] - command[:, 0]).tolist(), strict=True))


def audit(rows, score, case_states, tokenizer, scope_cut, action_cut):
    details = []
    for row, (pick, gap, action) in zip(rows, score, strict=True):
        state = case_states[row['key']]
        target = row['target'] if row['confirm'] else None
        accepted = action > action_cut and gap > scope_cut
        selected = pick if accepted else None
        expected = dict(state)
        patch = {}
        if target is not None:
            option = row['options'][target]
            expected[option['field']] = option['value']
        if accepted:
            option = row['options'][pick]
            field = option['field']
            draft = RevisionDraft(field).observe(RevisionDecision.PROPOSE, field=field, value=option['value'])
            patch = draft.observe(RevisionDecision.CONFIRM, field=field).patch()
        after = state_values(tokenizer, state_for(tokenizer, state).update(encode_changes(tokenizer, patch)))
        details.append({'key': row['key'], 'topic': row['topic'], 'bridge': row['bridge'], 'utterance': row['utterance'],
                        'drafts': list(row['descriptions']), 'reference': row['target'], 'target': target,
                        'confirm': row['confirm'], 'top': pick, 'gap': gap, 'action': action,
                        'selected': selected, 'state_exact': after == expected})
    return details


def summarize(rows):
    return {'turns': len(rows), 'should_commit': sum(row['target'] is not None for row in rows),
            'wrong_commit': sum(row['selected'] is not None and row['selected'] != row['target'] for row in rows),
            'right_commit': sum(row['selected'] is not None and row['selected'] == row['target'] for row in rows),
            'state_exact': sum(row['state_exact'] for row in rows),
            'reference_rank': sum(row['top'] == row['reference'] for row in rows),
            'reference_cases': sum(row['reference'] is not None for row in rows)}


def cuts(rows, scores):
    action_cut = max(margin for row, (_, _, margin) in zip(rows, scores, strict=True) if not row['confirm'])
    scope_cut = max((gap for row, (pick, gap, margin) in zip(rows, scores, strict=True)
                     if row['confirm'] and margin > action_cut and pick != row['target']), default=0.)
    return max(0., scope_cut), action_cut


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    report, weights = args.output / 'result.json', args.output / 'heads.pt'
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    prior = read(args.reference)
    protected = (args.checkpoint, DATA, DESIGN, EPISODES, OBSERVATIONS, args.reference,
                 *(layer / name for layer in args.layers for name in ('attention.pt', 'comparison.json')))
    sources = {str(path): sha(path) for path in protected}
    original = {Path(path).resolve(): digest for path, digest in prior['sources'].items()}
    if any(sources[str(path)] != original.get(path.resolve()) for path in protected if path != args.reference):
        raise ValueError('Frozen inputs differ from the reference study')
    expected = read(report) if args.verify else None
    if expected is not None and (expected['sources'] != sources or expected['weights_sha256'] != sha(weights)):
        raise ValueError('Independent replay inputs changed')
    specs, data, fixture = read(EPISODES)['cases'], read(DATA), read(OBSERVATIONS)['topics']
    spoken = {topic: (*ROLE_NAMES[topic], EXTRA_NAMES[topic]) for topic in EXTRA_NAMES}
    names = {row['case']: dict(zip(row['fields'], spoken[row['case']], strict=True)) for row in specs}
    training = [row for row in training_cases(specs, fixture, spoken, data['values']['train'])
                if row['topic'] != UNSEEN_TOPIC]
    calibration = [row for row in training_cases(specs, fixture, spoken, data['values']['held'], calibration=True)
                   if row['topic'] != UNSEEN_TOPIC]
    held = evaluation_cases(held_episodes({'cases': specs}))
    sets = {'calibration': build_cases(calibration, names, 'calibration'),
            'original': build_cases(held, names, 'held', original=True),
            'held': build_cases(held, names, 'held')}
    if not args.verify:
        sets['train'] = build_cases(training, names, 'train')

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed()
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    if individual != prior['individual_sha256']:
        raise ValueError('The individual changed')
    extend_positions(decoder, rotary, 2048)
    feature = FrozenTurnFeatures(decoder, tokenizer, depth=20)
    # Encode the evaluation partition separately to keep its batches identical on reload.
    testing = {name: rows for name, rows in sets.items() if name != 'train'}
    eval_features = encode(feature, testing)
    torch.manual_seed(SEED)
    reference = DraftReferenceMatcher(base.config.hidden_size)
    action = nn.Sequential(nn.Linear(base.config.hidden_size, 64), nn.GELU(), nn.Linear(64, 2))
    if args.verify:
        saved = torch.load(weights, map_location='cpu', weights_only=True)
        reference.load_state_dict(saved['reference'])
        action.load_state_dict(saved['action'])
    else:
        train_features = encode(feature, {'train': sets['train']})
        train_heads(reference, action, tensors(sets['train'], train_features))
        atomic_save({'reference': reference.state_dict(), 'action': action.state_dict()}, weights)
    reference.eval()
    action.eval()
    scored = {name: predictions(reference, action, tensors(rows, eval_features)) for name, rows in testing.items()}
    scope_cut, action_cut = cuts(testing['calibration'], scored['calibration'])
    results = {}
    for name, rows in testing.items():
        states = {case['key']: case['initial'] for case in (calibration if name == 'calibration' else held)}
        details = audit(rows, scored[name], states, tokenizer, scope_cut, action_cut)
        results[name] = {'summary': summarize(details), 'rows': details}
        print(name, results[name]['summary'], flush=True)
    unseen = summarize([row for row in results['held']['rows'] if row['topic'] == UNSEEN_TOPIC])
    print('unseen topic', unseen, flush=True)
    layers = load_recurrent_replacements(decoder, args.layers, sources[str(args.checkpoint)])
    with ExitStack() as stack:
        install(stack, layers)
        memory = evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))
    if (memory != prior['memory'] or frozen_digest(runtime.organism, runtime.state) != individual
            or any(sha(path) != sources[str(path)] for path in protected)):
        raise ValueError('Frozen sources or durable memory changed')
    result = {'sources': sources, 'weights_sha256': sha(weights), 'individual_sha256': individual,
              'cuts': {'scope': scope_cut, 'action': action_cut}, 'sets': results, 'unseen': unseen, 'memory': memory}
    if args.verify:
        if any(expected[key] != value for key, value in result.items()):
            raise ValueError('Independent replay did not match')
        store_json(args.output / 'reload.json', {'weights_sha256': result['weights_sha256'], 'seconds': perf_counter() - started})
    else:
        store_json(report, {**result, 'train_turns': len(sets['train']), 'seconds': perf_counter() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'reference', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('layers', nargs=2, type=Path)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
