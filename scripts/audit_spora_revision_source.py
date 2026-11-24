"""Separate literal updates, references to proposals, and turns with no value."""

import argparse
import json
from pathlib import Path

from scripts.audit_spora_clause_intent import clause_features
from scripts.audit_spora_decision_transitions import ACTIONS, replay, test_rows
from scripts.audit_spora_revision_value import LABELS, SCENES, SCENE_FIELDS, head_inputs, read_values
from scripts.audit_spora_write_intent import MODEL, SEED, TRAIN_NAMES, TRAIN_VALUES, fit_intent
from scripts.train_spora_open_name_roles import (
    AutoTokenizer, FrozenTurnFeatures, atomic_save, frozen_digest, load_full_streamed,
    load_integrated_runtime, sha, store_json, torch,
)
from spora.research_clause_intent import ClauseIntent
from spora.research_revision_value import RevisionValueHead


SOURCES = ('no_value', 'literal', 'proposal')
FORMAT = 'spora-revision-source-v1'
PATTERNS = {
    'no_value': ('想问{name}的{field}是否真的变成{value}。', '我念一下草稿，{name}的{field}改成{value}，仅供讨论。',
                 '关于{name}的{field}不再考虑{value}，保持原值。', '把{name}那条{field}调成{value}的要求撤掉。',
                 '原来是{value}吗？只读给我听，不改{name}的{field}。', '我只是想查{name}原先定的{field}。'),
    'literal': ('最终确认{name}的{field}写成{value}。', '把{name}的{field}更新为{value}，可以提交了。',
                '已有新决定，给{name}采用{value}作为{field}。', '有个备选，给{name}考虑用{value}当作{field}。',
                '先拟个方案，{name}那栏的{field}填{value}，等下再核实。', '我提议将{name}的{field}换成{value}，暂缓提交。'),
    'proposal': ('沿用刚才的提议，确认更新{name}的{field}。', '已经决定采用之前谈好的{field}，把{name}那项写入。',
                 '给{name}用之前定的那个备选，{field}可以更新了。', '之前考虑的值已经确认，改{name}的{field}吧。',
                 '请执行上一条关于{name}的{field}更正。', '我批准刚才的提案，更新{name}的{field}。'),
}


def training_rows():
    rows = []
    for name in TRAIN_NAMES:
        for i, field in enumerate(LABELS):
            value = TRAIN_VALUES[i]
            prior = f'{name}的{field}有人提议写为{value}，仍在讨论。\n'
            for source, templates in PATTERNS.items():
                for pattern in templates:
                    text = pattern.format(name=name, field=field, value=value)
                    row = {'text': text, 'label': SOURCES.index(source)}
                    rows.extend((row, {**row, 'text': prior + text, 'focus_start': len(prior)}))
    return rows


def annotate(rows):
    fields = {scene[0]: pair for scene, pair in zip(SCENES, SCENE_FIELDS, strict=True)}
    return [{**row, 'field': fields[row['scene']][row['address']],
             'source': 'literal' if row['value'] is not None else
             'proposal' if ACTIONS[row['label']] == 'confirm' else 'no_value'} for row in rows]


def replay_sources(rows, actions, values, reference):
    accepted_actions, inputs = [], []
    for row, action, value, source in zip(rows, actions, values, reference, strict=True):
        allowed = action
        candidate = None
        if ACTIONS[action] in ('propose', 'confirm'):
            if source == 'literal' and value is not None:
                candidate = value
            elif source != 'proposal' or ACTIONS[action] != 'confirm':
                allowed = ACTIONS.index('other')
        accepted_actions.append(allowed)
        inputs.append({**row, 'candidate': candidate, 'predicted_source': source})
    return replay(inputs, accepted_actions)


def main(checkpoint, actions, values, output, verify):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    paths = {'checkpoint': checkpoint, 'actions': actions, 'values': values}
    hashes = {name: sha(path) for name, path in paths.items()}
    action = torch.load(actions, map_location='cpu', weights_only=True)
    value = torch.load(values, map_location='cpu', weights_only=True)
    if (action['sources']['checkpoint'] != hashes['checkpoint'] or value['sources']['checkpoint'] != hashes['checkpoint']
            or value['sources']['actions'] != hashes['actions'] or not action['sources'].get('focus_current')
            or value['sources'].get('separate_loss')):
        raise ValueError('The source experiment requires the paired action and zero-floor value weights')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    if any(saved['sources']['individual'] != individual for saved in (action, value)):
        raise ValueError('Action and value weights must use the same individual')
    # The audit calls the decoder prefix directly and never projects to vocabulary logits.
    max_depth = max(action['sources']['depth'], value['sources']['depth'])
    base = runtime.decoder.base
    base.model.layers = base.model.layers[:max_depth]
    base.lm_head = torch.nn.Identity()
    torch.cuda.empty_cache()
    deep = FrozenTurnFeatures(runtime.decoder, tokenizer, depth=action['sources']['depth'])
    shallow = FrozenTurnFeatures(runtime.decoder, tokenizer, depth=value['sources']['depth'])
    width = runtime.decoder.base.config.hidden_size
    torch.manual_seed(SEED)
    source_head = ClauseIntent(width, recurrent=True, classes=len(SOURCES))
    train = training_rows()
    if verify is None:
        fit_intent(source_head, *clause_features(deep, train, width), train, epochs=6)
    else:
        saved = torch.load(verify / 'source.pt', map_location='cpu', weights_only=True)
        if saved['format'] != FORMAT or saved['inputs'] != hashes or saved['individual'] != individual:
            raise ValueError('Value source weights belong to a different checkpoint')
        source_head.load_state_dict(saved['state'], strict=True)
        source_head.eval().requires_grad_(False)
    rows = annotate(test_rows(context=True, focus_current=True))
    clause_batch = clause_features(deep, rows, width)
    action_head = ClauseIntent(width, recurrent=True, classes=len(ACTIONS)).eval().requires_grad_(False)
    action_head.load_state_dict(action['state'], strict=True)
    predicted_actions = action_head(*clause_batch).argmax(-1).tolist()
    predicted_source = [SOURCES[i] for i in source_head(*clause_batch).argmax(-1).tolist()]
    value_head = RevisionValueHead(width, len(LABELS), contextual=value['sources'].get('contextual', False))
    value_head.load_state_dict(value['state'], strict=True)
    value_head.eval().requires_grad_(False)
    current = [{**row, 'text': row['turn']} for row in rows]
    literal_values = read_values(value_head, current, head_inputs(current, shallow, width))
    expected_sources = [row['source'] for row in rows]
    expected_actions = [row['label'] for row in rows]
    baseline = replay([{**row, 'candidate': item} for row, item in zip(rows, literal_values, strict=True)], predicted_actions)
    arms = {'implicit_fallback': baseline,
            'known_source': replay_sources(rows, predicted_actions, literal_values, expected_sources),
            'predicted_source': replay_sources(rows, predicted_actions, literal_values, predicted_source),
            'source_only': replay_sources(rows, expected_actions, [row['value'] for row in rows], predicted_source)}
    counts = {kind: {'correct': sum(found == expected == kind for found, expected in
                                  zip(predicted_source, expected_sources, strict=True)),
                     'total': expected_sources.count(kind)} for kind in SOURCES}
    print('Value source decisions', counts, flush=True)
    for name, results in arms.items():
        print(name, results['summary'], flush=True)
    if any(sha(path) != hashes[key] for key, path in paths.items()) or frozen_digest(runtime.organism, runtime.state) != individual:
        raise ValueError('The source weights or individual state changed')
    output.mkdir(parents=True, exist_ok=False)
    report = {'format': FORMAT, 'inputs': hashes, 'individual': individual, 'training': len(train),
              'counts': counts, 'arms': arms}
    if verify is None:
        atomic_save({'format': FORMAT, 'inputs': hashes, 'individual': individual,
                     'state': source_head.state_dict()}, output / 'source.pt')
    else:
        expected = json.loads((verify / 'result.json').read_text(encoding='utf-8'))
        if json.loads(json.dumps(report)) != expected:
            raise ValueError('Reloaded value source predictions differ from their recorded run')
        report = {'identical': True, 'inputs': hashes, 'individual': individual}
    store_json(output / 'result.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'actions', 'values', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--verify', type=Path)
    main(**vars(parser.parse_args()))
