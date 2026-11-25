"""Compare identical confirmation heads trained with different clause order."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn

from scripts import audit_spora_dense_joint_address as dense
from scripts import audit_spora_read_confirmation_action as action
from scripts import audit_spora_spoken_query_plan as plan
from scripts.audit_spora_crosswoz_field_transfer import make_cases
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save


SCOPE_PAIRS = (
    ('{a}{x}我想弄明白。等会还要给{b}发消息。',
     '{a}{x}先搁一搁。等会还要给{b}发消息。'),
    ('顺嘴问一下{a}{x}？剩下的我会自己张罗。',
     '顺嘴提到{a}{x}。这件我也会自己张罗。'),
    ('{a}{x}？我出门前得知道。对了，别替我找{b}。',
     '{a}{x}？我已经有答案。对了，别替我找{b}。'),
    ('{b}今天不来。{a}{x}？这份我还没个准信。',
     '{b}今天不来。{a}{x}？这句话我念给你听听而已。'),
    ('我本来以为不用打听{a}{x}。现在还是得麻烦你告诉我。',
     '我本来打算打听{a}{x}。现在想想还是我亲自问。'),
    ('先别理{b}，我需要弄清{a}{x}；我先去收下衣服。',
     '先别理{b}，{a}{x}也过一阵再说；我先去收下衣服。'),
    ('刚想到{a}{x}？{b}那项不劳你费心。', '刚想到{a}{x}。两项都只是在举例。'),
    ('我以前听{a}讲过。不过{a}{x}我想请你提醒下。剩下不用管。',
     '我以前听{a}讲过。不过{a}{x}不用你提醒了。剩下不用管。'),
)

FRESH = (
    ('碰到夏宁瑶再聊。夏宁遥当时让我给她买哪种喝的？我下午还得接人。', 1, 'drinks'),
    ('夏宁遥我手里有数。跟夏宁瑶会合选在哪儿了？回头把伞也带上。', 0, 'places'),
    ('夏宁瑶的安排不用麻烦你。夏宁遥托我带的到底是个啥？', 1, 'items'),
    ('我想想，夏宁瑶挑的是哪册来着？夏宁遥那册我自己去翻。', 0, 'books'),
    ('原想先管郭回霜的事。郭回双说的那杯你还记得是哪种吗？', 3, 'drinks'),
    ('给郭回霜留个位置。我得先弄清上回说定在哪聚。郭回双不会到。', 2, 'places'),
    ('郭回霜那一份不用找；郭回双让我拿去的物件你说个名字。', 3, 'items'),
    ('郭回霜先前提过一本，我想翻翻，是叫什么？郭回双那本已还了。', 2, 'books'),
    ('夏宁瑶想喝哪种先不说了。夏宁遥我约过了，你帮忙拟个迟到理由。', None, None),
    ('郭回霜说的用具我刚问到了，郭回双借的读物也由我自己找。', None, None),
    ('请给「夏宁遥说在哪里集合」加个问号，再把夏宁瑶这个名字译出来。', None, None),
    ('我本来想请你说说郭回霜选的书。现在她人就在这儿，我开口问就成。', None, None),
)


def diagnose(cases, addresses, active):
    positive = [(expected == address, allowed)
                for (_, expected), address, allowed in zip(cases, addresses, active, strict=True)
                if expected is not None]
    return {
        'positives': len(positive),
        'both_correct': sum(right and allowed for right, allowed in positive),
        'correct_if_action_given': sum(right for right, _ in positive),
        'correct_if_address_given': sum(allowed for _, allowed in positive),
        'neither_correct': sum(not right and not allowed for right, allowed in positive),
        'negative_action_triggers': sum(allowed and expected is None
                                        for (_, expected), allowed in zip(cases, active, strict=True)),
    }


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    previous = json.loads((args.previous / 'result.json').read_text(encoding='utf-8'))
    fixed_head = args.previous / 'action-19.pt'
    source_paths = {Path(p): checksum for p, checksum in previous['sources'].items()}
    source_paths[fixed_head] = previous['results']['19']['sha256']
    if any(action.digest(path) != checksum for path, checksum in source_paths.items()):
        raise ValueError('Previous experiment sources have changed')
    archive = args.profile / 'profile' / 'versions.sqlite3'
    reader_file, _, writer_file, binding = action.artifacts(args.reader, args.normalizer)
    backbone = (args.recurrent[0], *(folder / 'attention.pt' for folder in args.recurrent[1:]))
    pointer_file = args.transfer / 'personal.pt'
    bound = {path.resolve() for path in source_paths}
    if any(path.resolve() not in bound for path in (archive, reader_file, writer_file, pointer_file, *backbone)):
        raise ValueError('Expected the same profile, reader, pointer and recurrent weights')
    args.output.mkdir(parents=True, exist_ok=False)

    train = action.pairs(SCOPE_PAIRS, dense.ALIASES, action.TRAIN_NAMES)
    calibration = action.pairs(action.CAL, dense.CAL_ALIASES, action.CAL_NAMES)
    old_training = action.pairs(action.TRAIN, dense.ALIASES, action.TRAIN_NAMES)
    if len(train) != len(old_training):
        raise ValueError('Training arm must match the baseline pair count')
    train_text = [question for pair in train for question in pair]
    calibration_text = [question for pair in calibration for question in pair]
    evaluation = {'previous': make_cases(action.HELD), 'fresh': make_cases(FRESH)}
    fresh_keys = {plan.anonymize(question, dense.NAMES)[0] for question, _ in evaluation['fresh']}
    seen_keys = {
        plan.anonymize(question, people)[0]
        for rows, people in ((train, action.TRAIN_NAMES), (old_training, action.TRAIN_NAMES),
                             (calibration, action.CAL_NAMES))
        for pair in rows for question in pair
    }
    for batch in previous['results']['19']['held'].values():
        seen_keys.update(plan.anonymize(row['question'], dense.NAMES)[0] for row in batch['rows'])
    if fresh_keys & seen_keys or len(fresh_keys) != len(FRESH):
        raise ValueError('Fresh question split contains repeated prompts')

    excluded = set()
    ordinary = [question for *_, question in action.corpus_sample(args.corpus, 'train', excluded)]
    test_ordinary = [question for *_, question in action.corpus_sample(args.corpus, 'test', excluded)]
    train_ordinary = [question for index, question in enumerate(ordinary) if index % 4]
    tokenizer = action.base.AutoTokenizer.from_pretrained(
        action.base.MODEL, local_files_only=True, trust_remote_code=False)
    settings = torch.load(writer_file, map_location='cpu', weights_only=True)
    results = {}
    with action.local_model(args.recurrent) as model, action.VersionStore(archive, binding) as store:
        state = torch.load(reader_file, map_location='cpu', weights_only=True)
        reader = action.VersionReplyActions(model.config.hidden_size, state['scaffold_tokens'], slots=2)
        reader.load_state_dict(state)
        reader.eval().to(model.device)
        pointer = dense.JointPointer(model.config.hidden_size).eval()
        pointer.load_state_dict(torch.load(pointer_file, map_location='cpu', weights_only=True)['pointer'])
        features = action.FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=19)
        pointer_features = action.FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=plan.DEPTH)
        training = action.action_features(features, train_text + train_ordinary, action.TRAIN_NAMES)
        calibration_features = action.action_features(features, calibration_text, action.CAL_NAMES)
        ordinary_calibration = action.action_features(features, ordinary[::4], ())
        ordinary_test = action.action_features(features, test_ordinary, ())
        prompts = {name: [question for question, _ in cases] for name, cases in evaluation.items()}
        inputs = {name: action.action_features(features, questions, dense.NAMES) for name, questions in prompts.items()}
        addresses = {
            name: action.rank(pointer, dense.align(pointer_features, questions, dense.NAMES), store.catalog())[0]
            for name, questions in prompts.items()
        }
        trained = action.fit(training, len(train), len(train_ordinary))
        baseline = nn.Sequential(nn.LayerNorm(training.shape[-1]), nn.Linear(training.shape[-1], 64),
                                 nn.GELU(), nn.Linear(64, 1))
        baseline.load_state_dict(torch.load(fixed_head, map_location='cpu', weights_only=True))
        baseline.eval()
        cutoff = action.choose_threshold(
            action.predict(trained, calibration_features),
            [action.eligible_spoken_lookup(question) for question in calibration_text],
            action.predict(trained, ordinary_calibration),
        )
        for name, head, limit in (('baseline', baseline, previous['results']['19']['cutoff']),
                                  ('scope_pairs', trained, cutoff)):
            public_scores = action.predict(head, ordinary_test)
            batches = {}
            for batch, cases in evaluation.items():
                scores = action.predict(head, inputs[batch])
                active = [score >= limit for score in scores]
                replay = action.run_dialogue(store, model, tokenizer, reader, settings, cases, addresses[batch], active)
                batches[batch] = dict(replay, action_scores=scores, decomposition=diagnose(cases, addresses[batch], active))
            results[name] = {'cutoff': limit, 'public_rejected': sum(score < limit for score in public_scores),
                             'batches': batches}
            print(name, limit, results[name]['public_rejected'],
                  {key: data['summary'] for key, data in batches.items()}, flush=True)
        expected = [address for _, address in evaluation['fresh']]
        oracle = action.run_dialogue(store, model, tokenizer, reader, settings, evaluation['fresh'],
                                     expected, [address is not None for address in expected])
        print('fresh_oracle', oracle['summary'], flush=True)

    earlier = previous['results']['19']['held']['new']['summary']
    if results['baseline']['batches']['previous']['summary'] != earlier:
        raise RuntimeError('The baseline did not reproduce the previous replay')
    if any(action.digest(path) != checksum for path, checksum in source_paths.items()):
        raise RuntimeError('A bound checkpoint or profile changed')
    weight_path = args.output / 'action-19.pt'
    atomic_save(trained.state_dict(), weight_path)
    store_json(args.output / 'result.json', {
        'results': results, 'fresh_oracle': oracle, 'train_pairs': len(train),
        'public_test_size': len(test_ordinary), 'seconds': perf_counter() - started,
        'action_sha256': action.digest(weight_path),
        'sources': {str(path): checksum for path, checksum in source_paths.items()},
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous', type=Path)
    parser.add_argument('transfer', type=Path)
    parser.add_argument('profile', type=Path)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reader', type=Path, default=Path('runs/spora-dual-version-reply-v1'))
    parser.add_argument('--normalizer', type=Path, default=Path('runs/spora-version-normalizer-v1'))
    parser.add_argument('--recurrent', type=Path, nargs=3, required=True)
    main(parser.parse_args())
