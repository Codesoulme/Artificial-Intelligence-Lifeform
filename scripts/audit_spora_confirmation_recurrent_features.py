"""Isolate whether the migrated attention layers change usable read-intent features."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from scripts import audit_spora_confirmation_scope_transfer as scope
from scripts import audit_spora_dense_joint_address as dense
from scripts import audit_spora_read_confirmation_action as action
from scripts import audit_spora_spoken_query_plan as plan
from scripts.audit_spora_crosswoz_field_transfer import make_cases
from scripts.benchmark_spora_recurrent_dialogue import extend_positions, load_full_streamed, load_integrated_runtime, using
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_recurrent_decode import load_recurrent_replacements


class SingleTurnFeatures(action.FrozenTurnFeatures):
    """Read unpadded prompts one at a time on both attention layouts."""

    def __call__(self, texts):
        width = max(map(len, texts))
        extract = super().__call__
        return torch.cat([F.pad(extract([text]), (0, 0, 0, width - len(text))) for text in texts])


def inputs(extract, groups):
    return {name: action.action_features(extract, questions, people)
            for name, (questions, people) in groups.items()}


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    prior = json.loads((args.previous / 'result.json').read_text(encoding='utf-8'))
    archive = args.profile / 'profile' / 'versions.sqlite3'
    reader_file, _, writer_file, binding = action.artifacts(args.reader, args.normalizer)
    checkpoint, *layers = args.recurrent
    pointer_file = args.transfer / 'personal.pt'
    source_paths = (archive, reader_file, writer_file, pointer_file, checkpoint,
                    *(folder / 'attention.pt' for folder in layers))
    sources = {path.resolve(): action.digest(path) for path in source_paths}
    known = {Path(path).resolve(): checksum for path, checksum in prior['sources'].items()}
    if any(known.get(path) != checksum for path, checksum in sources.items()):
        raise ValueError('The profile, recurrent weights and address head must match the prior experiment')
    args.output.mkdir(parents=True, exist_ok=False)
    paired = action.pairs(action.TRAIN, dense.ALIASES, action.TRAIN_NAMES)
    calibration = [question for pair in action.pairs(action.CAL, dense.CAL_ALIASES, action.CAL_NAMES)
                   for question in pair]
    held = {'previous': make_cases(action.HELD), 'fresh': make_cases(scope.FRESH)}
    excluded = set()
    ordinary = [question for *_, question in action.corpus_sample(args.corpus, 'train', excluded)]
    public_test = [question for *_, question in action.corpus_sample(args.corpus, 'test', excluded)]
    groups = {
        'train': ([question for pair in paired for question in pair]
                  + [question for i, question in enumerate(ordinary) if i % 4], action.TRAIN_NAMES),
        'calibration': (calibration, action.CAL_NAMES),
        'ordinary_calibration': (ordinary[::4], ()),
        'public_test': (public_test, ()),
        **{name: ([question for question, _ in cases], dense.NAMES) for name, cases in held.items()},
    }
    tokenizer = action.base.AutoTokenizer.from_pretrained(
        action.base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed(stage_output_projection=True)
    rotary = model.model.rotary_emb
    runtime = load_integrated_runtime(checkpoint, model, action.base.MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    torch.cuda.empty_cache()
    model.lm_head.to(model.device)
    replacements = load_recurrent_replacements(decoder, layers, sources[checkpoint.resolve()])
    if tuple(replacements) != (19, 22):
        raise ValueError('Expected the established layers 19 and 22')
    settings = torch.load(writer_file, map_location='cpu', weights_only=True)
    state = torch.load(reader_file, map_location='cpu', weights_only=True)
    reader = action.VersionReplyActions(model.config.hidden_size, state['scaffold_tokens'], slots=2)
    reader.load_state_dict(state)
    reader.eval().to(model.device)
    pointer = dense.JointPointer(model.config.hidden_size).eval()
    pointer.load_state_dict(torch.load(pointer_file, map_location='cpu', weights_only=True)['pointer'])
    shim = SimpleNamespace(base=model)
    early = action.FrozenTurnFeatures(shim, tokenizer, depth=plan.DEPTH)
    shared_early = SingleTurnFeatures(shim, tokenizer, depth=19)
    deeper = SingleTurnFeatures(shim, tokenizer, depth=23)
    extracted, early_control = {}, {}
    for layout in ('parent', 'recurrent'):
        with using(replacements, layout):
            early_control[layout] = action.action_features(shared_early, groups['fresh'][0], dense.NAMES)
            extracted[layout] = inputs(deeper, groups)
        print(layout, 'features extracted', flush=True)
    if not torch.equal(early_control['parent'], early_control['recurrent']):
        raise RuntimeError('Layouts diverged before the first migrated layer')
    if torch.equal(extracted['parent']['fresh'], extracted['recurrent']['fresh']):
        raise RuntimeError('The migrated layers did not change any held-out features')

    results = {}
    with using(replacements, 'recurrent'), action.VersionStore(archive, binding) as store:
        catalog = store.catalog()
        addresses = {name: action.rank(pointer, dense.align(early, groups[name][0], dense.NAMES), catalog)[0]
                     for name in held}
        for layout, data in extracted.items():
            head = action.fit(data['train'], len(paired), len(groups['train'][0]) - 2 * len(paired))
            threshold = action.choose_threshold(
                action.predict(head, data['calibration']),
                [action.eligible_spoken_lookup(question) for question in calibration],
                action.predict(head, data['ordinary_calibration']),
            )
            public_scores = action.predict(head, data['public_test'])
            batches = {}
            for name, cases in held.items():
                scores = action.predict(head, data[name])
                active = [score >= threshold for score in scores]
                replay = action.run_dialogue(store, model, tokenizer, reader, settings, cases, addresses[name], active)
                batches[name] = dict(replay, action_scores=scores,
                                     decomposition=scope.diagnose(cases, addresses[name], active))
            weights = args.output / f'{layout}.pt'
            atomic_save(head.state_dict(), weights)
            results[layout] = {'threshold': threshold, 'batches': batches, 'sha256': action.digest(weights),
                               'public_rejected': sum(score < threshold for score in public_scores)}
            print(layout, threshold, results[layout]['public_rejected'],
                  {name: data['summary'] for name, data in batches.items()}, flush=True)
    if any(action.digest(path) != checksum for path, checksum in sources.items()):
        raise RuntimeError('An input archive or checkpoint changed')
    store_json(args.output / 'result.json', {
        'results': results, 'inputs': {name: len(rows) for name, (rows, _) in groups.items()},
        'held_before_migration_equal': True,
        'held_after_migration_mean_abs_delta': float((extracted['parent']['fresh']
                                                     - extracted['recurrent']['fresh']).abs().mean()),
        'seconds': perf_counter() - started, 'sources': {str(path): checksum for path, checksum in sources.items()},
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
