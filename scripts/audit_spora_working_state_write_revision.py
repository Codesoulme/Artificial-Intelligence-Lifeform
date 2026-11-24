"""Compare writing supervision and joint span scoring with a frozen working-state reader."""

import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.audit_spora_latent_working_state import (
    DATA, FEATURE_DEPTH, FORMAT, MODEL, SEED, WRITER_WIDTH, FrozenTurnFeatures, AutoTokenizer,
    baselines, evaluate_reader, fit_writer, frozen_digest, initial_version, load_full_streamed,
    load_integrated_runtime, load_reader, measure, rollout_states, score_case, settings_for, sha, state_from, torch,
)
from scripts.train_spora_dialogue_working_state import annotated, dialogue_data
from spora.language_training import atomic_save
from spora.research_dialogue_working_state import JointTurnValueWriter, TurnValueWriter


REVISION = DATA.with_name('dialogue-working-state-write-revision-v1.json')
LAYER = 19
CONFIGS = {'original': (TurnValueWriter, 2.), 'curriculum': (TurnValueWriter, 2.),
           'joint': (JointTurnValueWriter, 0.)}


def revised_data(source, revision):
    patterns = {}
    for kind, original in source['kinds'].items():
        revised = revision['train'][kind]
        patterns[kind] = {**original, 'train_first': original['train_first'] + revised['first'],
                          'train_update': original['train_update'] + revised['update']}
    train = dialogue_data({**source, 'kinds': patterns}, held=False, count=128)
    held = []
    for row in revision['held']:
        first, initial = row['first']
        update, changes = row['update']
        original_values = {f'old_{field}': value for field, value in initial.items()}
        current = initial | changes
        template = source['kinds'][row['kind']]['answer']
        held.append({**row, 'first': annotated(first, initial), 'update': annotated(update, original_values | changes),
                     'initial': initial, 'changes': changes, 'current': current,
                     'initial_answer': template.format(**initial), 'answer': template.format(**current)})
    training_texts = {text for case in train for text in (case['first'][0], case['update'][0], case['question'])}
    if any(text in training_texts for case in held for text in (case['first'][0], case['update'][0], case['question'])):
        raise ValueError('Held turns overlap revised writer training')
    return train, held


def summarize_writes(rows):
    counts = Counter()
    for row in rows:
        case, (_, update, query), written = row['case'], row['writes'], row['written']
        counts['state_exact'] += written == case['current']
        counts['update_exact'] += update == case['changes']
        counts['changed_total'] += len(case['changes'])
        counts['changed_correct'] += sum(written.get(key) == value for key, value in case['changes'].items())
        kept = case['initial'].keys() - case['changes'].keys()
        counts['retained_total'] += len(kept)
        counts['retained_correct'] += sum(written.get(key) == case['initial'][key] for key in kept)
        counts['unchanged_overwrites'] += sum(key in update for key in kept)
        counts['missing_fields'] += len(case['current'].keys() - written.keys())
        counts['wrong_fields'] += sum(written[key] != case['current'][key] for key in written.keys() & case['current'].keys())
        counts['extra_fields'] += len(written.keys() - case['current'].keys())
        counts['query_writes'] += bool(query)
    return {'count': len(rows), **counts}


def writes_for(writer, features, tokenizer, references, margin):
    device = writer.embedding.weight.device

    def locate(text):
        return writer.write(text, margin, features=features(text).to(device))

    rows = []
    for reference in references:
        case = reference['case']
        turns = (case['first'][0], case['update'][0], case['question'])
        state, writes = rollout_states(locate, tokenizer, turns)
        rows.append({'case': case, 'writes': writes, 'written': writes[0] | writes[1] | writes[2], 'state': state})
    return rows


def evaluate_writes(decoder, reader, tokenizer, references, settings, rows):
    for row, reference in zip(rows, references, strict=True):
        case, history, question = row['case'], reference['history'], row['case']['question']
        states = {'initial': state_from(row['writes'][0], tokenizer), 'current': row.pop('state')}
        for name, state in states.items():
            generated = measure(decoder, reader, tokenizer, settings, history, question, state)
            score = score_case(case if name == 'current' else initial_version(case), generated)
            row[name] = {'generated': generated, **score}
    return {'answer_complete': sum(row['current']['complete'] for row in rows),
            'both_states': sum(row['current']['complete'] and row['initial']['complete']
                               and row['current']['generated']['answer'] != row['initial']['generated']['answer']
                               for row in rows)}


def verify_writes(args):
    result_path = args.output / 'comparison.json'
    result = json.loads(result_path.read_text(encoding='utf-8'))
    protected = {Path(path): value for path, value in result['sources'].items()}
    if any(sha(path) != value for path, value in protected.items()) or args.checkpoint not in protected:
        raise ValueError('Writer comparison inputs changed before reload')
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(args.checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    device = decoder.base.device
    features = FrozenTurnFeatures(decoder, tokenizer)

    @lru_cache(maxsize=None)
    def turn_features(text):
        return features([text])

    reloaded = {}
    for name, (writer_type, margin) in CONFIGS.items():
        if name == 'original':
            continue
        arm = result['arms'][name]
        writer_path = args.output / f'{name}.pt'
        writer_sha = sha(writer_path)
        if writer_sha != arm['writer_sha256']:
            raise ValueError('A candidate changed before reload')
        saved = torch.load(writer_path, map_location='cpu', weights_only=True)
        reader_path = args.reader / f'layer-{LAYER}.pt'
        if (saved['format'] != 'spora-working-state-write-revision-v1' or saved['architecture'] != name
                or saved['margin'] != margin or saved['checkpoint_sha256'] != protected[args.checkpoint]
                or saved['reader_sha256'] != protected[reader_path] or saved['feature_depth'] != FEATURE_DEPTH
                or saved['source_sha256'] != protected[REVISION]):
            raise ValueError('Candidate writer metadata changed')
        writer = writer_type(WRITER_WIDTH, feature_width=decoder.base.config.hidden_size).to(device)
        writer.load_state_dict(saved['writer'], strict=True)
        writer.eval().requires_grad_(False)
        reloaded[name] = {}
        for split, section in arm['splits'].items():
            reference = section['rows']
            rows = writes_for(writer, turn_features, tokenizer, reference, margin)
            if [row['writes'] for row in rows] != [row['writes'] for row in reference]:
                raise ValueError('Candidate writes changed after reload')
            reloaded[name][split] = len(rows)
        protected[writer_path] = writer_sha
    if (frozen_digest(runtime.organism, runtime.state) != result['individual_sha256']
            or any(sha(path) != value for path, value in protected.items())):
        raise ValueError('A protected input changed during reload')
    with (args.output / 'reload.json').open('x', encoding='utf-8') as stream:
        json.dump({'comparison_sha256': sha(result_path), 'writer_rows_identical': reloaded}, stream, indent=2)
        stream.write('\n')
    print('Reload', reloaded, flush=True)


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    writer_source = args.reader / f'layer-{LAYER}-writer.pt'
    baseline_source = args.reader / f'layer-{LAYER}-writer.json'
    original_sources = {path: sha(path) for path in (args.checkpoint, DATA)}
    protected = original_sources | {path: sha(path) for path in (writer_source, baseline_source, REVISION)}
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(args.checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    digest = frozen_digest(runtime.organism, runtime.state)
    reader_args = SimpleNamespace(output=args.reader, checkpoint=args.checkpoint)
    reader, reference, reader_path, reader_sha = load_reader(reader_args, LAYER, decoder, digest, original_sources)
    protected[reader_path] = reader_sha
    old = json.loads(baseline_source.read_text(encoding='utf-8'))
    parent = torch.load(writer_source, map_location='cpu', weights_only=True)
    if (old['writer_sha256'] != protected[writer_source] or parent['format'] != FORMAT
            or parent['checkpoint_sha256'] != protected[args.checkpoint] or parent['reader_sha256'] != reader_sha
            or parent['feature_depth'] != FEATURE_DEPTH):
        raise ValueError('Writer baseline is incompatible with the frozen reader')

    source = json.loads(DATA.read_text(encoding='utf-8'))
    train, held = revised_data(source, json.loads(REVISION.read_text(encoding='utf-8')))
    settings = settings_for(decoder, tokenizer, 48)
    splits = {'previous': reference['rows'], 'new': baselines(decoder, tokenizer, held, settings)}
    oracle = evaluate_reader(decoder, reader, tokenizer, splits['new'], settings)
    device = decoder.base.device
    features = FrozenTurnFeatures(decoder, tokenizer)
    training_features = lru_cache(maxsize=1)(features)

    @lru_cache(maxsize=None)
    def turn_features(text):
        return features([text])

    result = {'sources': {str(path): value for path, value in protected.items()}, 'individual_sha256': digest,
              'training_cases': len(train), 'epochs': 6, 'oracle': oracle,
              'new_baselines': {'off': sum(row['off']['complete'] for row in oracle),
                                'oracle': sum(row['current']['complete'] for row in oracle)}, 'arms': {}}
    for name, (writer_type, margin) in CONFIGS.items():
        torch.manual_seed(SEED)
        writer = writer_type(WRITER_WIDTH, feature_width=decoder.base.config.hidden_size).to(device)
        if name == 'original':
            writer.load_state_dict(parent['writer'], strict=True)
            trace = old['trace']
        else:
            trace = fit_writer(writer, train, source['no_write'], result['epochs'], device, features=training_features)
        writer.eval().requires_grad_(False)
        arm = {'margin': margin, 'trace': trace, 'splits': {}}
        for split, references in splits.items():
            rows = writes_for(writer, turn_features, tokenizer, references, margin)
            scores = summarize_writes(rows)
            if name == 'original' and split == 'previous' and [row['writes'] for row in rows] != [row['writes'] for row in old['rows']]:
                raise ValueError('The original writer does not reproduce its saved writes')
            scores.update(evaluate_writes(decoder, reader, tokenizer, references, settings, rows))
            sensitivity = {extra: summarize_writes(writes_for(writer, turn_features, tokenizer, references, margin + extra))
                           for extra in (-2., 2., 4.)}
            arm['splits'][split] = {'summary': scores, 'margin_offsets': sensitivity, 'rows': rows}
            print(name, split, scores, flush=True)
        if name != 'original':
            path = args.output / f'{name}.pt'
            atomic_save({'format': 'spora-working-state-write-revision-v1', 'architecture': name, 'margin': margin,
                         'checkpoint_sha256': protected[args.checkpoint], 'reader_sha256': reader_sha,
                         'feature_depth': FEATURE_DEPTH, 'source_sha256': protected[REVISION],
                         'writer': writer.cpu().state_dict()}, path)
            arm['writer_sha256'] = sha(path)
        result['arms'][name] = arm
    if frozen_digest(runtime.organism, runtime.state) != digest or any(sha(path) != value for path, value in protected.items()):
        raise ValueError('A protected working-state input changed')
    result['peak_bytes'] = torch.cuda.max_memory_allocated()
    with (args.output / 'comparison.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('reader', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--verify', action='store_true')
    options = parser.parse_args()
    (verify_writes if options.verify else main)(options)
