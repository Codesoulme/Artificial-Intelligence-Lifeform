"""Test targeted write supervision and assess response-format transfer separately."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoTokenizer

from audit_spora_dialogue_working_state import (
    FEATURE_DEPTH, FORMAT, FrozenTurnFeatures, SemanticLocator, rollout_states, state_from,
    verify_date_isolation,
)
from audit_spora_ordered_working_state import OrderedRead, generate, read_alone
from scripts.audit_spora_native_query_key import write
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_dialogue_working_state import DATA, SEED, WRITER_WIDTH, dialogue_data, fit_writer
from spora.integrated_durable import load_integrated_durable
from spora.language_training import atomic_save, frozen_digest
from spora.research_dialogue_working_state import FIELDS, TurnValueWriter, WorkingTurnReader


TRANSFER = ROOT / 'data/dialogue-writer-transfer.json'


def examples_for(data, transfer):
    added = deepcopy(data)
    for kind, patterns in transfer['augmentation'].items():
        added['kinds'][kind]['train_first'] = patterns['first']
        added['kinds'][kind]['train_update'] = patterns['update']
    added['values']['train']['direction'].extend(('朝东北', '往西南', '向东偏北', '往东偏南'))
    added['values']['train']['quantity'].extend(('十五箱', '十六张', '二十册', '十七支'))
    added['values']['train']['gear'].extend(('防风镜', '绳索包', '无线电', '擦镜布'))
    old = dialogue_data(data, held=False, count=64)
    new = dialogue_data(added, held=False, count=128)
    evaluation = {row[key] for row in transfer['held'] for key in ('first', 'update', 'question')}
    training = {text for row in old + new for text in (row['first'][0], row['update'][0], row['question'])}
    if evaluation & training:
        raise ValueError('Training and evaluation turns overlap')
    return old + new


def new_cases(transfer, data):
    cases = []
    for row in transfer['held']:
        current = row['fields'] | row['changes']
        cases.append({'kind': row['kind'], 'turns': [row['first'], row['update'], row['question']],
                      'initial': row['fields'], 'changes': row['changes'], 'current': current,
                      'canonical': data['kinds'][row['kind']]['answer'].format(**current),
                      'requested': row['format'].format(**current)})
    return cases


def original_cases(data):
    return [{'kind': row['kind'], 'turns': [row['first'][0], row['update'][0], row['question']],
             'initial': row['initial'], 'changes': row['changes'], 'current': row['current'],
             'canonical': row['answer'], 'requested': row['answer']}
            for row in dialogue_data(data, held=True, count=4)]


def write_metrics(locator, tokenizer, case):
    state, writes = rollout_states(locator, tokenizer, case['turns'])
    observed = {}
    for update in writes:
        observed.update(update)
    preserved = case['initial'].keys() - case['changes'].keys()
    stale = sum(observed.get(field) == case['initial'][field] for field in case['changes'])
    return state, {'writes': writes, 'final': observed, 'state_exact': observed == case['current'],
                   'missing': sorted(case['current'].keys() - observed.keys()),
                   'wrong': sorted(field for field in observed if observed[field] != case['current'].get(field)),
                   'retained': sum(observed.get(field) == case['initial'][field] for field in preserved),
                   'retained_total': len(preserved), 'stale': stale, 'query_writes': len(writes[-1])}


def summarize(rows):
    counts = {}
    for mode in ('base', 'augmented'):
        counts[mode] = {key: sum(case[mode][key] for case in rows)
                        for key in ('state_exact', 'canonical_exact', 'requested_exact', 'stale', 'query_writes')}
    counts['oracle'] = {key: sum(case['oracle'][key] for case in rows)
                        for key in ('canonical_exact', 'requested_exact')}
    return counts


def audit(cases, tokenizer, decoder, locators, reader, glue):
    device = decoder.base.device
    rows = []

    def score(generated, case):
        return {**generated, **{key + '_exact': generated['ended'] and generated['answer'] == case[key]
                                for key in ('canonical', 'requested')}}

    for case in cases:
        row = dict(case)
        oracle = state_from(case['current'], tokenizer)
        row['oracle'] = score(generate(decoder, tokenizer, case['turns'][-1], oracle, reader, glue, 16., .95), case)
        for mode, locate in locators.items():
            state, measures = write_metrics(locate, tokenizer, case)
            answer = generate(decoder, tokenizer, case['turns'][-1], state, reader, glue, 16., .95)
            row[mode] = measures | score(answer, case)
            if any(state.slots):
                row[mode]['reader_only'] = score(read_alone(reader, tokenizer, state, glue, device), case)
        rows.append(row)
        print(case['kind'], {mode: row[mode]['state_exact'] for mode in locators}, flush=True)
    return rows


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    protected = {path: sha(path) for path in (DATA, TRANSFER, args.checkpoint, args.database, args.writer, args.reader)}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    data = json.loads(DATA.read_text(encoding='utf-8'))
    transfer = json.loads(TRANSFER.read_text(encoding='utf-8'))
    writer_payload = torch.load(args.writer, map_location='cpu', weights_only=True)
    reader_payload = torch.load(args.reader, map_location='cpu', weights_only=True)
    expected = protected[args.checkpoint]
    if ((writer_payload['format'], writer_payload['parent_sha256'], writer_payload['fields'],
         writer_payload['feature_depth'], writer_payload['writer_width']) !=
            (FORMAT, expected, FIELDS, FEATURE_DEPTH, WRITER_WIDTH)
            or (reader_payload['format'], reader_payload['parent_sha256']) != ('spora-ordered-read-v1', expected)):
        raise ValueError('Incompatible writer or reader')
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    args.output.mkdir(parents=True, exist_ok=False)
    with load_integrated_durable(args.database, args.checkpoint, load_full_streamed(), MODEL) as runtime:
        decoder = runtime.decoder.eval().requires_grad_(False)
        individual = frozen_digest(runtime.organism, runtime.state)
        device = decoder.base.device
        features = FrozenTurnFeatures(decoder, tokenizer)
        baseline = TurnValueWriter(WRITER_WIDTH, feature_width=writer_payload['hidden']).to(device)
        baseline.load_state_dict(writer_payload['writer'])
        candidate = deepcopy(baseline)
        training = examples_for(data, transfer)
        trace = fit_writer(candidate, training, data['no_write'], args.epochs, device, features=features)
        candidate.eval()
        settings = {'base': SemanticLocator(baseline.eval(), features),
                    'augmented': SemanticLocator(candidate, features)}
        glue = reader_payload['glue']
        reader = OrderedRead(len(glue)).to(device)
        reader.load_state_dict(reader_payload['model'])
        reader.eval()
        file = args.output / 'writer.pt'
        atomic_save({**writer_payload, 'writer': {key: value.cpu() for key, value in candidate.state_dict().items()}}, file)
        print('Saved candidate writer', flush=True)
        results = {}
        for split, cases in (('legacy', original_cases(data)), ('novel', new_cases(transfer, data))):
            results[split] = rows = audit(cases, tokenizer, decoder, settings, reader, glue)
            print(split, summarize(rows), flush=True)
        dates = verify_date_isolation(runtime, WorkingTurnReader(writer_payload['hidden']).to(device), tokenizer,
                                      state_from(cases[0]['current'], tokenizer))
        if frozen_digest(runtime.organism, runtime.state) != individual:
            raise ValueError('Writer probe modified the individual')
    if any(sha(path) != value for path, value in protected.items()):
        raise ValueError('An input changed during writer training')
    write(args.output / 'result.json', {'cases': results, 'scores': {s: summarize(rows) for s, rows in results.items()},
                                        'date': dates, 'digest': individual, 'trace': trace,
                                        'train_cases': len(training), 'writer_sha256': sha(file),
                                        'inputs': {str(path): value for path, value in protected.items()},
                                        'seconds': time.perf_counter() - started,
                                        'peak_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'database', 'writer', 'reader', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    main(args)
