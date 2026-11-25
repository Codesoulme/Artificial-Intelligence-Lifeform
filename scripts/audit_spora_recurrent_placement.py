"""Verify comparability and summarize two bounded-attention placements."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verified_sources(sources):
    for path, expected in sources.items():
        if digest(Path(path)) != expected:
            raise ValueError(f'Input changed: {path}')


def references(paths):
    first, second = reports = tuple(map(read, paths))
    for report in reports:
        verified_sources(report['sources'])
    if (any(len(report['layers']) != 2 or len(set(report['layers'])) != 2 for report in reports)
            or first['layers'] == second['layers']):
        raise ValueError('Expected two distinct placements of exactly two layers')
    for key in ('position_table', 'individual_sha256'):
        if first[key] != second[key]:
            raise ValueError(f'Layouts have different {key}')
    shared = first['sources'].keys() & second['sources'].keys()
    required_files = {'candidate.pt', 'recurrent-attention-dialogue-audit-v1.json'}
    if (not required_files <= {Path(path).name for path in shared}
            or any(first['sources'][path] != second['sources'][path] for path in shared)):
        raise ValueError('The layouts use different parent weights or prompts')
    if first['dialogue']['parent'] != second['dialogue']['parent']:
        raise ValueError('Complete-attention outputs differ across references')
    return reports


def outcome(result):
    recall = Counter()
    for row in result['recall']:
        length, text = row['length'], row['answer']
        recall.update({(length, 'count'): 1, (length, 'new'): row['values']['new'] in text,
                       (length, 'old'): row['values']['old'] in text, (length, 'keep'): row['values']['keep'] in text,
                       (length, 'complete'): row['passed']})
    summary = {length: {item: value for (label, item), value in recall.items() if label == length}
               for length in result['recall_passed']}
    return {'recall': summary, 'open': [result['open_passed'], result['open_turns_passed']],
            'counterfactual': result.get('counterfactual')}


def recovery(reference, reference_path, directory):
    result = read(directory / 'result.json')
    training = read(directory / 'training.json')
    if any(result[key] != value for key, value in training.items()):
        raise ValueError(f'The results use different training weights: {directory}')
    direct_names = {'candidate.pt', 'recurrent-attention-dialogue-audit-v1.json', 'attention.pt', 'comparison.json'}
    if (training['individual_sha256'] != reference['individual_sha256']
            or training['sources'][str(reference_path)] != digest(reference_path)
            or any(training['sources'].get(path) != expected for path, expected in reference['sources'].items()
                   if Path(path).name in direct_names)):
        raise ValueError(f'Recovery uses a different reference: {directory}')
    verified_sources(training['sources'])
    trained = {int(layer): value for layer, value in training['candidate_hashes'].items()}
    if (set(trained) != set(reference['layers'])
            or any(digest(directory / f'layer-{layer}' / 'attention.pt') != value for layer, value in trained.items())):
        raise ValueError(f'Recovery weights changed: {directory}')
    return result


def training_schedule(result):
    return [(row['step'], row['pair_index'], row['tokens']) for row in result['trace']]


def recall_inputs(result):
    fields = ('case', 'variant', 'length', 'values', 'question', 'history', 'input_tokens')
    return [tuple(row[key] for key in fields) for row in result['recall']]


def crossed_effects(paths):
    anchor = paths[0]
    reports = [references((anchor, path))[1] for path in paths[1:]]
    reports.insert(0, read(anchor))
    layouts = {tuple(row['layers']): row['dialogue']['recurrent'] for row in reports}
    first_slots = sorted({first for first, _ in layouts})
    second_slots = sorted({second for _, second in layouts})
    if len(layouts) != 4 or len(first_slots) != 2 or len(second_slots) != 2:
        raise ValueError('Expected all four combinations of two first and two second layers')
    inputs = recall_inputs(reports[0]['dialogue']['recurrent'])
    weights = {}
    for report in reports:
        if recall_inputs(report['dialogue']['recurrent']) != inputs:
            raise ValueError('The crossed layouts used different histories')
        candidates = {}
        for source in report['sources']:
            path = Path(source)
            if path.name == 'comparison.json':
                layer = read(path)['layer']
                candidates[layer] = report['sources'][str(path.with_name('attention.pt'))]
        if set(candidates) != set(report['layers']):
            raise ValueError('Candidate sources and layer placement differ')
        for layer, digest in candidates.items():
            if weights.setdefault(layer, digest) != digest:
                raise ValueError(f'Layout changed the initial weights for layer {layer}')

    def move(before, after):
        rows = {}
        for left, right in zip(before['recall'], after['recall'], strict=True):
            count = rows.setdefault(left['length'], Counter())
            for key, old, new in (
                ('complete', left['passed'], right['passed']),
                ('new', left['values']['new'] in left['answer'], right['values']['new'] in right['answer']),
                ('keep', left['values']['keep'] in left['answer'], right['values']['keep'] in right['answer']),
            ):
                count[f'{key}_gain'] += int(new and not old)
                count[f'{key}_loss'] += int(old and not new)
        return rows

    first, alternate = first_slots
    second, other = second_slots
    return {'layouts': {','.join(map(str, placement)): outcome(row) for placement, row in layouts.items()},
            'replace_first': {partner: move(layouts[first, partner], layouts[alternate, partner])
                              for partner in second_slots},
            'replace_second': {partner: move(layouts[partner, second], layouts[partner, other])
                               for partner in first_slots}}


def main(args):
    if args.crossed:
        print(json.dumps(crossed_effects([*args.references, *args.crossed]), ensure_ascii=False, indent=2))
        return
    first, second = reports = references(args.references)
    result = {'parent': outcome(first['dialogue']['parent']),
              'before': {','.join(map(str, row['layers'])): outcome(row['dialogue']['recurrent']) for row in reports},
              'individual_sha256': first['individual_sha256'], 'positions': second['position_table']}
    if args.recoveries:
        old, new = recovered = tuple(recovery(report, path, directory) for report, path, directory in
                                     zip(reports, args.references, args.recoveries, strict=True))
        fields = ('training_pairs', 'steps', 'learning_rate')
        old_schedule, new_schedule = training_schedule(old), training_schedule(new)
        if (any(old[key] != new[key] for key in fields) or old.get('field_weight', 0.) != new.get('field_weight', 0.)
                or old_schedule != new_schedule or len(new_schedule) != new['steps']):
            raise ValueError('Recovery settings or ordered training pairs differ')
        keys = old['sources'].keys() & new['sources'].keys()
        if not {'pretrain-train.jsonl', 'manifest.json', 'dialogue-working-state-v1.json'} <= {
            Path(path).name for path in keys
        } or any(old['sources'][path] != new['sources'][path] for path in keys):
            raise ValueError('Recovery corpus or parent differs')
        paired_inputs = recall_inputs(old['paired_before'])
        if (any(recall_inputs(row[phase]) != paired_inputs for row in recovered
                for phase in ('paired_before', 'paired_after'))
                or recall_inputs(old['original_cases_after']) != recall_inputs(new['original_cases_after'])):
            raise ValueError('Recovery evaluation used different dialogue histories')
        result['training'] = {key: new[key] for key in fields}
        result['after'] = {
            ','.join(map(str, report['layers'])): {
                'paired_before': outcome(row['paired_before']), 'paired_after': outcome(row['paired_after']),
                'original_cases_after': outcome(row['original_cases_after']), 'memory': row['memory_after']['summary'],
                'seconds': row['seconds'], 'training_peak_bytes': row['training_peak_bytes'],
            }
            for report, row in zip(reports, recovered, strict=True)
        }
        result['memory_identical'] = old['memory_after'] == new['memory_after']
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('references', nargs=2, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--recoveries', nargs=2, type=Path)
    mode.add_argument('--crossed', nargs=2, type=Path, help='Compare four fixed-budget layouts that share layer weights')
    main(parser.parse_args())
