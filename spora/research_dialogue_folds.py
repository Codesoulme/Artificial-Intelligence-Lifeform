"""Source-purged research folds, including borrowed conversation histories."""
import hashlib


def make_folds(rows, message_to_tree, folds=5):
    if folds < 3:
        raise ValueError('At least three folds required')
    cases = {}
    for row in rows:
        case = row['case']
        value = (row['group'], row['history_dependent'])
        if case in cases and cases[case] != value:
            raise ValueError('Inconsistent case')
        cases[case] = value
    if len({v[0] for v in cases.values()}) != len(cases):
        raise ValueError('Cases must have unique source trees')
    assignment = {}
    for label in (0, 1):
        ordered = sorted([c for c, (_, y) in cases.items() if y == label],
                         key=lambda c: hashlib.sha256(cases[c][0].encode()).hexdigest())
        if len(ordered) < folds:
            raise ValueError('Not enough cases per class')
        assignment.update({c: i % folds for i, c in enumerate(ordered)})

    def sources(row):
        ids = [row['current_message_id']] + row['history_message_ids']
        try:
            trees = {message_to_tree[key] for key in ids}
        except KeyError as exc:
            raise ValueError('Unknown provenance') from exc
        if row['group'] not in trees:
            raise ValueError('Current source mismatch')
        return trees

    provenance = [sources(r) for r in rows]
    results = []
    for fold in range(folds):
        test = [i for i, r in enumerate(rows) if assignment[r['case']] == fold]
        blocked = set().union(*(provenance[i] for i in test))
        nominal_cal = [i for i, r in enumerate(rows) if assignment[r['case']] == (fold+1) % folds]
        calibration = [i for i in nominal_cal if not provenance[i] & blocked]
        cal_sources = set().union(*(provenance[i] for i in calibration))
        nominal_train = [i for i, r in enumerate(rows) if assignment[r['case']] not in (fold, (fold+1) % folds)]
        train = [i for i in nominal_train if not provenance[i] & (blocked | cal_sources)]
        for name, indices in [('train', train), ('calibration', calibration), ('test', test)]:
            if {rows[i]['need'] for i in indices} != {0, 1}:
                raise ValueError('Both classes missing in ' + name)
        train_sources = set().union(*(provenance[i] for i in train))
        if train_sources & blocked or train_sources & cal_sources or cal_sources & blocked:
            raise ValueError('Source tree leakage')
        results.append({'fold': fold, 'train': train, 'calibration': calibration, 'test': test,
                        'purged_train_rows': len(nominal_train)-len(train),
                        'purged_calibration_rows': len(nominal_cal)-len(calibration),
                        'source_trees': {'train': sorted(train_sources), 'calibration': sorted(cal_sources),
                                         'test': sorted(blocked)}})
    if sorted(i for fold in results for i in fold['test']) != list(range(len(rows))):
        raise ValueError('Test coverage mismatch')
    return results
