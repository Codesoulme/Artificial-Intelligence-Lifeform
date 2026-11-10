"""Compare saved predictions with a small, agent-annotated dialogue subset."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts.deepseek_relative_identity_probe import ADDRESSEE, SPEAKER, UNKNOWN
from scripts.open_base_baseline import sha

SOURCE = ROOT / 'runs/deepseek-dulemon-persona-audit-20261110.json'
SOURCE_SHA = '7b38a1f44662a5be7813d0f2dd76b24e6bc0e7161a3a9993953b6b1fb0fb7d1a'
OUTPUT = ROOT / 'runs/deepseek-dulemon-subject-annotation-20261110.json'

# First 30 rows in the frozen SHA-sorted audit. Labels target the persona-linked
# assertion; ambiguous, third-person and non-assertion cases are deferred.
LABELS = (
    (SPEAKER, 'implicit occupation answer'),
    (ADDRESSEE, 'recipient character and job'),
    (UNKNOWN, 'third-person income'),
    (ADDRESSEE, 'recipient graduation'),
    (ADDRESSEE, 'recipient workplace'),
    (ADDRESSEE, 'recipient eyesight'),
    (SPEAKER, 'speaker cooking preference'),
    (SPEAKER, 'speaker swimming preference'),
    (SPEAKER, 'speaker income'),
    (SPEAKER, 'speaker employment'),
    (ADDRESSEE, 'recipient travel'),
    (SPEAKER, 'speaker friends'),
    (ADDRESSEE, 'recipient studies'),
    (UNKNOWN, 'both speaker and recipient travel'),
    (UNKNOWN, 'parents rather than participants'),
    (ADDRESSEE, 'recipient suggested activity'),
    (UNKNOWN, 'both speaker and recipient'),
    (SPEAKER, 'speaker coffee habit'),
    (UNKNOWN, 'addressed name and speaker availability'),
    (ADDRESSEE, 'recipient inferred from previous turn'),
    (ADDRESSEE, 'recipient spending'),
    (ADDRESSEE, 'recipient environmental view'),
    (ADDRESSEE, 'recipient inferred from previous turn'),
    (SPEAKER, 'speaker ambition'),
    (UNKNOWN, 'greeting without a persona assertion'),
    (SPEAKER, 'speaker age'),
    (SPEAKER, 'speaker reading habit'),
    (SPEAKER, 'speaker food preference'),
    (UNKNOWN, 'speaker dislike and recipient question'),
    (ADDRESSEE, 'recipient car and activity'),
)


def compare(rows, labels):
    if len(rows) != len(labels):
        raise ValueError('Expected one annotation per row')
    scored = []
    for row, (label, reason) in zip(rows, labels):
        prediction = row['predicted']
        if label not in (SPEAKER, ADDRESSEE, UNKNOWN) or prediction not in (SPEAKER, ADDRESSEE, UNKNOWN):
            raise ValueError('Invalid subject class')
        scored.append({'sample_sha256': row['sample_sha256'], 'label': label, 'reason': reason,
                       'predicted': prediction, 'correct': prediction == label})
    explicit = [row for row in scored if row['label'] != UNKNOWN]
    ambiguous = [row for row in scored if row['label'] == UNKNOWN]
    return {'examples': len(scored), 'correct': sum(row['correct'] for row in scored),
            'single_subject': len(explicit), 'single_correct': sum(row['correct'] for row in explicit),
            'single_deferred': sum(row['predicted'] == UNKNOWN for row in explicit),
            'wrong_person': sum(row['predicted'] not in (row['label'], UNKNOWN) for row in explicit),
            'defer_expected': len(ambiguous), 'unsafe_accept': sum(row['predicted'] != UNKNOWN for row in ambiguous),
            'rows': scored}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Source audit fingerprint mismatch')
    with SOURCE.open(encoding='utf-8') as stream:
        source = json.load(stream)
    reference = source['results']['single']['baseline']['rows'][:len(LABELS)]
    if len(reference) != len(LABELS):
        raise ValueError('Source audit is incomplete')
    identifiers = [row['sample_sha256'] for row in reference]
    if len(set(identifiers)) != len(identifiers) or identifiers != sorted(identifiers):
        raise ValueError('Unexpected sample order')
    results = {}
    for condition in ('single', 'previous'):
        results[condition] = {}
        for model in ('baseline', 'expanded'):
            rows = source['results'][condition][model]['rows'][:len(LABELS)]
            if [row['sample_sha256'] for row in rows] != identifiers:
                raise ValueError('Condition sample mismatch')
            results[condition][model] = compare(rows, LABELS)
    report = {'status': 'agent_annotated_development_audit', 'source_sha256': SOURCE_SHA,
              'annotation': 'Codex assessed the persona-linked assertion in current and prior turn; no independent human review.',
              'results': results,
              'limitations': ['Thirty SHA-ordered samples selected from an existing 80-sample development audit.',
                              'Task-specific subjective labels may be wrong or underspecified; no inter-annotator agreement.',
                              'A persona-linked subject is not proof of truth, source reliability, or permission to write.',
                              'Previous-turn prompt format was not part of classifier training.',
                              'Model pretraining overlap is unknown; no open-domain or multi-turn performance claim.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({condition: {model: {k: v for k, v in result.items() if k != 'rows'}
                                      for model, result in models.items()}
                      for condition, models in results.items()}), flush=True)


if __name__ == '__main__':
    run()
