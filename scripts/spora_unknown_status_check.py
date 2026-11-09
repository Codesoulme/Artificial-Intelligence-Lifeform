"""Check the read-only native known/unknown interface on a saved fresh state."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.spora_unknown_state_probe import CHECKPOINT, OUTPUT
from spora.current_memory_status import current_memory_status
from spora.growing import load
from spora.language_training import frozen_digest

RESULT = ROOT / 'runs/spora-current-status-check-20261109.json'


def run():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    with OUTPUT.open(encoding='utf-8') as stream:
        reference = json.load(stream)
    checkpoint_sha = reference['checkpoint_sha256']
    if sha(INDIVIDUAL) != INDIVIDUAL_SHA or sha(CHECKPOINT) != checkpoint_sha:
        raise ValueError('Checkpoint or original individual mismatch')
    torch.set_num_threads(2)
    model, state, _ = load(CHECKPOINT, 'cpu')
    before = frozen_digest(model, state)
    if before != reference['checkpoint_frozen_digest']:
        raise ValueError('Loaded state digest differs')
    rows = []
    for original in reference['phases']['interference']['rows']:
        observed = current_memory_status(model, state, original['key'])
        rows.append({'key': original['key'], 'expected_known': original['known_expected'],
                     'status': observed['status'], 'unknown_probability': observed['unknown_probability']})
    if frozen_digest(model, state) != before or sha(CHECKPOINT) != checkpoint_sha or sha(INDIVIDUAL) != INDIVIDUAL_SHA:
        raise ValueError('Protected input changed')
    report = {'status': 'native_current_known_unknown_development_only', 'checkpoint_sha256': checkpoint_sha,
              'individual_sha256': INDIVIDUAL_SHA, 'rows': rows,
              'false_known': sum(not r['expected_known'] and r['status'] == 'known' for r in rows),
              'false_unknown': sum(r['expected_known'] and r['status'] == 'unknown' for r in rows),
              'limitations': ['Single fresh state built from the same trained individual weights.',
                              'Truth is from controlled direct event log, used only for evaluation.',
                              'Predicted key is supplied directly; address and intent errors are not tested.',
                              'Unknown probability is uncalibrated and not a guarantee.',
                              'No historical lookup, language generation, or crash recovery is tested.']}
    with RESULT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'examples': len(rows), 'false_known': report['false_known'],
                      'false_unknown': report['false_unknown']}))


if __name__ == '__main__':
    run()
