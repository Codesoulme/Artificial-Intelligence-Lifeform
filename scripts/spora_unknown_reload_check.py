"""Reopen the fresh-state probe in a separate Python process and compare all keys."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.spora_unknown_state_probe import CHECKPOINT, OUTPUT, measure
from spora.growing import load
from spora.language_training import frozen_digest

RESULT = ROOT / 'runs/spora-unknown-separate-process-20261109.json'


def run():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    with OUTPUT.open(encoding='utf-8') as stream:
        previous = json.load(stream)
    checkpoint_sha = previous['checkpoint_sha256']
    if sha(INDIVIDUAL) != INDIVIDUAL_SHA or sha(CHECKPOINT) != checkpoint_sha:
        raise ValueError('Checkpoint or original individual mismatch')
    torch.set_num_threads(2)
    model, state, _ = load(CHECKPOINT, 'cpu')
    digest = frozen_digest(model, state)
    if digest != previous['checkpoint_frozen_digest']:
        raise ValueError('Loaded state digest differs')
    expected = {row['key']: row['expected'] for row in previous['phases']['interference']['rows']
                if row['known_expected']}
    observed = measure(model, state, expected)
    identical = observed['rows'] == previous['phases']['interference']['rows']
    if not identical or sha(CHECKPOINT) != checkpoint_sha or sha(INDIVIDUAL) != INDIVIDUAL_SHA:
        raise ValueError('Separate-process result or source differs')
    report = {'status': 'separate_process_native_reload_only', 'checkpoint_sha256': checkpoint_sha,
              'individual_sha256': INDIVIDUAL_SHA, 'frozen_digest': digest, 'identical_rows': identical,
              'summary': {k: v for k, v in observed.items() if k != 'rows'},
              'limitations': ['One fresh state and same trained model, not arbitrary independent callers.',
                              'No language-model interface or crash recovery is tested.',
                              'Reference labels are used for evaluation only, not fed to the model.']}
    with RESULT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report['summary']))


if __name__ == '__main__':
    run()
