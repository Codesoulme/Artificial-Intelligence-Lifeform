"""Apply the retrained heads to the isolated natural-route stress audit."""
import json
from pathlib import Path

from scripts import deepseek_spora_natural_route_stress_audit as stress
from scripts import deepseek_spora_natural_observation_transaction_pilot as pilot


ROOT = Path(__file__).resolve().parents[1]
HEADS = ROOT / 'runs/deepseek-spora-natural-write-coverage-retrain-20261113/heads.pt'
OUTPUT = ROOT / 'runs/deepseek-spora-natural-route-retrained-stress-audit-v2-20261113.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not HEADS.exists():
        raise FileNotFoundError(HEADS)
    stress.NATURAL_HEADS = HEADS
    stress.pilot.NATURAL_HEADS = HEADS
    stress.OUTPUT = OUTPUT
    stress.main()
    report = json.loads(OUTPUT.read_text(encoding='utf-8'))
    report['status'] = 'natural_route_retrained_stress_audit_no_writes'
    report['heads_path'] = str(HEADS)
    report['heads_sha256'] = stress.runtime.sha(HEADS)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
