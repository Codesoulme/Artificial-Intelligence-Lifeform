"""Re-run the real observation route with explicit calibrated write authority.

The imported route still performs all subject, polarity and value checks. This
wrapper only separates the calibrated gate score from the parser's raw score;
the audit never persists a candidate.
"""
import json
from pathlib import Path

from scripts import deepseek_spora_end_to_end_candidate_audit as route
from spora.observation_candidate import calibrated_write_guard


OUTPUT = route.ROOT / 'runs/deepseek-spora-calibrated-observation-route-audit-20261113.json'
POLICY_ID = 'spora-polarity-calibration-20261111-v1'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    gate = route.torch.load(route.POLARITY / 'gate.pt', map_location='cpu', weights_only=True)
    threshold = float(gate['threshold'])
    original = route.write_guard

    def guarded(candidate, *, key_count, value_count, current_cursor, minimum_confidence=0.9):
        # The route's polarity model selected this threshold on its calibration
        # split with zero negative accepts. The later route code still applies
        # its independent polarity and value gates to the same candidate.
        return calibrated_write_guard(
            candidate,
            {'policy_id': POLICY_ID,
             'authorized': candidate.confidence >= threshold,
             'confidence': 1.0 if candidate.confidence >= threshold else 0.0},
            key_count=key_count, value_count=value_count,
            current_cursor=current_cursor, policy_id=POLICY_ID)

    route.write_guard = guarded
    route.OUTPUT = OUTPUT
    route.main()
    report = json.loads(OUTPUT.read_text(encoding='utf-8'))
    report['status'] = 'calibrated_observation_route_audit_no_writes'
    report['calibration'] = {'policy_id': POLICY_ID, 'threshold': threshold,
                             'source_gate_sha256': route.sha(route.POLARITY / 'gate.pt'),
                             'raw_write_guard': original.__name__}
    report['limitations'].append('Authorization threshold is calibrated only on the finite development split; no production write was performed.')
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
