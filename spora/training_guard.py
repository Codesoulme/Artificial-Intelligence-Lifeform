"""Optional development regression checks for bounded training runs."""
import math


def validate_guard(config):
    if config is None:
        return
    if not isinstance(config, dict) or set(config) != {'max_dialogue_nll_increase', 'max_ended_drop'}:
        raise ValueError('Invalid regression guard configuration')
    nll, drop = config['max_dialogue_nll_increase'], config['max_ended_drop']
    if (type(nll) not in (int, float) or not math.isfinite(nll) or nll < 0
            or type(drop) is not int or drop < 0):
        raise ValueError('Invalid regression guard thresholds')


def assess_guard(config, baseline, current):
    validate_guard(config)
    if config is None:
        return None
    for report in (baseline, current):
        if not isinstance(report, dict):
            raise ValueError('Missing guard report')
        nll, samples = report.get('dialogue_nll'), report.get('samples')
        if (type(nll) not in (int, float) or not math.isfinite(nll) or not isinstance(samples, list)
                or not samples or any(not isinstance(s, dict) or not isinstance(s.get('prompt'), str)
                                      or type(s.get('ended_with_eos')) is not bool for s in samples)):
            raise ValueError('Invalid guard report')
    if [s['prompt'] for s in baseline['samples']] != [s['prompt'] for s in current['samples']]:
        raise ValueError('Guard prompts differ from baseline')
    delta = current['dialogue_nll'] - baseline['dialogue_nll']
    ended_drop = (sum(s['ended_with_eos'] for s in baseline['samples'])
                  - sum(s['ended_with_eos'] for s in current['samples']))
    reasons = []
    if delta > config['max_dialogue_nll_increase']:
        reasons.append('dialogue_nll_increase')
    if ended_drop > config['max_ended_drop']:
        reasons.append('ended_count_drop')
    return {'stop': bool(reasons), 'reasons': reasons,
            'dialogue_nll_increase': delta, 'ended_count_drop': ended_drop}
