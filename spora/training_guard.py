"""Optional development regression checks for bounded training runs."""
import math
import re


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


def validate_multiturn_guard(config):
    if config is None:
        return
    required = {'cases', 'cases_sha256', 'budget', 'max_ended_drop', 'max_loop_increase'}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError('Invalid multi-turn guard configuration')
    if (not isinstance(config['cases'], str) or not config['cases']
            or not isinstance(config['cases_sha256'], str)
            or re.fullmatch(r'[0-9a-f]{64}', config['cases_sha256']) is None
            or type(config['budget']) is not int or config['budget'] < 1
            or type(config['max_ended_drop']) is not int or config['max_ended_drop'] < 0
            or type(config['max_loop_increase']) is not int or config['max_loop_increase'] < 0):
        raise ValueError('Invalid multi-turn guard thresholds')


def assess_multiturn_guard(config, baseline, current):
    validate_multiturn_guard(config)
    if config is None:
        return None
    for report in (baseline, current):
        if not isinstance(report, dict) or set(report) != {'turns', 'ended', 'loops'}:
            raise ValueError('Invalid multi-turn guard report')
        if any(type(report[k]) is not int for k in report) or not (
                report['turns'] > 0 and 0 <= report['ended'] <= report['turns']
                and 0 <= report['loops'] <= report['turns']):
            raise ValueError('Invalid multi-turn guard counts')
    if baseline['turns'] != current['turns']:
        raise ValueError('Multi-turn guard counts differ')
    ended_drop = baseline['ended'] - current['ended']
    loop_increase = current['loops'] - baseline['loops']
    reasons = []
    if ended_drop > config['max_ended_drop']:
        reasons.append('multiturn_ended_drop')
    if loop_increase > config['max_loop_increase']:
        reasons.append('multiturn_loop_increase')
    return {'stop': bool(reasons), 'reasons': reasons,
            'ended_count_drop': ended_drop, 'loop_count_increase': loop_increase}
