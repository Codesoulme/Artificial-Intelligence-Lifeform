"""Compare field routing with and without a recent awake topic prior."""

import argparse
import time
from pathlib import Path

from deepseek_continuous_memory_chat import load_chat
from deepseek_memory_attribute_span import KEYS, route_action
from deepseek_memory_dialogue_context import features_for, route
from deepseek_memory_inheritance_gate import OUT as GATE_RUN
from deepseek_memory_turn_priority import gate_probabilities
from deepseek_output_aligned_memory import ROOT, read, sha, torch, write
from spora.awake_topics import AwakeTopics

OUT = ROOT / 'runs/deepseek-awake-topic-recall'
PRIOR_WEIGHT = .4
MIN_ACTIVITY = .2
MIN_CONCENTRATION = .65


def topic_activity(probabilities, ticks):
    topics = AwakeTopics(KEYS)
    topics.observe(probabilities)
    for _ in range(ticks):
        topics.advance(250_000_000)
    return topics.activity


def boost_uncertain(row, cutoff, inherits, activity):
    mass = float(activity.sum())
    if not inherits or row['confidence'] >= cutoff or mass < MIN_ACTIVITY:
        return row
    prior = activity / mass
    if float(prior.max()) < MIN_CONCENTRATION:
        return row
    distribution = torch.tensor(row['probabilities']).lerp(prior, PRIOR_WEIGHT)
    return dict(row, selected=KEYS[int(distribution.argmax())], confidence=float(distribution.max()),
                probabilities=distribution.tolist())


def use_topic_without_history(row, inherits, activity):
    if not inherits:
        return row
    mass = float(activity.sum())
    if mass < MIN_ACTIVITY:
        return dict(row, confidence=0.)
    prior = activity / mass
    field = int(prior.argmax())
    confidence = float(prior[field])
    return dict(row, selected=KEYS[field], confidence=confidence if confidence >= MIN_CONCENTRATION else 0.,
                probabilities=prior.tolist())


def summarize(rows):
    return dict(rows=len(rows), field_correct=sum(row['selected'] == row['field'] for row in rows),
                correct_reads=sum(row['action'] == 'read' and row['selected'] == row['field'] for row in rows),
                wrong_reads=sum(row['action'] == 'read' and row['selected'] != row['field'] for row in rows),
                changes=sum(row['changed'] for row in rows))


@torch.inference_mode()
def main(out):
    out.mkdir()
    started = time.perf_counter()
    plan_path = GATE_RUN / 'plan.json'
    plan_sha = sha(plan_path)
    if plan_sha != read(GATE_RUN / 'result.json')['artifact_sha256'][plan_path.name]:
        raise ValueError('Inheritance evaluation plan changed')
    chat = load_chat(out / 'unused.pt', use_inherited_router=True, use_retrained_gate=True)
    suites = {}
    for name, rows in read(plan_path)['suites'].items():
        if name == 'calibration':
            continue
        results = {mode: [] for mode in ('baseline', 'recent', 'swapped', 'expired', 'current_only',
                                        'topic_only', 'topic_only_swapped', 'topic_only_expired')}
        for row in rows:
            current = features_for(chat.decoder, chat.tokenizer, row, False)
            preceding = features_for(chat.decoder, chat.tokenizer, dict(question=row['history']), False)
            key = row['key']
            gate, = gate_probabilities(chat.inheritance, [row], {key: current})
            inherits = gate >= chat.gate_cutoff
            router, features = (chat.inherited_router, preceding) if inherits else (chat.direct, current)
            baseline, = route(router, [row], {key: features})
            last_message, = route(chat.direct, [row], {key: preceding})
            current_only, = route(chat.direct, [row], {key: current})
            activity = topic_activity(last_message['probabilities'], 4)
            expired = topic_activity(last_message['probabilities'], 40)
            variants = {'baseline': baseline,
                        'recent': boost_uncertain(baseline, chat.cutoff, inherits, activity),
                        'swapped': boost_uncertain(baseline, chat.cutoff, inherits, activity.roll(1)),
                        'expired': boost_uncertain(baseline, chat.cutoff, inherits, expired),
                        'current_only': current_only,
                        'topic_only': use_topic_without_history(current_only, inherits, activity),
                        'topic_only_swapped': use_topic_without_history(current_only, inherits, activity.roll(1)),
                        'topic_only_expired': use_topic_without_history(current_only, inherits, expired)}
            baseline_action = route_action(baseline, chat.cutoff)
            for mode, result in variants.items():
                action = route_action(result, chat.cutoff)
                changed = (action, result['selected']) != (baseline_action, baseline['selected'])
                results[mode].append(dict(key=key, field=row['field'], selected=result['selected'],
                                          action=action, confidence=result['confidence'], inherits=inherits,
                                          previous_selected=last_message['selected'], changed=changed))
        suites[name] = {mode: dict(summary=summarize(results[mode]), rows=results[mode]) for mode in results}
        print(name, {mode: data['summary'] for mode, data in suites[name].items()}, flush=True)
    if sha(plan_path) != plan_sha or (out / 'unused.pt').exists():
        raise ValueError('An evaluation input or working checkpoint changed')
    write(out / 'result.json', dict(suites=suites, plan_sha256=plan_sha, prior_weight=PRIOR_WEIGHT,
                                   min_activity=MIN_ACTIVITY, min_concentration=MIN_CONCENTRATION,
                                   elapsed_seconds=time.perf_counter() - started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    main(parser.parse_args().out)
