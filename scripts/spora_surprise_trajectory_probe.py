"""Measure novelty labels across new, correction, repeat, and quality updates."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from spora.growing import new_state, observe
from spora.internal_memory import WriteControl, attach, observe_sourced
from spora.temporal import TemporalConfig, TemporalSPORA

OUTPUT = ROOT / 'runs/spora-surprise-trajectory-20261111.json'


def force_action(model, action):
    with torch.no_grad():
        for parameter in model.write_control.parameters():
            parameter.zero_()
        model.write_control.network[-1].bias[action] = 20


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.manual_seed(90117)
    torch.set_num_threads(2)
    model = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2, memory_width=64,
                                         expert_rank=4, memory_layers=(0, 1),
                                         history_writer='rls', current_address='orthogonal'))
    state = attach(model, new_state(model), WriteControl())
    events = []
    for label, action, value, quality in (
            ('new', 0, 2, .4), ('quality_upgrade', 2, 2, .9),
            ('correction', 1, 3, .9), ('repeat', 3, 3, .9)):
        force_action(model, action)
        state, decision = observe_sourced(model, state, 1, value, 'direct', quality)
        events.append({'label': label, 'value': value, 'quality': quality,
                       'action': decision['action'], 'current_write': decision['current_write'],
                       'novelty': decision['novelty'],
                       'novelty_components': decision['novelty_components'],
                       'cursor': state.cursor})
    report = {'status': 'surprise_trajectory_development_probe', 'events': events,
              'expected_novelty_order': ['new', 'correction', 'quality_upgrade', 'repeat'],
              'limitations': ['Synthetic controlled actions and one key.',
                              'Controller actions are forced to isolate novelty labels.',
                              'No language parser or learned novelty controller was evaluated.']}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(events, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
