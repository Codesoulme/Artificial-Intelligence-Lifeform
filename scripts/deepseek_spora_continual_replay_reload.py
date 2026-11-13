"""Reload and ablate the replay candidate in a separate process."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_continual_replay_pilot as configuration

pilot = configuration.pilot
pilot.RELOAD_FROM = ROOT / 'runs/deepseek-spora-continual-replay-pilot-20261113'
pilot.audit.OUT = ROOT / 'runs/deepseek-spora-continual-replay-reload-20261113'
pilot.EVAL_ATTENTION_ABLATION = True

if __name__ == '__main__':
    pilot.audit.main()
    if not pilot.done:
        raise ValueError('Replay candidate reload did not run')
