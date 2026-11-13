"""Continue from the same candidate while oversampling previously trained facts."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_attention_expanded_values_pilot as configuration

pilot = configuration.pilot
pilot.audit.OUT = ROOT / 'runs/deepseek-spora-continual-replay-pilot-20261113'
pilot.INIT_FROM = ROOT / 'runs/deepseek-spora-attention-higher-lr-pilot-20261113'
pilot.EXPECTED_INITIAL_RESULT = ROOT / 'runs/deepseek-spora-attention-expanded-baseline-20261113/reload-result.json'
pilot.REPLAY_FACTOR = 2

if __name__ == '__main__':
    pilot.audit.main()
    if not pilot.done:
        raise ValueError('Replay continual pilot did not run')
