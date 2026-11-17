"""No-update ablation of the higher-rate continuous attention candidate."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts.deepseek_spora_continuous_attention_fast_20261117 import pilot

if len(sys.argv) != 2 or sys.argv[1] not in ('72381', '72382'):
    raise ValueError('Specify a predeclared seed: 72381 or 72382')
pilot.SEED = int(sys.argv[1])
pilot.OUT = ROOT / f'runs/deepseek-spora-continuous-attention-fast-ablation-{pilot.SEED}-20261117'
pilot.START_CANDIDATE = ROOT / 'runs/deepseek-spora-continuous-attention-fast-64-20261117/candidate.pt'
pilot.RECOVERY_FILE = None
pilot.SPAN_GRADIENT_DIAGNOSTIC = True
pilot.SPAN_GRADIENT_COMPARE = True
pilot.DIAGNOSTIC_ONLY = True
pilot.PATH_ABLATION_DIAGNOSTIC = True


if __name__ == '__main__':
    pilot.main()
