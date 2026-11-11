"""Compare constraint-trained pair with its parent on prior reviewed prompts."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import deepseek_spora_pair_oasst_reviewed_eval as audit

audit.PAIR = ROOT / 'runs/deepseek-spora-pair-general-replay-20261112'
audit.CANDIDATE = ROOT / 'runs/deepseek-spora-pair-constraints-pilot-20261112'
audit.OUTPUT = ROOT / 'runs/deepseek-spora-pair-constraints-oasst-eval-20261112.json'

if __name__ == '__main__':
    audit.main()
