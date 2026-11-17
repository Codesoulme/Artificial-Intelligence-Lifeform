"""Reload the gain-10 research candidate and compare six saved generations."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts.deepseek_spora_continuous_attention_gain10_20261117 import pilot
from scripts import deepseek_spora_bridge_reload_audit as audit
from scripts.deepseek_multitoken_strict_report_20261117 import VALUES
from spora.research_relation_scoring import score_relation_answer

audit.SOURCE = ROOT / 'runs/deepseek-spora-continuous-attention-gain10-64-20261117'
audit.OUT = ROOT / 'runs/deepseek-spora-continuous-attention-gain10-reload-20261117'
audit.PARITY_VALUE_INDICES = (4, 5, 10, 11, 16, 17)
audit.QUESTION_BY_VALUE = tuple(pilot.relation_for_value(i)[1][0] for i in range(len(pilot.WORDS)))
audit.VERIFY_CODEC_READS = True
audit.EXTRA_ONLY = True
audit.CROSS_ABLATION = True
audit.correct_answer = lambda answer, name, value: score_relation_answer(answer, name, VALUES[value], value)


if __name__ == '__main__':
    audit.main()
