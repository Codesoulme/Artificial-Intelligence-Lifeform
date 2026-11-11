"""Compare the answer-distilled structural candidates on paired memory values."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import deepseek_spora_counterfactual_memory_audit as audit

audit.CANDIDATE_SOURCE = ROOT/'runs/deepseek-spora-memory-answer-distill-v4-20261111'
audit.OUTPUT = ROOT/'runs/deepseek-spora-counterfactual-memory-audit-v3-20261111.json'
audit.PARTIAL = ROOT/'runs/deepseek-spora-counterfactual-memory-audit-v3-20261111.jsonl'

if __name__ == '__main__':
    audit.run()
