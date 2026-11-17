"""Same-budget natural QA training with question-first input ordering."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import deepseek_cmrc_reading_train_pilot_20261117 as pilot
from scripts.deepseek_cmrc_question_first_probe_20261117 import question_first

ORIGINAL_EXTRACT = pilot.extract


def ordered_extract(chat, rows):
    # Natural reading rows use this explicit boundary; synthetic replay remains
    # unchanged to preserve the original negative-control task format.
    transformed = [question_first(row) if '\n问题：' in row['text'] else row for row in rows]
    return ORIGINAL_EXTRACT(chat, transformed)


def main():
    pilot.OUT = ROOT / 'runs/deepseek-cmrc-question-first-training-20261117'
    pilot.extract = ordered_extract
    pilot.main()


if __name__ == '__main__':
    main()
