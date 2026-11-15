"""Distill fact-conditioned host hidden states into the memory bridge.

The student receives only the question and the supplied internal memory
snapshot. The teacher receives an explicit natural-language fact prefix. The
teacher is used only during isolated training; this does not add an answer
table or change the adopted runtime.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_semantic_bridge_pilot as pilot


pilot.OUT = pilot.p.ROOT / 'runs/deepseek-spora-hidden-bridge-distill-20261115'
pilot.CODEC_SLOT_COUNT = 5
pilot.CODEC_ADD_EOS = True
pilot.STORE_FULL_FACT = True
pilot.STORE_IDENTITY = True
pilot.BRIDGE_CONDITION_SLOTS = 4
pilot.TRAIN_VALUE_COUNT = len(pilot.WORDS)
pilot.TRAIN_CORRECTION_SEQUENCES = False
pilot.TRAIN_MULTI_ENTITY_STATES = False
pilot.TRAIN_EXTRA_QUESTIONS = True
pilot.FOCUS_EXTRA_QUESTIONS = True
pilot.EXTRA_QUESTION_TEMPLATES = (
    '请查阅记忆后告诉我{name}爱吃什么水果，只回答水果名称。',
    '关于{name}爱吃的水果，请回答名称。',
    '记忆里的{name}水果偏好是什么？只写名称。',
)
pilot.DIRECT_RESIDUAL = True
pilot.DIRECT_RESIDUAL_LAYER = 20
pilot.FINAL_RESIDUAL = True
pilot.STRICT_SWAP_EVAL = True
pilot.START_CANDIDATE = pilot.p.ROOT / 'runs/continuous-reader-final-hidden-residual-pilot-20261113/candidate.pt'
pilot.LANGUAGE_STEPS = 320
pilot.LANGUAGE_LR = 3e-6
pilot.TEACHER_HIDDEN_WEIGHT = 1.0
pilot.TEACHER_HIDDEN_LOGIT_WEIGHT = 0.0


if __name__ == '__main__':
    pilot.main()
