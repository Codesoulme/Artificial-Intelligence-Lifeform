"""Evaluate a nonlinear memory-code decompressor on the current bridge."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_semantic_bridge_pilot as pilot


class TokenCodeDecoder(pilot.p.torch.nn.Module):
    def __init__(self, slot_width, embedding_width):
        super().__init__()
        self.net = pilot.p.torch.nn.Sequential(
            pilot.p.torch.nn.Linear(slot_width, 512),
            pilot.p.torch.nn.GELU(),
            pilot.p.torch.nn.Linear(512, embedding_width),
        )
        pilot.p.torch.nn.init.xavier_uniform_(self.net[0].weight)
        pilot.p.torch.nn.init.xavier_uniform_(self.net[2].weight)

    def forward(self, codes):
        return pilot.p.torch.nn.functional.normalize(self.net(codes.float()), dim=-1)


pilot.TokenCodeDecoder = TokenCodeDecoder
pilot.OUT = ROOT / 'runs/deepseek-spora-mlp-hidden-bridge-20261115'
pilot.BRIDGE_STEPS = 600
pilot.LOAD_START_BRIDGE = False
pilot.CODEC_SLOT_COUNT = 5
pilot.CODEC_ADD_EOS = True
pilot.STORE_FULL_FACT = True
pilot.STORE_IDENTITY = True
pilot.BRIDGE_CONDITION_SLOTS = 4
pilot.BRIDGE_CONDITION_OFFSET = 0
pilot.REMAP_CONDITION_INIT = False
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
pilot.START_CANDIDATE = ROOT / 'runs/continuous-reader-final-hidden-residual-pilot-20261113/candidate.pt'
pilot.LANGUAGE_STEPS = 320
pilot.LANGUAGE_LR = 3e-6
pilot.TEACHER_HIDDEN_WEIGHT = 1.0
pilot.TEACHER_HIDDEN_LOGIT_WEIGHT = 0.0
pilot.TEACHER_HIDDEN_DELTA_WEIGHT = 0.0


if __name__ == '__main__':
    pilot.main()
