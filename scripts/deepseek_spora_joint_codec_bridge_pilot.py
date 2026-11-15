"""Jointly train the continuous codec, memory bridge, and language adapters."""
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_semantic_bridge_pilot as pilot


class JointCodec(pilot.ContinuousTextCodec):
    """Research codec with a differentiable write-time projection."""

    def __init__(self, embedding, **kwargs):
        super().__init__(embedding, **kwargs)
        self.embedding_source = embedding.detach()
        initial = self.projection.detach().clone()
        del self._buffers['projection']
        self.projection = torch.nn.Parameter(initial)

    def encode_differentiable(self, token_ids):
        if (not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 1
                or token_ids.dtype != torch.long or not token_ids.numel()):
            raise ValueError('Differentiable codec ids must be a nonempty vector')
        if self.eos_token_id is None:
            sequence = token_ids
        else:
            if bool((token_ids == self.eos_token_id).any()):
                raise ValueError('Input must not contain codec EOS')
            sequence = torch.cat((token_ids, token_ids.new_tensor([self.eos_token_id]))).detach()
        if sequence.numel() > self.slot_count:
            raise ValueError('Differentiable sequence exceeds codec capacity')
        source_ids = sequence.detach().cpu()
        source = self.embedding_source.index_select(0, source_ids).to(self.projection.device)
        codes = F.normalize(source.float() @ self.projection, dim=-1)
        result = torch.zeros((self.slot_count, self.slot_width),
                             device=self.projection.device, dtype=codes.dtype)
        result[:sequence.numel()] = codes
        return result.reshape(-1)

    @torch.no_grad()
    def refresh_codebook(self):
        """Synchronize inference codes with the current trainable projection."""
        source = self.embedding_source.detach().cpu().float()
        projection = self.projection.detach().cpu().float()
        updated = F.normalize(source @ projection, dim=-1)
        if not bool(torch.isfinite(updated).all()):
            raise ValueError('Nonfinite refreshed codebook')
        self.codebook.copy_(updated.to(self.codebook.device))


pilot.CODEC_FACTORY = JointCodec
pilot.OUT = ROOT / 'runs/deepseek-spora-joint-codec-bridge-20261115'
pilot.JOINT_CODEC_TRAINING = True
pilot.JOINT_CODEC_LR = 1e-5
pilot.LOAD_START_BRIDGE = False
pilot.BRIDGE_RETRIEVAL_RESIDUAL = False
pilot.BRIDGE_STEPS = 400
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
pilot.TRAIN_BRIDGE = True
pilot.BRIDGE_LANGUAGE_LR = 1e-5
pilot.LANGUAGE_STEPS = 128
pilot.LANGUAGE_LR = 3e-6
pilot.TEACHER_HIDDEN_WEIGHT = 1.0
pilot.TEACHER_HIDDEN_LOGIT_WEIGHT = 0.0
pilot.TEACHER_HIDDEN_DELTA_WEIGHT = 0.0


if __name__ == '__main__':
    pilot.main()
