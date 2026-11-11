"""Research temporal head with replay distillation from the original head."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch import nn
from torch.nn import functional as F
from scripts import deepseek_spora_temporal_gate_v2 as previous
from scripts.deepseek_local_baseline import sha
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_named_binding_probe import NAMES
from spora.temporal_routing import PAST

trainer = previous.trainer
trainer.OUTPUT = ROOT / 'runs/deepseek-spora-temporal-gate-v3-20261112'
trainer.SEED = 510249
trainer.STEPS = 200
TEACHER_DIR = ROOT / 'runs/deepseek-spora-temporal-gate-train-20261111'
DISTILL_WEIGHT = 2.0
TEMPERATURE = 2.0
captured = {}


def extract_with_replay(base, tokenizer, questions, batch_size=4):
    features, masks = previous.low_memory_extract(base, tokenizer, questions, batch_size=1)
    added_templates = trainer.EXTRA_TRAIN[PAST][-5:] + trainer.TRAIN_ABSTAIN[-4:]
    added_questions = {t.format(name=name) for t in added_templates for name in NAMES[:6]}
    captured['vectors'] = pool(features, masks).float()
    captured['old_mask'] = torch.tensor([q not in added_questions for q in questions], dtype=torch.bool)
    return features, masks


def main():
    path = TEACHER_DIR / 'temporal.pt'
    info = json.loads((TEACHER_DIR / 'result.json').read_text(encoding='utf-8'))
    if sha(path) != info['temporal_sha256']:
        raise ValueError('Teacher digest mismatch')
    teacher_data = torch.load(path, map_location='cpu', weights_only=True)
    if teacher_data['base_sha256'] != trainer.WEIGHT_SHA:
        raise ValueError('Teacher base mismatch')
    teacher = nn.Linear(teacher_data['state']['weight'].shape[1], 4).eval()
    teacher.load_state_dict(teacher_data['state'], strict=True)
    teacher.requires_grad_(False)

    def initialized_linear(width, classes):
        if width != teacher.in_features or classes != 4:
            raise ValueError('Unexpected temporal head dimensions')
        student = nn.Linear(width, classes)
        student.load_state_dict(teacher.state_dict(), strict=True)
        return student

    def replay_loss(logits, labels):
        supervised = F.cross_entropy(logits, labels)
        mask = captured['old_mask'][:len(labels)]
        vectors = captured['vectors'][:len(labels)]
        if mask.shape[0] != logits.shape[0] or not bool(mask.any()) or bool(mask.all()):
            raise ValueError('Invalid replay partition')
        with torch.no_grad():
            reference = F.softmax(teacher(vectors[mask]) / TEMPERATURE, dim=-1)
        divergence = F.kl_div(F.log_softmax(logits[mask] / TEMPERATURE, dim=-1), reference,
                              reduction='batchmean') * TEMPERATURE**2
        return supervised + DISTILL_WEIGHT * divergence

    with patch.object(trainer.AutoModelForCausalLM, 'from_pretrained', side_effect=previous.streamed_base), \
            patch.object(trainer, 'extract', side_effect=extract_with_replay), \
            patch.object(trainer, 'nn', SimpleNamespace(Linear=initialized_linear)), \
            patch.object(trainer, 'F', SimpleNamespace(cross_entropy=replay_loss)):
        trainer.run()


if __name__ == '__main__':
    main()
