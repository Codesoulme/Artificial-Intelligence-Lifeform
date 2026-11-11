"""Full-batch replay gradient projection for a research-only time head."""
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
from scripts.deepseek_named_binding_probe import NAMES
from spora.gradient_constraints import project_against_reference
from spora.temporal_routing import PAST

trainer = previous.trainer
trainer.OUTPUT = ROOT / 'runs/deepseek-spora-temporal-gate-v4-projected-20261112'
trainer.SEED = 510250
trainer.STEPS = 200
TEACHER_DIR = ROOT / 'runs/deepseek-spora-temporal-gate-train-20261111'
learning_rate = 0.001
captured = {'projected_steps': 0, 'steps': 0, 'negative_dots_after': 0}
sgd = torch.optim.SGD


def extract_with_partition(base, tokenizer, questions, batch_size=4):
    features, masks = previous.low_memory_extract(base, tokenizer, questions, batch_size=1)
    added_templates = trainer.EXTRA_TRAIN[PAST][-5:] + trainer.TRAIN_ABSTAIN[-4:]
    added = {t.format(name=name) for t in added_templates for name in NAMES[:6]}
    captured['old_mask'] = torch.tensor([q not in added for q in questions], dtype=torch.bool)
    return features, masks


def main():
    path = TEACHER_DIR / 'temporal.pt'
    info = json.loads((TEACHER_DIR / 'result.json').read_text(encoding='utf-8'))
    if sha(path) != info['temporal_sha256']:
        raise ValueError('Original temporal head digest mismatch')
    original = torch.load(path, map_location='cpu', weights_only=True)
    if original['base_sha256'] != trainer.WEIGHT_SHA:
        raise ValueError('Original temporal head base mismatch')

    def initialized_linear(width, classes):
        model = nn.Linear(width, classes)
        model.load_state_dict(original['state'], strict=True)
        captured['model'] = model
        return model

    def projected_loss(logits, labels):
        model = captured['model']
        params = (model.weight, model.bias)
        old = captured['old_mask'][:len(labels)]
        if len(old) != len(labels) or not bool(old.any()) or not bool((~old).any()):
            raise ValueError('Invalid old/new replay split')
        new_loss = F.cross_entropy(logits[~old], labels[~old])
        old_loss = F.cross_entropy(logits[old], labels[old])
        new_grad = torch.autograd.grad(new_loss, params, retain_graph=True)
        old_grad = torch.autograd.grad(old_loss, params, retain_graph=True)
        projected, changed = project_against_reference(new_grad, old_grad)
        dot = sum(float((g.double() * r.double()).sum()) for g, r in zip(projected, old_grad))
        captured['steps'] += 1
        captured['projected_steps'] += int(changed)
        captured['negative_dots_after'] += int(dot < -1e-7)
        # trainer.run adds 0.01*weight². Cancel it here so the SGD update uses
        # exactly the projected gradient, without an unprojected regularizer.
        return sum((p * g).sum() for p, g in zip(params, projected)) - 0.01 * model.weight.square().sum()

    def plain_sgd(params, lr, weight_decay):
        if lr != learning_rate:
            raise ValueError('Unexpected trainer learning rate')
        return sgd(params, lr=lr, weight_decay=0)

    with patch.object(trainer.AutoModelForCausalLM, 'from_pretrained', side_effect=previous.streamed_base), \
            patch.object(trainer, 'extract', side_effect=extract_with_partition), \
            patch.object(trainer, 'nn', SimpleNamespace(Linear=initialized_linear)), \
            patch.object(trainer, 'F', SimpleNamespace(cross_entropy=projected_loss)), \
            patch.object(trainer.torch.optim, 'AdamW', side_effect=plain_sgd):
        trainer.run()
    summary = {'steps': captured['steps'], 'projected_steps': captured['projected_steps'],
               'negative_dots_after': captured['negative_dots_after'], 'teacher_sha256': info['temporal_sha256'],
               'optimizer': 'sgd', 'learning_rate': learning_rate,
               'description': 'Full-batch new supervised gradient projected against old supervised gradient.',
               'limitations': ['First-order constraint only; finite step and threshold calibration can still regress.',
                               'Not a faithful online task-stream A-GEM reproduction.',
                               'Old/new examples are synthetic templates; no real individual changes.']}
    with (trainer.OUTPUT / 'projection.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
