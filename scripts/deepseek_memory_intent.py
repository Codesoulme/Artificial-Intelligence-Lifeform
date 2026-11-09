"""Train a finite current-memory intent gate from frozen DeepSeek question states."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_learned_address import EXPANDED_PROMPTS, HELD_PROMPTS, NOVEL_PROMPTS
from scripts.deepseek_local_baseline import MODEL, REVISION, WEIGHT_SHA
from scripts.deepseek_pooled_address import extract
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load
from spora.language_training import frozen_digest

TRAIN_NEGATIVE = {
    'history': ('对象{key}以前的值是什么？', '编号{key}之前的值是多少？', '对象{key}过去记录是多少？',
                '对象{key}原来的数值是几？'),
    'write': ('请修改对象{key}的数值。', '把对象{key}当前的值设为2。', '给对象{key}写入新的数值。'),
    'multiple': ('对象{key}和对象{other}的数值之和是多少？', '比较对象{key}和对象{other}的值。'),
}
HELD_NEGATIVE = {
    'history': ('对象{key}前天的值是多少？', '对象{key}更早的数值是什么？'),
    'write': ('对象{key}现在的值改为5。', '替对象{key}更新数值。'),
    'multiple': ('计算对象{key}和对象{other}的和。', '对象{key}与对象{other}的数值相同吗？'),
}
TRAIN_UNRELATED = ('你好，请介绍一下自己。', '天气怎么样？', '明天要做什么？', '请说个笑话。',
                   '如何学习中文？', '我有点累。', '一加一等于几？', '解释什么是记忆。')
HELD_UNRELATED = ('今天的心情如何？', '给我推荐一首音乐。', '社会是什么？', '三乘二等于多少？')
NEXT_CURRENT = ('请告诉我编号{key}目前是多少？', '请查阅对象{key}此刻的数值。',
                '对象{key}眼下记录的数字是几？')
NEXT_NEGATIVE = {
    'history': ('对象{key}昨天是什么值？', '查询对象{key}上一条记录。'),
    'write': ('将对象{key}更新成8。', '给对象{key}设定为3。'),
    'multiple': ('对象{key}加对象{other}等于多少？', '请合计对象{key}与对象{other}的值。'),
}
NEXT_UNRELATED = ('今天过得怎么样？', '你会说中文吗？', '帮我计算三加五。', '如何改善睡眠？')
TRAIN_MALFORMED = ('对象的值是多少？', '对象现在的值是什么？', '编号当前是多少？',
                   '请查询对象的当前值。', '查询编号对应的数值。', '对象目前记录的值是？')
HELD_MALFORMED = ('请告诉我对象的数值。', '编号对应的值是多少？', '对象目前记录多少？',
                  '查询当前数值。', '对象的当前数字是几？', '请读取编号的值。')
HELD_UNSUPPORTED = ('对象二十六的值是多少？', '对象-1的值是多少？', '对象8.5的值是多少？',
                    '对象007的值是多少？', '编号100的值是什么？',
                    '请把对象8目前的值告诉我，并把对象9修改为3。')
CONTRAST_CURRENT = ('当前对象{key}存储的数字是几？', '对象{key}当前的数值是多少？')
CONTRAST_NEGATIVE = {
    'derived': ('对象{key}的值是奇数吗？', '对象{key}的值大于零吗？', '判断编号{key}的数值是否为偶数。'),
    'source': ('对象{key}的记录来自哪里？', '对象{key}的值是谁告诉你的？'),
    'negated': ('不要读取对象{key}的值。', '我不想知道对象{key}的值。'),
}
CONTRAST_HELD = {
    'derived': ('对象{key}存的数字是否为偶数？', '对象{key}的值能被二整除吗？'),
    'source': ('对象{key}的值最初由谁提供？', '编号{key}的信息来源是什么？'),
    'negated': ('别告诉我对象{key}的值。', '我不需要查询编号{key}。'),
}
BOUNDARY = (('history', '对象26昨天的值是多少？'), ('history', '对象8上一次的值是多少？'),
            ('write', '请把对象26的值改成3。'), ('multiple', '对象8和对象9的值之和是多少？'),
            ('out_of_range', '对象64的值是多少？'), ('out_of_range', '对象99的值是多少？'),
            ('unrelated', '今天天气怎么样？'), ('malformed', '对象的值是多少？'))


class IntentGate(nn.Module):
    def __init__(self, hidden_width):
        super().__init__()
        self.hidden_width = hidden_width
        self.score = nn.ModuleList(nn.Linear(hidden_width, 1, bias=False) for _ in range(2))
        self.classifier = nn.Linear(2 * hidden_width, 1)
        for layer in self.score:
            nn.init.zeros_(layer.weight)

    def forward(self, features, mask):
        if (features.ndim != 4 or features.shape[1] != 2 or features.shape[-1] != self.hidden_width
                or mask.shape != (features.shape[0], features.shape[2]) or mask.dtype != torch.bool
                or not bool(mask.any(-1).all())):
            raise ValueError('Expected nonempty question features and mask')
        features = features.float()
        pooled = []
        for site in range(2):
            x = features[:, site]
            weights = self.score[site](x).squeeze(-1).masked_fill(~mask, float('-inf')).softmax(-1)
            pooled.append(F.layer_norm((x * weights.unsqueeze(-1)).sum(1), (self.hidden_width,)))
        return self.classifier(torch.cat(pooled, dim=-1)).squeeze(-1)


def build_cases(held=False, augmented=False, malformed=False, contrastive=False):
    if malformed and not augmented:
        raise ValueError('Malformed cases require augmented dataset')
    if contrastive and not malformed:
        raise ValueError('Contrastive cases require malformed examples')
    if augmented:
        templates = (NEXT_CURRENT if held else EXPANDED_PROMPTS + HELD_PROMPTS + NOVEL_PROMPTS +
                     (CONTRAST_CURRENT if contrastive else ()))
        negatives = NEXT_NEGATIVE if held else {
            category: TRAIN_NEGATIVE[category] + HELD_NEGATIVE[category] for category in TRAIN_NEGATIVE}
    else:
        templates = (HELD_PROMPTS + NOVEL_PROMPTS) if held else EXPANDED_PROMPTS
        negatives = HELD_NEGATIVE if held else TRAIN_NEGATIVE
    keys = range(64)
    rows = [{'category': 'current', 'question': template.format(key=key), 'label': 1}
            for key in keys for template in templates]
    for category, choices in negatives.items():
        rows.extend({'category': category, 'question': template.format(key=key, other=(key + 1) % 64), 'label': 0}
                    for key in keys for template in choices)
    if contrastive:
        contrast = CONTRAST_HELD if held else CONTRAST_NEGATIVE
        for category, choices in contrast.items():
            rows.extend({'category': category, 'question': template.format(key=key), 'label': 0}
                        for key in keys for template in choices)
    out_templates = (templates if augmented else (('编号{key}当前的值是多少？',) if held else
                     ('对象{key}现在的值是多少？', '对象{key}的当前值是什么？')))
    rows.extend({'category': 'out_of_range', 'question': template.format(key=key), 'label': 0}
                for key in range(64, 100) for template in out_templates)
    rows.extend({'category': 'unrelated', 'question': question, 'label': 0}
                for question in (NEXT_UNRELATED if augmented and held else
                                 TRAIN_UNRELATED + HELD_UNRELATED if augmented else
                                 HELD_UNRELATED if held else TRAIN_UNRELATED))
    if malformed:
        rows.extend({'category': 'malformed', 'question': question, 'label': 0}
                    for question in (HELD_MALFORMED if held else TRAIN_MALFORMED))
        if held:
            rows.extend({'category': 'unsupported', 'question': question, 'label': 0}
                        for question in HELD_UNSUPPORTED)
    return rows


@torch.no_grad()
def evaluate(gate, features, masks, rows, batch_size=64):
    scores = []
    for offset in range(0, len(rows), batch_size):
        logits = gate(features[offset:offset + batch_size].to('cuda'), masks[offset:offset + batch_size].to('cuda'))
        if not bool(torch.isfinite(logits).all()):
            raise ValueError('Nonfinite gate outputs')
        scores.extend(logits.sigmoid().cpu().tolist())
    output = [dict(row, probability=float(score), accepted=score >= 0.5) for row, score in zip(rows, scores)]
    categories = {}
    for category in sorted({row['category'] for row in rows}):
        group = [row for row in output if row['category'] == category]
        categories[category] = {'examples': len(group), 'accepted': sum(row['accepted'] for row in group),
                                'correct': sum(row['accepted'] == bool(row['label']) for row in group)}
    return {'examples': len(rows), 'correct': sum(row['accepted'] == bool(row['label']) for row in output),
            'categories': categories, 'rows': output}


def run(output, steps, augmented=False, malformed=False, contrastive=False, initial=None):
    output = Path(output).resolve()
    runs = (ROOT / 'runs').resolve()
    if not output.is_relative_to(runs) or output == runs or output.exists() or not 1 <= steps <= 5000:
        raise ValueError('Expected a new output under runs and valid steps')
    initial = Path(initial).resolve() if initial is not None else None
    if initial is not None and not initial.is_relative_to(runs):
        raise ValueError('Initial gate must be under runs')
    initial_sha = sha(initial) if initial is not None else None
    def verify():
        return (sha(MODEL / 'model.safetensors') == WEIGHT_SHA and sha(INDIVIDUAL) == INDIVIDUAL_SHA
                and (initial is None or sha(initial) == initial_sha))
    if not verify():
        raise ValueError('Protected source mismatch')
    torch.manual_seed(4610)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(organism, original)
    training = build_cases(augmented=augmented, malformed=malformed, contrastive=contrastive)
    held = build_cases(True, augmented=augmented, malformed=malformed, contrastive=contrastive)
    if set(row['question'] for row in training) & set(row['question'] for row in held):
        raise ValueError('Train and held questions overlap')
    boundary = [{'category': category, 'question': question, 'label': 0} for category, question in BOUNDARY]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    features, masks = extract(model, tokenizer, [r['question'] for r in training])
    held_features, held_masks = extract(model, tokenizer, [r['question'] for r in held])
    boundary_features, boundary_masks = extract(model, tokenizer, [r['question'] for r in boundary])
    labels = torch.tensor([row['label'] for row in training], dtype=torch.float32)
    positive = torch.nonzero(labels == 1).flatten()
    negative = torch.nonzero(labels == 0).flatten()
    generator = torch.Generator().manual_seed(4610)
    gate = IntentGate(model.config.hidden_size).to('cuda')
    if initial is not None:
        checkpoint = torch.load(initial, map_location='cpu', weights_only=True)
        if (checkpoint.get('format') != 'deepseek-spora-current-intent-gate-v1'
                or checkpoint.get('weight_sha256') != WEIGHT_SHA or checkpoint.get('individual_sha256') != INDIVIDUAL_SHA
                or checkpoint.get('layers') != [5, 17] or checkpoint.get('threshold') != 0.5):
            raise ValueError('Initial gate provenance mismatch')
        gate.load_state_dict(checkpoint['state'], strict=True)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=0.001, weight_decay=0.)
    log = []
    for step in range(steps):
        indices = torch.cat((positive[torch.randint(len(positive), (32,), generator=generator)],
                             negative[torch.randint(len(negative), (32,), generator=generator)]))
        logits = gate(features[indices].to('cuda'), masks[indices].to('cuda'))
        loss = F.binary_cross_entropy_with_logits(logits, labels[indices].to('cuda'))
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite gate loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(gate.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 250 == 0 or step + 1 == steps:
            entry = {'step': step + 1, 'loss': float(loss.detach()), 'gradient_norm': float(norm)}
            log.append(entry)
            print(json.dumps(entry), flush=True)
    gate.eval()
    if any(p.grad is not None for p in model.parameters()) or frozen_digest(organism, original) != before or not verify():
        raise ValueError('Protected inputs changed')
    output.mkdir(parents=False, exist_ok=False)
    snapshot = output / 'gate.pt'
    torch.save({'format': 'deepseek-spora-current-intent-gate-v1',
                'state': {k: v.detach().cpu() for k, v in gate.state_dict().items()},
                'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA,
                'layers': [5, 17], 'threshold': 0.5}, snapshot)
    print('gate checkpoint saved; evaluation starting', flush=True)
    results = {'train': evaluate(gate, features, masks, training),
               'held': evaluate(gate, held_features, held_masks, held),
               'boundary_development': evaluate(gate, boundary_features, boundary_masks, boundary)}
    if frozen_digest(organism, original) != before or not verify():
        raise ValueError('Protected input changed during evaluation')
    report = {'status': 'finite_intent_gate_development_only', 'revision': REVISION, 'weight_sha256': WEIGHT_SHA,
              'individual_sha256': INDIVIDUAL_SHA, 'gate_sha256': sha(snapshot), 'steps': steps,
              'threshold': 0.5, 'augmented': augmented, 'malformed_training': malformed,
              'contrastive_training': contrastive, 'initial_gate_sha256': initial_sha,
              'optimizer_restarted': initial is not None,
              'results': results, 'log': log,
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Only detects a narrow current-value numbered-query class.',
                              'All examples are synthetic; held templates share known entities.',
                              'Boundary cases were inspected during development and are not final holdout.',
                              'Unknown in-range memory values require a separate state-aware check.',
                              'Gate has not been integrated into a user-facing runtime.',
                              'Fixed threshold was selected before evaluation and is not calibrated.']}
    with (output / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({name: {k: v for k, v in result.items() if k != 'rows'}
                      for name, result in results.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--augmented', action='store_true')
    parser.add_argument('--malformed', action='store_true')
    parser.add_argument('--contrastive', action='store_true')
    parser.add_argument('--initial', type=Path)
    args = parser.parse_args()
    run(args.output, args.steps, args.augmented, args.malformed, args.contrastive, args.initial)
