"""Fresh adversarial wording audit of the frozen relation selection probe."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch import nn
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_relation_selection_probe import LABELS

SOURCE = ROOT / 'runs/deepseek-spora-relation-selection-probe-20261116/head.pt'
OUT = ROOT / 'runs/deepseek-spora-relation-selection-stress-20261116.json'
CASES = (
    ('drink', '按记载，阿宁常喝什么？'),
    ('drink', '告诉我阿宁的首选饮料名称。'),
    ('color', '阿宁中意的色调是什么？'),
    ('color', '根据记忆，阿宁更喜欢哪种颜色？'),
    ('activity', '阿宁的主要业余爱好是什么？'),
    ('activity', '记忆里阿宁常做的活动是哪一项？'),
    ('reject', '不要告诉我阿宁喜欢的颜色。'),
    ('reject', '阿宁喜欢的饮料温度是热还是凉？'),
    ('reject', '阿宁喜欢的颜色是哪家店买的？'),
    ('reject', '阿宁喜欢什么饮料和颜色？'),
    ('reject', '帮阿宁挑一种饮品，不需要查记忆。'),
    ('reject', '请把阿宁喜欢的活动更新为游泳。'),
    ('reject', '假设阿宁喜欢咖啡，它会有什么味道？'),
    ('reject', '忘记阿宁的色彩喜好。'),
    ('reject', '阿宁喜欢什么电影？'),
    ('reject', '今天星期几？'),
    ('reject', '不查记录，猜猜阿宁通常喝什么。'),
    ('reject', '阿宁喜欢的活动一般在什么时间进行？'),
)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    source_sha = p.sha(SOURCE)
    data = torch.load(SOURCE, map_location='cpu', weights_only=True)
    if (base_sha != p.WEIGHT_SHA or data['base_sha256'] != base_sha
            or data['checkpoint_sha256'] != checkpoint_sha or tuple(data['labels']) != LABELS):
        raise ValueError('Probe lineage mismatch')
    torch.set_num_threads(2)
    head = nn.Linear(data['state']['weight'].shape[1], len(LABELS))
    head.load_state_dict(data['state'], strict=True)
    with tempfile.TemporaryDirectory(prefix='spora-relation-stress-') as temp:
        p.runtime.WORKING_CHECKPOINT = Path(temp) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        before = p.frozen_digest(chat.organism, chat.state)
        features, masks = extract(chat.base, chat.tokenizer, [q for _, q in CASES], batch_size=1)
        with torch.inference_mode():
            logits = head(pool(features, masks).float())
        rows = []
        for (expected, question), scores in zip(CASES, logits):
            predicted = LABELS[int(scores.argmax())]
            rows.append({'expected': expected, 'question': question, 'predicted': predicted,
                         'correct': predicted == expected, 'logits': scores.tolist()})
        if p.frozen_digest(chat.organism, chat.state) != before:
            raise ValueError('Individual changed')
    if any(p.sha(path) != digest for path, digest in ((SOURCE, source_sha),
            (p.MODEL / 'model.safetensors', base_sha), (p.runtime.CHECKPOINT, checkpoint_sha))):
        raise ValueError('Source changed')
    summary = {label: {'correct': sum(r['correct'] for r in rows if r['expected'] == label),
                       'total': sum(r['expected'] == label for r in rows)} for label in LABELS}
    report = {'rows': rows, 'summary': summary, 'probe_sha256': source_sha,
              'limitations': ['Authored exploratory cases, not population statistics.',
                              'No identity lookup, memory read, or runtime refusal is evaluated.',
                              'Unknown entities cannot be detected from relation classification alone.']}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
