"""Test whether masking a confirmed subject helps select a memory version."""

import argparse
import json
from pathlib import Path

from scripts.audit_named_memory_interleaved import NAMES as NEW_NAMES
from scripts.audit_qwen_compatible_natural_memory_20261121 import MODEL, NAMES, extract, load_head, sha, torch
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_temporal_contrast_cases import (
    CAL_NON_READ, CAL_TRIPLES, NON_READ, VERSION_TRIPLES, temporal_rows,
)
from scripts.train_qwen_temporal_token_gate_20261122 import INDICES, SEED, evaluate, fit, suites, threshold_for
from spora.integrated_checkpoint import load_integrated
from spora.language_training import atomic_save
from spora.research_temporal_intent import temporal_logits
from transformers import AutoTokenizer


EXTRA_TRAIN = (
    ('{name}上一次写错的{field}是哪一项？', 1),
    ('换了一页，给{name}新填进去的{field}是什么？', 2),
    ('刚替换到表里的{name}的{field}，报给我听。', 2),
    ('移交新版后，请读出{name}新采用的{field}。', 2),
    ('这次调整把{name}的{field}变成了哪一个？', 2),
    ('请按今天现行的名单复述{name}的{field}。', 0),
    ('给我重读一下{name}初次提报的{field}。', 1),
    ('给{name}换一页新的{field}，不需要读出任何值。', 3),
    ('谁刚替{name}换了一份{field}单？只问经手人。', 3),
    ('新版本的{name}的{field}以后会是什么？猜一下。', 3),
    ('把刚替换的单子递给{name}，先别说里面的{field}。', 3),
    ('「{name}新填的{field}是什么」这句改得礼貌一点。', 3),
)
EXTRA_CAL = (
    ('沿用现在执行那份，{name}的{field}填写了什么？', 0),
    ('追溯到本次替换之前，{name}记录的{field}是哪一项？', 1),
    ('这轮变动生效以后，{name}那条{field}是什么？', 2),
    ('{name}上交修改单的具体时刻是什么？别报{field}。', 3),
    ('用{name}的{field}变化构造一个类似游戏例子。', 3),
    ('把{name}的{field}改到下一版，先不要获取内容。', 3),
)
EXTRA_HELD = (
    ('以今天值班的人收到的最新版为准，{name}现用什么{field}？', 0),
    ('给{name}填写修改单以前，那栏{field}原文是什么？', 1),
    ('{name}重交的那张表通过审核以后，{field}一栏写了什么？', 2),
    ('新条目取代旧条目后，{name}的{field}究竟是哪项？', 2),
    ('查收新件，{name}刚登记成功的是哪款{field}？', 2),
    ('给{name}的{field}添个新版本即可，内容不用回显。', 3),
    ('{name}为什么要用新条目取代旧条目？我想问理由，别读{field}。', 3),
    ('你预计{name}下周重交表格时会把{field}改成什么？', 3),
    ('把「{name}刚登记成功的是哪款{field}」改成法语问句。', 3),
)


def instantiate(templates, names):
    fields = ('饮品', '收件地点', '展位', '餐点')
    return [(template.format(name=name, field=fields[index % len(fields)]), label)
            for index, name in enumerate(names) for template, label in templates]


def mask_subject(text, names):
    matches = [name for name in names if name in text]
    if len(matches) != 1 or text.count(matches[0]) != 1:
        return text
    return text.replace(matches[0], '这位登记人', 1)


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    source, baseline_sha = load_head(args.temporal, 'temporal.pt', 'temporal_sha256', sha(args.checkpoint))
    recorded = json.loads(args.interleaved.read_text(encoding='utf-8'))
    data = suites()
    groups = {key: value for key, value in data.items() if key not in ('train', 'calibration')}
    groups['interleaved'] = [(row['question'], row['expected_mode']) for row in recorded['rows']
                             if row['expected_key'] is not None]
    groups['train'] = list(dict.fromkeys(temporal_rows(('罗淇', '吕恬', '贺南'),
        ('饮品', '地址', '饮料'), VERSION_TRIPLES, NON_READ)))
    groups['calibration'] = temporal_rows(('郑斐', '江畔'), ('饮品', '尺码'), CAL_TRIPLES, CAL_NON_READ)
    extended_train = ('耿怡', '康彬', '舒文安', '傅同', '钟海', '钱沛')
    extended_calibration = ('范蕴', '池岚风', '潘茹')
    extra_held = ('左荷', '包诗晴', '赵恪', '唐菲')
    groups['extra_held'] = instantiate(EXTRA_HELD, extra_held)
    groups['extra_train'] = [*data['train'], *instantiate(EXTRA_TRAIN, extended_train)]
    groups['extra_calibration'] = [*data['calibration'], *instantiate(EXTRA_CAL, extended_calibration)]
    people = {'train': ('罗淇', '吕恬', '贺南'), 'calibration': ('郑斐', '江畔'), 'interleaved': NEW_NAMES,
              'extra_held': extra_held}
    flattened, normalized, slices = [], [], {}
    for group, rows in groups.items():
        start = len(flattened)
        flattened.extend(text for text, _ in rows)
        normalized.extend(text if group in ('extra_train', 'extra_calibration') else
                          mask_subject(text, people.get(group, NAMES)) for text, _ in rows)
        slices[group] = slice(start, len(flattened))
    unique = list(dict.fromkeys(normalized + flattened))
    indices = {text: i for i, text in enumerate(unique)}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    decoder, _, _ = load_integrated(args.checkpoint, load_full_streamed(), MODEL)
    features, masks = extract(decoder.base, tokenizer, unique, batch_size=2, hidden_indices=INDICES,
                              last_logit_only=True)
    del decoder
    clean = [indices[text] for text in normalized]
    original = [indices[text] for text in flattened]
    targets = torch.tensor([label for rows in groups.values() for _, label in rows])
    train = clean[slices['train']]
    candidate = dict(source, state=fit('factorized', features[train], masks[train], targets[slices['train']]),
                     confirmed_subject_mask='这位登记人')
    cal = clean[slices['calibration']]
    probability = temporal_logits(candidate, features[cal], masks[cal]).softmax(-1)
    candidate['threshold'] = threshold_for(probability, targets[slices['calibration']], factorized=True)
    extra = original[slices['extra_train']]
    expanded = dict(source, state=fit('factorized', features[extra], masks[extra], targets[slices['extra_train']]))
    cal = original[slices['extra_calibration']]
    probability = temporal_logits(expanded, features[cal], masks[cal]).softmax(-1)
    expanded['threshold'] = threshold_for(probability, targets[slices['extra_calibration']], factorized=True)
    summary = {}
    for mode, head, lookup in (('original', source, original), ('mask_only', source, clean),
                              ('trained_mask', candidate, clean), ('expanded', expanded, original)):
        summary[mode] = {}
        for group, rows in groups.items():
            if group in ('train', 'extra_train'):
                continue
            part = lookup[slices[group]]
            scores = temporal_logits(head, features[part], masks[part]).softmax(-1)
            summary[mode][group] = evaluate(rows, scores, head['threshold'], factorized=True)
        print(mode, {group: values['by_label'] for group, values in summary[mode].items()}, flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    weights = args.output / 'temporal.pt'
    atomic_save(expanded, weights)
    result = {'temporal_sha256': sha(weights), 'baseline_temporal_sha256': baseline_sha,
              'interleaved_sha256': sha(args.interleaved), 'results': summary}
    (args.output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                                             encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'temporal', 'interleaved', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
