"""Fit a three-way version selector while preserving the original read gate."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from scripts.audit_qwen_compatible_natural_memory_20261121 import fresh_version_cases, load_head
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_temporal_contrast_cases import ABSTAIN, temporal_rows
from scripts.train_spora_revision_contrast import build_data, fit
from spora.integrated_checkpoint import load_integrated_runtime
from spora.language_training import atomic_save
from spora.research_temporal_intent import (ordered_question, question_mean, temporal_character_features,
                                           temporal_logits, time_selection)


POLARITY_TRIPLES = (
    ('失效的方案先放一边，按现在有效的说明，{name}的{field}是哪项？',
     '现行方案先放一边，查被替下的那份，{name}的{field}是哪项？',
     '比那次更新更早的先放一边，查当时刚启用的{name}的{field}。'),
    ('按还在执行的规则给{name}确认{field}，旧版本可以略过。',
     '按被更新取代的规则给{name}确认{field}，现在的版本可以略过。',
     '按那天变更开始执行的规则给{name}确认{field}，之后的可以略过。'),
    ('{name}以前的{field}已撤销，今天按哪项办理？',
     '{name}今天的{field}已变化，撤销的是哪项？',
     '{name}那条{field}完成变更时，新启用的是哪项？'),
    ('{name}的{field}只查现用，草拟时期的不查。',
     '{name}的{field}只查旧稿，现用的内容不用列。',
     '{name}的{field}只查那次修改刚定下的，别浏览更初的材料。'),
)
SCOPE_TRIPLES = (
    ('{name}的{field}以现行条款为准，已作废的内容不要列。',
     '{name}的{field}以原条款为准，我想知道被换下的内容。',
     '{name}的{field}以第二份条款刚生效的内容为准，初稿跳过。'),
    ('拿现在生效的一份，只给我{name}的{field}。',
     '拿生效前的一份，只给我{name}原来的{field}。',
     '拿改动生效当时的一份，只给我{name}新换的{field}。'),
    ('对{name}来说，{field}早年有变化。哪项目前依然有效？',
     '对{name}来说，{field}早年有变化。哪项后来被停用了？',
     '对{name}来说，{field}早年有变化。那项改动开始实行时选了什么？'),
    ('不要回顾已经注销的内容，{name}有效的{field}是哪项？',
     '不要回顾现行的内容，{name}被注销的{field}最初是哪项？',
     '不要回顾没有修改的原稿，{name}的{field}那回正式执行时是哪项？'),
    ('{name}第一版提交过{field}。先放下那版，我要核对今天在用什么。',
     '{name}现有表格也有{field}。先放下那版，我要核对最早交了什么。',
     '{name}的{field}更新过多回，我只要首轮修订落实时替换成的内容。'),
    ('请告诉我{name}的{field}还保留哪一项，已经淘汰的省去。',
     '请告诉我{name}的{field}淘汰掉哪一项，手头在用的省去。',
     '请告诉我{name}的{field}定稿时加进来哪一项，过往的别复述。'),
)
SCOPE_HELD = (
    ('哪份已撤回不用说，{name}的{field}在现行册里怎么填？',
     '哪份现行不用说，{name}的{field}在撤回那册原本怎么填？',
     '现行和旧稿都跳过，{name}的{field}在那回切换落定时怎么填？'),
    ('先别翻过期单据。{name}还能按哪一项{field}执行？',
     '先别翻完工单据。{name}在动笔改之前用的哪一项{field}？',
     '先别翻开工时的单据。{name}首次重订开始执行哪一项{field}？'),
)
ORDER_HELD = (
    ('先前那份不再执行，请核对{name}还在执行的{field}。',
     '手头这份继续执行，但我想核对{name}被替换掉的{field}。',
     '手头这份仍在执行，但我只想核对{name}那回换上去的{field}。'),
    ('{name}登记过两套{field}，按手头还算数的给我对一下。',
     '{name}登记过两套{field}，按后来不算数的给我对一下。',
     '{name}登记过好几套{field}，按那次修订刚落地时采用的给我对一下。'),
    ('{name}那张{field}表，据说原件早就停了，今天用的写着什么？',
     '{name}那张{field}表，据说后来换了，原件停用以前写着什么？',
     '{name}那张{field}表，据说后来又换了，头一次切换落地当天写着什么？'),
)


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    torch.set_num_threads(2)
    digest = sha(args.checkpoint)
    head, source_sha = load_head(args.temporal, 'temporal.pt', 'temporal_sha256', sha(args.temporal_source))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(args.checkpoint, load_full_streamed(), MODEL)
    runtime.decoder.eval().requires_grad_(False)
    data = build_data()
    train = data['train'] + temporal_rows(('任亭', '卓悦', '赵枫', '姜文', '周梨', '吴璐'),
                                          ('邮寄方式', '主食', '订票选项'), POLARITY_TRIPLES, ())
    calibration = data['calibration']
    fresh = [(text, mode if mode is not None else ABSTAIN) for text, _, mode in fresh_version_cases()]
    challenge = data['new_holdout']
    groups = {'train': train, 'calibration': calibration, 'fresh': fresh, 'challenge': challenge}
    if args.expand_scope:
        train += temporal_rows(('梁灿', '关芸', '徐澜', '宁萍', '范乔', '席启'),
                               ('运费方案', '就餐时段', '取件柜', '包装材料', '备注写法', '开门时间'),
                               SCOPE_TRIPLES, ())
        groups['new_scope'] = temporal_rows(('严青', '高婉', '屈晴', '陆灼', '倪恒', '窦飞'),
                                            ('标牌文字', '办理柜台', '餐具'), SCOPE_HELD, ())
    if args.ordered:
        groups['unseen_order'] = temporal_rows(('穆清', '余芮', '佟柏', '聂津', '莫岩', '罗珈'),
                                                ('调度口令', '会议标签', '签发编号'), ORDER_HELD, ())
    questions = [text for rows in groups.values() for text, _ in rows]
    if len(set(questions)) != len(questions):
        raise ValueError('A version contrast question occurs in more than one split')
    slices = {}
    offset = 0
    for label, rows in groups.items():
        slices[label] = slice(offset, offset + len(rows))
        offset += len(rows)
    vectors, initial, ordered = [], [], []
    for start in range(0, len(questions), 64):
        texts = questions[start:start + 64]
        features, masks = extract(runtime.decoder.base, tokenizer, texts, batch_size=2,
                                  hidden_indices=tuple(head['hidden_indices']), last_logit_only=True)
        vectors.append(question_mean(features, masks))
        if args.ordered:
            ordered.append(ordered_question(features, masks))
        initial.append(temporal_logits(head, features, masks, texts=texts))
    vectors, initial = torch.cat(vectors), torch.cat(initial)
    targets = torch.tensor([mode for rows in groups.values() for _, mode in rows])
    baseline = time_selection(initial.softmax(-1), head['threshold'], factorized=True)
    version = F.linear(vectors, head['state']['version.weight'], head['state']['version.bias'])
    inputs = torch.cat(ordered) if args.ordered else vectors
    if args.lexical:
        inputs = torch.cat((vectors, temporal_character_features(questions, 1024)), dim=-1)
    training = slices['train']
    cal = slices['calibration']

    def fit_selector(inputs):
        weight = fit(inputs[training], version[training], targets[training])
        confidence, winner = (version + inputs @ weight.T).softmax(-1).max(-1)

        def select(threshold):
            return torch.where((baseline != ABSTAIN) & (confidence >= threshold), winner, baseline)

        threshold = max((.50, .55, .60, .65, .70, .75), key=lambda value: (
            int((select(value)[cal] == targets[cal]).sum()), value))
        return weight, threshold, select(threshold)

    weight, threshold, selected = fit_selector(inputs)
    predictions = {'baseline': baseline, 'candidate': selected}
    if args.ordered:
        _, mean_threshold, predictions['mean_comparison'] = fit_selector(vectors)
    if not torch.equal(selected == ABSTAIN, baseline == ABSTAIN) or sha(args.checkpoint) != digest:
        raise ValueError('The version-only adjustment changed abstentions or the source checkpoint')
    summary = {}
    for label in groups:
        if label == 'train':
            continue
        area = slices[label]
        rows = [{'question': text, 'expected': int(target),
                 **{mode: int(values[index]) for mode, values in predictions.items()}}
                for index, text, target in zip(range(area.start, area.stop), questions[area], targets[area], strict=True)]
        summary[label] = {mode: [sum(row['expected'] == index and row[mode] == index for row in rows)
                                 for index in range(4)] for mode in predictions}
        summary[label]['rows'] = rows
        print(label, summary[label]['baseline'], summary[label]['candidate'], flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    component = args.output / 'selector.pt'
    atomic_save({'weight': weight, 'threshold': threshold, 'char_width': 1024 if args.lexical else 0,
                 'feature_mode': 'ordered' if args.ordered else 'mean',
                 'checkpoint_sha256': digest,
                 'temporal_sha256': source_sha, 'format': 'spora-guarded-version-selector-v1'}, component)
    report = {'summary': summary, 'selector_sha256': sha(component), 'confidence_threshold': threshold,
              'train_count': len(train), 'seconds': time.perf_counter() - started,
              'sources': {'checkpoint_sha256': digest, 'temporal_sha256': source_sha}}
    if args.ordered:
        report['mean_comparison_threshold'] = mean_threshold
    (args.output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'temporal', 'temporal-source', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    features = parser.add_mutually_exclusive_group()
    features.add_argument('--lexical', action='store_true', help='Include character features in the learned residual')
    features.add_argument('--ordered', action='store_true', help='Compare position-aware clause features with a mean selector')
    parser.add_argument('--expand-scope', action='store_true', help='Fit varied timing and negation contrasts')
    main(parser.parse_args())
