"""Compare the inheritance gate with a dedicated single-turn repeat-intent head."""

import argparse
from pathlib import Path

from deepseek_continuous_memory_chat import load_chat
from deepseek_memory_dialogue_context import features_for
from deepseek_memory_turn_priority import gate_probabilities, train_inheritance
from deepseek_output_aligned_memory import ROOT, sha, torch, write
from spora.research_continuous_observation import ELLIPTICAL_QUERY


TRAIN = {
    'follow': (
        '可以把刚刚的答案复述一下吗？', '上面那个结果我想再听一次。', '请重复一下已查到的内容。',
        '资料里的那个值是什么来着？', '漏听了，你把答案再念一下。', '同一项，再告诉我具体值。',
        '刚才报的结果请再发一下。', '你刚刚说了什么结果？', '还是原来那项，读一下吧。',
        '能把查到的那个细节再讲一遍吗？', '查完的话报一下答案。', '再念一遍你找到的记录值。',
    ),
    'ack': ('了解了，多谢。', '这项我记下了。', '嗯，我已经听清了。', '那个值我收到了，谢谢。'),
    'cancel': ('不用念了。', '这一条先略过。', '算了，先别说。', '暂停一下，先不查了。'),
    'switch': ('改成查另一个人的。', '再看看其他人的资料。', '那另一项怎么写的？', '他从哪天开始的？'),
    'history': ('我想查改之前的那条。', '旧版本里怎么记的？', '那最早一笔写了什么？', '回到上一次更正之前查。'),
    'other': ('怎么把记录导出来？', '能给我一个例子吗？', '你还能查询哪些字段？', '帮我写一句开场白。'),
}
CALIBRATION = {
    'follow': ('刚刚的查询值，麻烦重复。', '还是那一条，告诉我答案。', '把上一项读给我听。', '查出来的结果，再报一次。'),
    'ack': ('上一条的内容我听到了。', '嗯，那个答案我清楚了。'),
    'cancel': ('先不要报结果。', '那一项不必再说。'),
    'switch': ('换个对象重新查一下。', '他到访是在几月？'),
    'history': ('另外说下更新之前的值。', '再看看倒数第二次的记录。'),
    'other': ('这个结果怎么导出？', '这些信息能用来做什么？'),
}
PROBE = {
    'follow': (
        '能再说一遍吗？', '你刚才查到的是什么来着？', '刚才那一项麻烦再说一下。', '还是上面那条，结果是什么？',
        '我没听清，答案是什么？', '麻烦只报一下记录里那个值。', '所以最后查出来的是啥？', '上一条的结果再发我一次。',
    ),
    'ack': ('这个我知道了。',),
    'cancel': ('不查了，谢谢。',),
    'switch': ('那他什么时候去的？', '换个人看看。'),
    'history': ('前一版记的是什么？',),
    'other': ('给我推荐一杯饮料。', '麻烦说一下怎么整理这些资料。', '还可以查哪些内容？'),
}
HOLDOUT = {
    'follow': ('我走神了，方才说的是哪项答案？', '麻烦把方才的查询结论再讲下。', '刚才那个答案可以发出来吗？',
               '同一条资料，具体结果请再给一次。', '你上句从资料里读出的内容是什么？', '我想听听那条的具体答案。'),
    'ack': ('刚才那个答案我已经收到。', '同一条资料的结果我明白了。'),
    'cancel': ('刚刚的问题不用回答了。', '不用告诉我那条的具体内容。'),
    'switch': ('同一个人，改查他去哪了。', '刚才那个答案说的是谁？', '现在查另外一个人。'),
    'history': ('这条后来被更正过吗？', '现在说说更正前查到的内容。'),
    'other': ('同一条资料可以分享给谁？', '为什么这里存了这些记录？', '如果没找到应该怎么办？'),
}
DEFAULT_DATABASE = ROOT / 'runs/deepseek-continuous-memory-capacity-batched/timeline-smoke.sqlite3'


def rows_for(group):
    return [dict(key=question, kind=kind, question=question) for kind, prompts in group.items() for question in prompts]


def evaluate(rows, probabilities, cutoff):
    accepted = [row for row, score in zip(rows, probabilities, strict=True) if score >= cutoff]
    correct = sum(row['kind'] == 'follow' for row in accepted)
    return dict(correct_reads=correct, false_reads=len(accepted) - correct)


def choose_cutoff(rows, probabilities):
    def rank(threshold):
        results = evaluate(rows, probabilities, threshold)
        return results['false_reads'], -results['correct_reads'], threshold

    return min((.5, .6, .7, .8, .9, .95, 1.), key=rank)


def combine_hidden(tail, tokens):
    mean = torch.nn.functional.normalize(tokens.mean(dim=0), dim=0)
    return torch.cat((tail, mean)) / 2**.5


def main(database, output):
    starting_digest = sha(database)
    groups = {name: rows_for(group) for name, group in (
        ('train', TRAIN), ('calibration', CALIBRATION), ('probe', PROBE), ('holdout', HOLDOUT),
    )}
    prompts = [row['question'] for rows in groups.values() for row in rows]
    if len(set(prompts)) != len(prompts):
        raise ValueError('Intent probe prompts must be disjoint')
    chat = load_chat(database, use_inherited_router=True, use_retrained_gate=True)
    features = {question: features_for(chat.decoder, chat.tokenizer, dict(question=question), False) for question in prompts}
    pooled = {question: (combine_hidden(feature[0], feature[1]), *feature[1:]) for question, feature in features.items()}
    calibration = groups['calibration']
    candidates = {}
    for name, vectors in (('tail', features), ('tail_and_mean', pooled)):
        head = train_inheritance(groups['train'], vectors)
        cutoff = choose_cutoff(calibration, gate_probabilities(head, calibration, vectors))
        candidates[name] = head, cutoff, vectors
    results = {}
    for name in ('calibration', 'probe', 'holdout'):
        rows = groups[name]
        old = gate_probabilities(chat.inheritance, rows, features)
        guarded = [score if ELLIPTICAL_QUERY.fullmatch(row['question']) else 0. for row, score in zip(rows, old, strict=True)]
        summary = dict(positive=sum(row['kind'] == 'follow' for row in rows),
                       negative=sum(row['kind'] != 'follow' for row in rows),
                       existing=evaluate(rows, old, chat.gate_cutoff),
                       existing_with_rule=evaluate(rows, guarded, chat.gate_cutoff),
                       candidates={})
        details = [dict(row, existing_score=score) for row, score in zip(rows, old, strict=True)]
        for candidate, (head, cutoff, vectors) in candidates.items():
            scores = gate_probabilities(head, rows, vectors)
            summary['candidates'][candidate] = evaluate(rows, scores, cutoff)
            for detail, score in zip(details, scores, strict=True):
                detail[candidate + '_score'] = score
        results[name] = dict(summary=summary, rows=details)
        print(name, summary, flush=True)
    if sha(database) != starting_digest:
        raise ValueError('Source database changed during the probe')
    write(output, dict(existing_cutoff=chat.gate_cutoff, cutoffs={name: item[1] for name, item in candidates.items()},
                       database_sha256=starting_digest,
                       train=groups['train'], results=results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=DEFAULT_DATABASE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    main(args.database, args.output)
