"""Compare the inheritance gate with a dedicated single-turn repeat-intent head."""

import argparse
from pathlib import Path

from deepseek_continuous_memory_chat import load_chat
from deepseek_memory_dialogue_context import features_for
from deepseek_memory_turn_priority import gate_probabilities, train_inheritance
from deepseek_output_aligned_memory import ROOT, sha, write
from spora.awake_recall_intent import FORMAT, intent_features
from spora.language_training import atomic_save
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
MATCHED_TRAIN = (
    ('刚才的回答，能再说一遍吗？', '刚才的回答，说的是哪个人？', 'switch'),
    ('还是这条，能重新报一下刚才的值吗？', '还是这条，能报一下它以前的值吗？', 'history'),
    ('告诉我刚刚查到的结果。', '别告诉我刚刚查到的结果。', 'cancel'),
    ('那个答案麻烦再发给我一下。', '那个答案已经发给我了。', 'ack'),
    ('再把刚才查出来的内容说一下。', '再说一下还能查哪些内容。', 'other'),
    ('刚刚的记录结论请重讲一遍。', '刚刚的记录结论对应的是谁？', 'switch'),
    ('能再讲一下方才那条的结果吗？', '能讲一下方才那条还没改时的结果吗？', 'history'),
    ('我还想再听一下答案。', '我不想再听一下答案了。', 'cancel'),
    ('你刚刚回答的还可以再说一次吗？', '你刚刚回答的我已经清楚了。', 'ack'),
    ('给我看刚才查出的值。', '告诉我怎么找到刚才那个页面。', 'other'),
    ('方才的结果再报一下。', '方才查完了，现在换另一个人。', 'switch'),
    ('请说说上一句给出的结论。', '请说说那条被更改之前的结论。', 'history'),
    ('我需要知道刚才查询返回了什么。', '我需要知道刚才查的是哪位。', 'switch'),
    ('再用一句话复述上面的结果。', '再用一句话介绍一下查询功能。', 'other'),
)
MATCHED_HOLDOUT = (
    ('你刚才报的值，可否重说一次？', '你刚才报的值，在更正之前是什么？', 'history'),
    ('上面那项的答案是什么，麻烦重报。', '上面那项的答案是关于谁的？', 'switch'),
    ('关于刚才查的那条，我还想听下结果。', '关于刚才查的那条，我已经清楚结果了。', 'ack'),
    ('方便再读一次你前面查出的答案吗？', '你前面查出的答案暂时别读了。', 'cancel'),
    ('我说的是前一条问答，麻烦把结果再发一遍。', '前一条问答以外，还能查些什么？', 'other'),
    ('刚才你回答的结果是啥，能不能再说一次？', '刚才你回答的结果是哪一天登记的？', 'switch'),
    ('请复述刚出来的查询结论。', '请讲一下这项更正前的查询结论。', 'history'),
    ('前面那个答案你再讲一下吧。', '前面那个答案我听清了，改问别的。', 'switch'),
)
DEFAULT_DATABASE = ROOT / 'runs/deepseek-continuous-memory-capacity-batched/timeline-smoke.sqlite3'


def rows_for(group):
    return [dict(key=question, kind=kind, question=question) for kind, prompts in group.items() for question in prompts]


def matched_rows(pairs):
    return [dict(key=question, kind=kind, question=question, pair=index)
            for index, (positive, negative, category) in enumerate(pairs)
            for question, kind in ((positive, 'follow'), (negative, category))]


def evaluate(rows, probabilities, cutoff):
    accepted = [row for row, score in zip(rows, probabilities, strict=True) if score >= cutoff]
    correct = sum(row['kind'] == 'follow' for row in accepted)
    return dict(correct_reads=correct, false_reads=len(accepted) - correct)


def choose_cutoff(rows, probabilities):
    def rank(threshold):
        results = evaluate(rows, probabilities, threshold)
        return results['false_reads'], -results['correct_reads'], threshold

    return min((.5, .6, .7, .8, .9, .95, 1.), key=rank)


def main(database, output, head_path=None):
    starting_digest = sha(database)
    groups = {name: rows_for(group) for name, group in (
        ('train', TRAIN), ('calibration', CALIBRATION), ('probe', PROBE), ('holdout', HOLDOUT),
    )}
    groups['matched_train'] = matched_rows(MATCHED_TRAIN)
    groups['matched_holdout'] = matched_rows(MATCHED_HOLDOUT)
    prompts = [row['question'] for rows in groups.values() for row in rows]
    if len(set(prompts)) != len(prompts):
        raise ValueError('Intent probe prompts must be disjoint')
    chat = load_chat(database, use_inherited_router=True, use_retrained_gate=True)
    features = {question: features_for(chat.decoder, chat.tokenizer, dict(question=question), False) for question in prompts}
    pooled = {question: (intent_features(*feature[:2]), *feature[1:]) for question, feature in features.items()}
    calibration = groups['calibration']
    candidates = {}
    for name, vectors, training in (
        ('tail', features, groups['train']),
        ('tail_and_mean', pooled, groups['train']),
        ('matched_tail_and_mean', pooled, groups['train'] + groups['matched_train']),
    ):
        head = train_inheritance(training, vectors)
        cutoff = choose_cutoff(calibration, gate_probabilities(head, calibration, vectors))
        candidates[name] = head, cutoff, vectors
    results = {}
    for name in ('calibration', 'probe', 'holdout', 'matched_holdout'):
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
                       train=groups['train'], matched_train=groups['matched_train'], results=results))
    if head_path is not None:
        head, cutoff, _ = candidates['matched_tail_and_mean']
        atomic_save(dict(format=FORMAT, fingerprint=chat.recall_fingerprint, cutoff=cutoff,
                         state=head.state_dict()), head_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=DEFAULT_DATABASE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--head', type=Path, help='Export the matched-example head for opt-in chat')
    args = parser.parse_args()
    for path in (args.output, args.head):
        if path is not None and path.exists():
            raise FileExistsError(path)
    if args.head is not None and args.head.resolve() == args.output.resolve():
        parser.error('--output and --head require separate paths')
    main(args.database, args.output, args.head)
