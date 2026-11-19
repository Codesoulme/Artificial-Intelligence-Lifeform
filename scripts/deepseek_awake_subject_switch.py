"""Test carrying a verified person into a new field question without transcript text."""

import argparse
import re
from pathlib import Path

from deepseek_awake_recall_intent import DEFAULT_DATABASE, choose_cutoff
from deepseek_continuous_memory_chat import load_chat
from deepseek_memory_attribute_span import route_action
from deepseek_memory_dialogue_context import features_for, route
from deepseek_memory_turn_priority import gate_probabilities, train_inheritance
from deepseek_output_aligned_memory import sha, torch, write
from spora.awake_recall_intent import SUBJECT_SCOPE_CONFLICT, SameSubjectIntent, intent_features
from spora.language_training import atomic_save
from spora.research_continuous_observation import query_subject


PROBE = (
    ('work', '那他的职业是什么？'), ('work', '同一个人，改查他当时做什么工作。'),
    ('place', '那他去过哪里？'), ('place', '他那次去了什么地方？'),
    ('drink', '她最喜欢喝什么？'), ('drink', '同一个人，改问她爱喝什么。'),
    ('time', '那他什么时候去的？'), ('time', '他是哪一天到的？'),
    ('switch', '那另一个人是做什么工作的？'), ('switch', '换个人，他喜欢喝什么？'),
    ('history', '在更正以前，他最喜欢喝什么？'),
    ('ordinary', '他平常应该怎么准备面试？'), ('ordinary', '顺便推荐一种好喝的饮料。'),
)
TRAIN = (
    ('work', '同一位，改查一下他的岗位。'), ('switch', '另外那位，查一下他的岗位。'),
    ('work', '接着查她从事哪份工作。'), ('ordinary', '接着聊她该找哪份工作。'),
    ('work', '他以什么营生为业？'), ('history', '更改之前记的是什么职业？'),
    ('work', '同一个对象，当时的工作叫什么？'), ('time', '同一个对象，入职是什么时间？'),
    ('work', '这个人的职业还有记录吗？'), ('ordinary', '这个人的职业有什么建议吗？'),
    ('place', '再查查他那一趟到了哪儿。'), ('time', '再查查他那一趟几点到的。'),
    ('place', '她那回的目的地是哪？'), ('history', '她那回原先登记的目的地是哪？'),
    ('place', '同一人，再说下到访地点。'), ('switch', '另一个人，再说下到访地点。'),
    ('place', '那位当事人那次到的地方是哪里？'), ('ordinary', '那位当事人适合去哪里度假？'),
    ('place', '那他去了哪个地方？'), ('time', '那他是哪一天去的？'),
    ('drink', '那他最爱的是哪款饮品？'), ('switch', '新的对象最爱哪款饮品？'),
    ('drink', '还是这个人，再查喜欢喝什么。'), ('ordinary', '还是这个人，推荐给他一杯好喝的。'),
    ('drink', '想知道她的饮料偏好。'), ('history', '想知道她更正之前的饮料偏好。'),
    ('drink', '同一个对象最常偏爱什么饮品？'), ('time', '同一个对象，哪天改了饮品偏好？'),
    ('drink', '改查他爱喝哪一种。'), ('repeat', '不改了，再把原来查到的说一遍。'),
)
CALIBRATION = (
    ('work', '再问一下这位靠什么谋生。'), ('switch', '再问另一位靠什么谋生。'),
    ('work', '同一个人，他的岗位叫啥？'), ('history', '同一个人，改动前的岗位叫啥？'),
    ('place', '继续说她那趟实际去了何处。'), ('time', '继续说她那趟是几月去的。'),
    ('place', '再查查此人到过哪个地点。'), ('ordinary', '再想想此人适合在哪儿度假。'),
    ('drink', '也查一下那位偏爱的饮料。'), ('repeat', '也把刚才查到的回答再说下。'),
    ('drink', '这个人比较爱喝哪种饮料？'), ('ordinary', '这个人可以尝试哪种饮料？'),
)
HOLDOUT = (
    ('work', '还是刚提到那个人，查他的职业。'), ('switch', '改看另一位，查他的职业。'),
    ('work', '她那时候具体是干哪行的？'), ('history', '她的职业在更新以前写成什么？'),
    ('work', '对了，那他从事什么工作？'), ('ordinary', '对了，那他该怎么找工作？'),
    ('place', '他那回到访的具体是哪个地方？'), ('time', '他那回是何时到访的？'),
    ('place', '还是这位，再查他那趟的目的地。'), ('history', '还是这位，查修正前记载的目的地。'),
    ('place', '查一下同一人那时去了哪里。'), ('switch', '查一下下一位那时去了哪里。'),
    ('drink', '也想知道他平时最偏爱哪种饮品。'), ('ordinary', '也想知道你最推荐哪种饮品。'),
    ('drink', '就刚提到的那位，她比较爱喝什么？'), ('history', '就刚提到的那位，原始记录说她爱喝什么？'),
    ('drink', '同一位，饮品喜好方面记了什么？'), ('repeat', '同一位，刚才查出的那个结果是什么？'),
    ('drink', '再看一下这个人的饮料偏好。'), ('switch', '再看一下温谨的饮料偏好。'),
)
SUPPORTED = frozenset(('drink', 'work', 'place'))
INTENTS = ('follow', 'switch', 'history', 'time', 'ordinary', 'repeat')
PRONOUN = re.compile(r'^(?:那|再查查|再问问|也查一下)?(?:他|她|那位|这位|此人|这个人)|^(?:还是)?同一(?:个)?(?:人|位|对象)')


def rows_for(group):
    return [dict(key=question, question=question, field=field, kind='follow' if field in SUPPORTED else field)
            for field, question in group]


def assess(rows, destinations):
    return dict(correct=sum(row['kind'] == 'follow' and field == row['field'] for row, field in zip(rows, destinations, strict=True)),
                wrong=sum(field is not None and (row['kind'] != 'follow' or row['field'] != field)
                          for row, field in zip(rows, destinations, strict=True)))


def guarded_field(row, cutoff):
    if row['subject_score'] >= cutoff and not SUBJECT_SCOPE_CONFLICT.search(row['question']):
        return row['direct']
    return None


def choose_guarded_cutoff(rows):
    def rank(cutoff):
        summary = assess(rows, [guarded_field(row, cutoff) for row in rows])
        return summary['wrong'], -summary['correct'], -cutoff

    return min((.5, .6, .7, .8, .9, .95, 1.), key=rank)


def train_intents(rows, features):
    vectors = torch.stack([features[row['key']][0] for row in rows])
    labels = torch.tensor([INTENTS.index(row['kind']) for row in rows])
    weights = labels.bincount(minlength=len(INTENTS)).float().reciprocal()
    torch.manual_seed(82437)
    head = torch.nn.Linear(vectors.shape[1], len(INTENTS))
    optimizer = torch.optim.AdamW(head.parameters(), lr=.025, weight_decay=.04)
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(head(vectors), labels, weight=weights)
        loss.backward()
        optimizer.step()
    return head.eval().requires_grad_(False)


def multiclass_field(row, cutoff):
    if row['intent'] == 'follow' and row['intent_score'] >= cutoff:
        return row['direct']
    return None


def main(database, output, head_path=None):
    digest = sha(database)
    groups = {name: rows_for(group) for name, group in (
        ('train', TRAIN), ('calibration', CALIBRATION), ('probe', PROBE), ('holdout', HOLDOUT),
    )}
    prompts = [row['question'] for rows in groups.values() for row in rows]
    if len(set(prompts)) != len(prompts):
        raise ValueError('Subject carryover prompts must be disjoint')
    chat = load_chat(database, use_inherited_router=True, use_retrained_gate=True)
    features = {question: features_for(chat.decoder, chat.tokenizer, dict(question=question), False) for question in prompts}
    pooled = {question: (intent_features(*feature[:2]), *feature[1:]) for question, feature in features.items()}
    head = train_inheritance(groups['train'], pooled)
    intent_head = train_intents(groups['train'], pooled)
    calibration = groups['calibration']
    cutoff = choose_cutoff(calibration, gate_probabilities(head, calibration, pooled))
    detailed_groups = {}
    for name in ('calibration', 'probe', 'holdout'):
        rows = groups[name]
        predictions = route(chat.direct, rows, features)
        scores = gate_probabilities(head, rows, pooled)
        logits = intent_head(torch.stack([pooled[row['key']][0] for row in rows]))
        intent_scores, intent_ids = logits.softmax(-1).max(-1)
        details = []
        for row, prediction, score, intent_score, intent_id in zip(rows, predictions, scores, intent_scores.tolist(),
                                                                  intent_ids.tolist(), strict=True):
            question = row['question']
            _, subject_status = query_subject(question, chat.addresses)
            field = prediction['selected'] if subject_status == 'no_subject' and route_action(prediction, chat.cutoff) == 'read' else None
            details.append(dict(row, direct=field, pronoun=field if PRONOUN.search(question) else None,
                                learned=field if score >= cutoff else None, subject_score=score,
                                intent=INTENTS[intent_id], intent_score=intent_score))
        detailed_groups[name] = details
    guarded_cutoff = choose_guarded_cutoff(detailed_groups['calibration'])
    multiclass_calibration = [dict(row, subject_score=row['intent_score'] if row['intent'] == 'follow' else 0.)
                              for row in detailed_groups['calibration']]
    multiclass_cutoff = choose_guarded_cutoff(multiclass_calibration)
    results = {}
    for name, rows in detailed_groups.items():
        for row in rows:
            row['guarded'] = guarded_field(row, guarded_cutoff)
            row['multiclass'] = multiclass_field(row, multiclass_cutoff)
            row['guarded_multiclass'] = row['multiclass'] if not SUBJECT_SCOPE_CONFLICT.search(row['question']) else None
        summary = {mode: assess(rows, [row[mode] for row in rows])
                   for mode in ('direct', 'pronoun', 'learned', 'guarded', 'multiclass', 'guarded_multiclass')}
        results[name] = dict(summary=summary, rows=rows)
        print(name, summary, flush=True)
    if sha(database) != digest:
        raise ValueError('Source memory database changed during audit')
    write(output, dict(cutoff=cutoff, guarded_cutoff=guarded_cutoff, multiclass_cutoff=multiclass_cutoff, database_sha256=digest,
                       train=groups['train'], results=results))
    if head_path is not None:
        atomic_save(dict(format=SameSubjectIntent.format, fingerprint=chat.recall_fingerprint,
                         cutoff=guarded_cutoff, state=head.state_dict()), head_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=DEFAULT_DATABASE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--head', type=Path, help='Export the binary same-subject head for opt-in chat')
    args = parser.parse_args()
    for path in (args.output, args.head):
        if path is not None and path.exists():
            raise FileExistsError(path)
    if args.head is not None and args.head.resolve() == args.output.resolve():
        parser.error('--output and --head require separate paths')
    main(args.database, args.output, args.head)
