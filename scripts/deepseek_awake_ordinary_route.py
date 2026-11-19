"""Test recognizing ordinary generation requests separately from stored-fact queries."""

import argparse
from pathlib import Path

from deepseek_awake_recall_intent import DEFAULT_DATABASE, choose_cutoff, rows_for
from deepseek_continuous_memory_chat import load_chat
from deepseek_memory_attribute_span import route_action
from deepseek_memory_dialogue_context import features_for, route
from deepseek_memory_turn_priority import gate_probabilities, train_inheritance
from deepseek_output_aligned_memory import sha, write
from spora.awake_recall_intent import OrdinaryIntent, intent_features
from spora.language_training import atomic_save


TRAIN = {
    'follow': (
        '为何冰块放在桌上会融化？', '杯里的茶为什么会变凉？', '饮料冰镇以后甜味会减弱吗？',
        '推荐几种适合晚上喝的无糖饮料。', '给我一个面试开头的例子。', '介绍一下什么是温差。',
        '把这句话翻成中文：Good afternoon.', '能帮我列一下今天开会的准备清单吗？',
        '画一张旅行准备流程图。', '我今天有点累。', '早上好，很高兴见到你。',
        '给我写一小段描述公共图书馆的散文。', '普通温度计是怎么工作的？', '说说散步有哪些好处。',
        '怎样提高求职面试的表达能力？', '天气冷的时候饮料放哪里合适？', '怎么按时间整理资料？',
        '生日会上可以说哪些祝福语？', '写一段虚构人物的职业介绍。', '改写这句：她爱喝绿茶。',
        '谢谢，我已经听到了。', '不查了，你休息一下吧。', '给我解释一下时间戳。', '设计一个校园竞赛流程。',
    ),
    'memory': (
        '他以前在做什么工作？', '那个人最爱喝哪种饮料？', '她那次到哪里去了？', '这人最喜欢的饮品是什么？',
        '数据库里记的那个人喜欢喝什么？', '前一次记录里他去了哪里？', '现在那条记录写的是哪种饮品？',
        '她过去靠什么工作生活？', '哪一天登记了他的到访地点？', '那位客人是从事哪个行业的？',
        '请复述你刚刚查出的那一项。', '具体是什么？', '告诉我上面查询的值。', '你有没有上次那笔记录？',
        '沈亦喜欢喝哪种饮料？', '齐宁那次去了哪个地方？', '温谨的工作有记录吗？',
        '上一版到访的是哪个地点？', '以前登记的职业是什么？', '查询余安近来最常买的饮料。',
        '那个人的偏好有更新吗？', '你还记得他的职务吗？', '在我更正之前，你存的是什么？',
        '回想一下之前那条记录的具体内容。',
    ),
}
CALIBRATION = {
    'follow': ('热咖啡变凉和室温有什么关系？', '帮我介绍一下面试的流程。', '推荐几种可以放凉的饮料。',
               '用中文解释一下冲泡温度。', '为故事里一名修理工写一句台词。', '周末去哪里散步比较好？'),
    'memory': ('这份记录中的当事人去了哪里？', '他实际从事的是哪份职业？', '他最喜欢喝哪款？',
               '查完请读出找到的值。', '原记录里她最爱喝什么？', '你还记不记得他偏好的那种饮品？'),
}
HOLDOUT = {
    'follow': (
        '为什么热水会慢慢冷却？', '帮我写一句开会前的问候。', '推荐一杯少糖的饮料。', '开水放冰箱会发生什么？',
        '面试时怎样介绍自己的经验？', '告诉我去图书馆学习的三个好处。', '用通俗语言解释热传导。',
        '三只杯子各装两块冰，一共多少块？', '请把「她最喜欢喝什么」翻译成英语。', '写一段旅行日记的开头。',
        '请写一个角色扮演的导游开场。', '我们换个话题，聊聊茶。', '不继续查了，先跟我聊聊电影。',
        '有适合新手的晨练建议吗？', '如果明天降温，出门怎么穿比较好？', '解释为什么面试前要做好准备。',
    ),
    'memory': (
        '那个人平常爱喝的饮品是什么？', '她后来去哪个地方了？', '这是谁的职业记录？',
        '帮我查他的工作是干什么的。', '你刚才报的值，可否重说一次？', '前面那条的结果是什么来着？',
        '那他什么时候去的？', '刚才那个答案说的是谁？', '那另一个人最喜欢喝什么？',
        '原先记的职业是哪份？', '之前有没有登记他的偏好？', '请告诉我记录里写的饮品。',
        '齐宁喜欢什么饮品？', '周平去过什么地方？', '沈亦上一版喜欢喝什么？', '继续查那个人的职业。',
    ),
}


def main(database, output, head_path=None):
    digest = sha(database)
    chat = load_chat(database, use_inherited_router=True, use_retrained_gate=True)
    train, calibration, holdout = (rows_for(group) for group in (TRAIN, CALIBRATION, HOLDOUT))
    prompts = [row['question'] for row in (*train, *calibration, *holdout)]
    if len(prompts) != len(set(prompts)):
        raise ValueError('Ordinary-route prompts must be disjoint')
    features = {prompt: features_for(chat.decoder, chat.tokenizer, dict(question=prompt), False) for prompt in prompts}
    pooled = {prompt: (intent_features(*value[:2]), *value[1:]) for prompt, value in features.items()}
    head = train_inheritance(train, pooled)
    cutoff = choose_cutoff(calibration, gate_probabilities(head, calibration, pooled))
    candidates = gate_probabilities(head, holdout, pooled)
    details = []
    for row, old, probability in zip(holdout, route(chat.direct, holdout, features), candidates, strict=True):
        baseline = route_action(old, chat.cutoff) == 'ordinary'
        details.append(dict(row, baseline=baseline, candidate=baseline or probability >= cutoff,
                            field=old['selected'], ordinary_score=probability))
    ordinary = [row for row in details if row['kind'] == 'follow']
    memory = [row for row in details if row['kind'] == 'memory']
    summary = {mode: dict(correct=sum(row[mode] for row in ordinary), false=sum(row[mode] for row in memory))
               for mode in ('baseline', 'candidate')}
    if sha(database) != digest:
        raise ValueError('Source memory database changed during audit')
    print(summary, flush=True)
    write(output, dict(cutoff=cutoff, train=train, calibration=calibration, rows=details, summary=summary, database_sha256=digest))
    if head_path is not None:
        atomic_save(dict(format=OrdinaryIntent.format, fingerprint=chat.recall_fingerprint, cutoff=cutoff,
                         state=head.state_dict()), head_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=DEFAULT_DATABASE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--head', type=Path, help='Export the ordinary-intent head for opt-in chat')
    args = parser.parse_args()
    for path in (args.output, args.head):
        if path is not None and path.exists():
            raise FileExistsError(path)
    if args.head is not None and args.head.resolve() == args.output.resolve():
        parser.error('--output and --head require separate paths')
    main(args.database, args.output, args.head)
