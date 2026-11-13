"""Balanced question/name crossing for the joint-span research head."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

from scripts.deepseek_spora_boundary_train_pilot import digest, extract_names
from scripts.deepseek_spora_conversation_locator_audit import select_names
from scripts.deepseek_spora_joint_span_pilot import main

OUTPUT = ROOT / 'runs/deepseek-spora-joint-span-crossed-pilot-20261113'
TRAIN_TEMPLATES = ('请问{name}平时爱喝哪种饮品？', '你知道{name}钟爱什么饮料吗？',
                   '能告诉我{name}喜欢什么饮品吗？')
TEST_TEMPLATES = ('我想知道{name}目前喜欢的饮品。', '请查询{name}当前偏爱的饮料。')


def build_records(training, validation):
    train_names = sorted(extract_names(training) - extract_names(validation), key=digest)[:48]
    eval_names = [name for _, name in select_names(validation, {r['text'] for r in training})]
    if len(train_names) != 48 or set(train_names) & set(eval_names):
        raise ValueError('Name split failure')
    train, evaluation = [], []
    for name in train_names:
        for subject in (name, name + '子', '小' + name):
            for template in TRAIN_TEMPLATES:
                train.append({'question': template.format(name=subject), 'subject': subject, 'group': 'single'})
    for i in range(0, 48, 2):
        a, b = train_names[i:i + 2]
        for question in (f'{a}和{b}分别喜欢喝什么？', f'比较{a}与{b}偏爱的饮料。'):
            train.append({'question': question, 'subject': None, 'group': 'multiple'})
    for question in ('今天能帮我做什么？', '推荐一种饮料。', '你记得什么事情？', '说一个故事吧。',
                     '我现在该喝什么？', '天气怎么样？', '记忆是什么？', '请介绍一下自己。'):
        train.append({'question': question, 'subject': None, 'group': 'none'})
    for name in eval_names:
        for group, subject in (('known', name), ('suffix_seen', name + '子'),
                               ('suffix_unseen', name + '月'), ('prefix_seen', '小' + name),
                               ('prefix_unseen', '阿' + name)):
            for template_name, template in (('seen', TRAIN_TEMPLATES[0]), ('unseen', TEST_TEMPLATES[0])):
                evaluation.append({'question': template.format(name=subject), 'subject': subject,
                                   'reference': name, 'group': group + '_' + template_name})
    for i in range(0, 16, 2):
        a, b = eval_names[i:i + 2]
        for group, question in (('multiple_seen', f'比较{a}与{b}偏爱的饮料。'),
                                ('multiple_unseen', f'你记得{a}以及{b}爱喝什么吗？')):
            evaluation.append({'question': question, 'subject': None, 'group': group})
    for question in ('你能推荐饮品吗？', '我该怎么休息？', '你现在在想什么？', '有记录吗？',
                     '讲个笑话。', '什么是学习？', '我最喜欢什么？', '当前有什么信息？'):
        evaluation.append({'question': question, 'subject': None, 'group': 'none_unseen'})
    return train, evaluation, len(train_names)


if __name__ == '__main__':
    main(output=OUTPUT, build_records=build_records, steps=2400)
