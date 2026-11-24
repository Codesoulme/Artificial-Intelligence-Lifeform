"""Compare typed memory-version choices across balanced option positions."""

from collections import Counter
import unicodedata

import torch

from .research_version_scope import _DATE_HINT, _ORDINAL_HINT


LETTERS = ('A', 'B', 'C', 'D')
ORDERS = tuple(tuple((position + shift) % len(LETTERS) for position in range(len(LETTERS)))
               for shift in range(len(LETTERS)))
NAMES = ('当前版', '首版', '紧邻前版', '暂缓读取')
DESCRIPTIONS = (
    '请求读取当前仍生效的版本，已被取代的版本不算当前。',
    '请求读取最早生效的版本，即这条记录最初采用的内容。',
    '请求读取当前生效版之前紧挨着的一版，即最近一次有效切换前的内容。',
    '话语明确禁止读取、只排除某版却没有指明要查哪版，或目标不能确定。',
)
EXAMPLES = (
    '「最新用的那张先不看，我要开头填表时那张。」选首版。',
    '「初次登记可以略过，告诉我眼下仍在用的。」选当前版。',
    '「从这次开始用新表算起，换表以前紧挨着的是哪份？」选紧邻前版。',
    '「划掉现在用的那份，剩下哪版我还没想好。」选暂缓读取。',
)


def menu_prompt(question, order, *, examples=False):
    labels = '\n'.join(f'{letter}. {NAMES[index]}。{DESCRIPTIONS[index]}'
                       for letter, index in zip(LETTERS, order, strict=True))
    demonstrations = '\n例子：\n' + '\n'.join(EXAMPLES) if examples else ''
    return (f'判断原话要求读取哪一版个人记录。区分要查阅的内容和要排除的内容。\n'
            f'{labels}{demonstrations}\n原话：{question}\n答案只写 A、B、C 或 D 中的一个字母。')


def relative_menu_allowed(question, address, scope_reason):
    text = unicodedata.normalize('NFKC', question)
    return (address is not None and scope_reason != 'outside_lookup_scope'
            and not _DATE_HINT.search(text) and not _ORDINAL_HINT.search(text))


def menu_evidence(logits):
    """Return semantic winners after mapping each physical position back."""
    scores = torch.as_tensor(logits, dtype=torch.float64)
    if scores.shape != (len(ORDERS), len(LETTERS)) or not torch.isfinite(scores).all():
        raise ValueError('Invalid menu scores')
    positions = scores.argmax(-1).tolist()
    votes = [order[position] for order, position in zip(ORDERS, positions, strict=True)]
    semantic = torch.stack([row[list(map(order.index, range(len(LETTERS))))]
                            for row, order in zip(scores, ORDERS, strict=True)])
    probabilities = semantic.mean(0).softmax(-1)
    best, options = probabilities.topk(2)
    chosen = int(options[0])
    return {'votes': votes, 'positions': positions, 'option': chosen, 'agreement': Counter(votes)[chosen],
            'margin': float(best[0] - best[1])}
