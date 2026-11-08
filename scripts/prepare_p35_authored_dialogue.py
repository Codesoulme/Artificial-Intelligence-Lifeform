"""Build a small, deterministic, role-explicit Chinese dialogue pilot."""
import hashlib
import json
import argparse
from pathlib import Path
import sys

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_conversation_v2 import encode_dialogue, sha

SOURCE = ROOT / 'runs/teaching-curriculum-v5'
OUT = ROOT / 'runs/p35-authored-dialogue-v1'
BASE = ROOT / 'runs/conversation-data-v5'

DIALOGUES = [
    [('user', '周末我要去取一件包裹。'), ('assistant', '记下了：周末去取包裹。'),
     ('user', '取件时间改到周日下午，地点不变。'), ('assistant', '好的，改为周日下午取包裹，地点不变。'),
     ('user', '请复述最新安排。'), ('assistant', '周日下午去原来的地点取包裹。')],
    [('user', '抽屉里有四支铅笔，我送给同桌一支。'), ('assistant', '送出一支后，抽屉里还剩三支。'),
     ('user', '同桌后来还回一支。'), ('assistant', '还回一支后，现在有四支。'),
     ('user', '最后剩几支？'), ('assistant', '四支。')],
    [('user', '兔子叫团团，乌龟叫慢慢。'), ('assistant', '明白了，兔子叫团团，乌龟叫慢慢。'),
     ('user', '团团吃胡萝卜，慢慢喜欢晒太阳。谁吃胡萝卜？'), ('assistant', '团团吃胡萝卜。'),
     ('user', '用一句话介绍慢慢。'), ('assistant', '慢慢是一只喜欢晒太阳的乌龟。')],
    [('user', '我想给表哥买礼物，他喜欢下棋，预算八十元。'), ('assistant', '可以看看八十元以内的棋谱、棋盘收纳袋或计时器。'),
     ('user', '他已经有计时器了。'), ('assistant', '那就避开计时器，考虑棋谱或收纳袋。'),
     ('user', '只保留一个建议。'), ('assistant', '可以买一本适合他的棋谱。')],
    [('user', '厨房台面上有杯子、调料和一叠纸。'), ('assistant', '可以先把纸移到干燥处，再集中整理杯子和调料。'),
     ('user', '我只有五分钟。'), ('assistant', '先把纸移走，避免它们被水或油弄脏。'),
     ('user', '先做哪件事？'), ('assistant', '先把纸移到干燥处。')],
    [('user', '把“请尽快回复我”改得更柔和。'), ('assistant', '方便时回复我就好。'),
     ('user', '再正式一些。'), ('assistant', '烦请您方便时回复。'),
     ('user', '保留原来的意思。'), ('assistant', '希望您在方便时尽快回复。')],
    [('user', '天气预报说可能降温，我带了一件外套。'), ('assistant', '带外套是为可能的降温做准备。'),
     ('user', '后来没有降温，这个决定就一定错误吗？'), ('assistant', '不一定。决定是根据当时的信息作出的，结果没有发生不等于准备本身错误。')],
    [('user', '我答应把资料发给同事，但今天忘了。'), ('assistant', '可以尽快说明情况，并告诉对方明确的发送时间。'),
     ('user', '我不想找借口。'), ('assistant', '那就直接承认忘记了并补上资料，这样更诚实也更负责。'),
     ('user', '这体现什么品质？'), ('assistant', '承认疏漏并及时补救，体现诚信和责任感。')],
    [('user', '个人层面的核心价值词语有哪四个？'), ('assistant', '个人层面包括爱国、敬业、诚信、友善。'),
     ('user', '认真完成分内工作主要对应哪个词？'), ('assistant', '主要对应敬业。'),
     ('user', '守信和待人友好分别对应什么？'), ('assistant', '守信对应诚信，待人友好对应友善。')],
    [('user', '冰饮料放在桌上，杯外为什么会有水珠？'), ('assistant', '空气中的水汽遇到较冷的杯子后凝结成小水珠。'),
     ('user', '换成更容易懂的说法。'), ('assistant', '杯子很冷，空气里的水汽碰到它就变成了水珠。')],
    [('user', '家里有土豆和胡萝卜，想做简单的菜。'), ('assistant', '可以把它们洗净切块后一起炖煮。'),
     ('user', '我不想放太多调料。'), ('assistant', '那就少放调料，先突出食材本身的味道。'),
     ('user', '已知食材是什么？'), ('assistant', '已知食材是土豆和胡萝卜。')],
    [('user', '我住的地方还没有告诉你。'), ('assistant', '对，我目前不知道你所在的地方。'),
     ('user', '那你能说出附近的公交站吗？'), ('assistant', '不能仅凭现在的信息确定。需要先知道城市或附近地标。')],
    [('user', '我想每周练三次字，每次十五分钟。'), ('assistant', '可以选三个固定时段，每次练习十五分钟。'),
     ('user', '周五总是有事。'), ('assistant', '那就避开周五，选择其他三个较稳定的时段。'),
     ('user', '一周练几次？'), ('assistant', '每周三次。')],
    [('user', '我今天的演讲效果不好，有些沮丧。'), ('assistant', '效果不如预期确实会让人沮丧。你想先休息，还是一起找一个具体问题？'),
     ('user', '先让我安静一会儿。'), ('assistant', '好，先安静休息一会儿，等你准备好再继续。')],
    [('user', '红色盒子里有邮票，白色盒子里有电池。'), ('assistant', '记下了：红色盒子放邮票，白色盒子放电池。'),
     ('user', '邮票移到白色盒子，电池仍在白色盒子。'), ('assistant', '更新后，邮票和电池都在白色盒子。'),
     ('user', '红色盒子现在有什么？'), ('assistant', '根据目前的信息，红色盒子里没有提到物品。')],
    [('user', '仓库有六个可用空箱子，用掉两个，又补进三个。'), ('assistant', '用掉两个后剩四个，再补进三个就是七个可用空箱子。'),
     ('user', '补进的箱子后来坏了一个。'), ('assistant', '坏掉一个后，还剩六个可用空箱子。'),
     ('user', '现在有几个可用空箱子？'), ('assistant', '六个。')],
    [('user', '我想让房间更容易找到东西，但不知道收纳盒尺寸。'), ('assistant', '在不知道尺寸前，可以先按物品用途分组，并记录每组的大致数量。'),
     ('user', '先给一个不需要尺寸的动作。'), ('assistant', '先把同类物品放在一起。')],
    [('user', '明早要带水壶和门卡。'), ('assistant', '明早要带水壶和门卡。'),
     ('user', '门卡改为钥匙，水壶不变。'), ('assistant', '已更新：明早带水壶和钥匙，不带门卡。'),
     ('user', '需要带什么？'), ('assistant', '带水壶和钥匙。')],
    [('user', '我想开始骑车，今天只骑了十五分钟。'), ('assistant', '十五分钟可以作为开始，之后按身体感受逐步增加。'),
     ('user', '你自己骑车有多久了？'), ('assistant', '我没有身体或亲身骑车经历，但可以根据你提供的情况给一般建议。')],
    [('user', '把“我打算晚上去便利店买牛奶”压缩一下。'), ('assistant', '我晚上去便利店买牛奶。'),
     ('user', '不要改变原意。'), ('assistant', '我计划晚上去便利店买牛奶。'),
     ('user', '句子说了什么？'), ('assistant', '你计划晚上去便利店买牛奶。')],
    [('user', '我有米和青豆，想做一份简单的饭。'), ('assistant', '可以把米和青豆一起煮熟，做成简单的青豆饭。'),
     ('user', '我没有说有没有肉。'), ('assistant', '对，目前不知道有没有肉；只根据已知信息，可以做米饭和青豆。'),
     ('user', '只总结已知食材。'), ('assistant', '已知食材是米和青豆。')],
    [('user', '我记得预约在周四九点，地点是一号大厅。'), ('assistant', '当前记录是周四九点，在一号大厅。'),
     ('user', '时间改成十点，地点不变。'), ('assistant', '已更新为周四十点，地点仍是一号大厅。'),
     ('user', '地点有变化吗？'), ('assistant', '没有，仍是一号大厅。')],
    [('user', '朋友没有及时回信，我有点不安。'), ('assistant', '暂时没有回信可能有多种原因，仅凭这一点不能确定对方的想法。'),
     ('user', '我怕自己让他不舒服。'), ('assistant', '可以等一会儿，再用平和的方式询问；现在还不能断定是你的话造成的。')],
    [('user', '我有五张贴纸，送给妹妹两张。'), ('assistant', '送出两张后，你还剩三张。'),
     ('user', '妹妹又还给我一张。'), ('assistant', '还回一张后，你现在有四张。'),
     ('user', '最后有几张？'), ('assistant', '四张。')],
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('authored', 'singleturn-control'), default='authored')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    output = ROOT / (args.output or ('runs/p35-authored-dialogue-v1' if args.mode == 'authored'
                                    else 'runs/p35-singleturn-control-v1'))
    if output.exists():
        raise FileExistsError(output)
    source_manifest = json.loads((SOURCE / 'manifest.json').read_text('utf-8'))
    if sha(SOURCE / 'manifest.json') != 'c2969c7d6dc86138810009c982bff93a6bc173ad6b21d928d034c7c86580775d':
        raise ValueError('Reviewed parent curriculum changed')
    for name, digest in source_manifest['files'].items():
        if sha(SOURCE / name) != digest:
            raise ValueError('Parent curriculum artifact changed: ' + name)
    tokenizer_path = BASE / 'tokenizer.json'
    base_manifest = json.loads((BASE / 'manifest.json').read_text('utf-8'))
    if sha(tokenizer_path) != base_manifest['files']['tokenizer.json']:
        raise ValueError('Tokenizer changed')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    output.mkdir(parents=True)
    for name in ('foundations.jsonl', 'foundations-tokens.jsonl'):
        (output / name).write_bytes((SOURCE / name).read_bytes())
    stats = {'dialogues': 0, 'assistant_turns': 0, 'assistant_targets': 0}
    parent_teacher = SOURCE / 'teacher.jsonl' if args.mode == 'singleturn-control' else None
    source_rows = [json.loads(line) for line in parent_teacher.open(encoding='utf-8')] if parent_teacher else None
    authored = DIALOGUES if args.mode == 'authored' else None
    with (output / 'teacher.jsonl').open('x', encoding='utf-8') as raw, \
            (output / 'teacher-tokens.jsonl').open('x', encoding='utf-8') as encoded:
        rows = source_rows if source_rows is not None else [None] * len(authored)
        for index, source_row in enumerate(rows):
            if source_row is not None:
                row = source_row
                messages = row['messages']
                with (SOURCE / 'teacher-tokens.jsonl').open(encoding='utf-8') as source_encoded:
                    token_row = next(json.loads(line) for line in source_encoded
                                     if json.loads(line).get('group') == row['group'])
                raw.write(json.dumps(row, ensure_ascii=False) + '\n')
                encoded.write(json.dumps(token_row) + '\n')
                stats['dialogues'] += 1
                stats['assistant_turns'] += sum(m['role'] == 'assistant' for m in messages)
                stats['assistant_targets'] += sum(token_row['mask'])
                continue
            dialogue = authored[index]
            messages = [dict(role=role, content=content) for role, content in dialogue]
            group = hashlib.sha256(('p35-authored:' + str(index)).encode()).hexdigest()
            text = ''.join(('<|user|>' if role == 'user' else '<|assistant|>') + content + '<|eos|>'
                           for role, content in dialogue)
            digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
            result = encode_dialogue(tokenizer, messages, limit=512)
            if result is None:
                raise ValueError('Authored dialogue exceeds context')
            tokens, mask = result
            row = dict(messages=messages, text=text, group=group, sha256=digest,
                       source='codex_authored_p35', index=index)
            raw.write(json.dumps(row, ensure_ascii=False) + '\n')
            encoded.write(json.dumps(dict(tokens=tokens, mask=mask, group=group)) + '\n')
            stats['dialogues'] += 1
            stats['assistant_turns'] += sum(role == 'assistant' for role, _ in dialogue)
            stats['assistant_targets'] += sum(mask)
    result = dict(status='prepared_pending_independent_review', ready_for_training=False,
                  source=('locally authored by Codex; no external model or corpus' if args.mode == 'authored'
                          else '42 previously reviewed single-turn rows from teaching-curriculum-v5'),
                  source_curriculum_manifest_sha256=sha(SOURCE / 'manifest.json'),
                  tokenizer_sha256=sha(tokenizer_path), stats=stats,
                  files={name: sha(output / name) for name in
                         ('teacher.jsonl', 'teacher-tokens.jsonl', 'foundations.jsonl', 'foundations-tokens.jsonl')},
                  design=(['Explicit user/assistant roles and complete short histories.',
                          'Tasks cover reference, correction, unknown facts, instruction following, and user values.',
                          'Assistant claims are limited to the supplied conversation; no biography is assigned.',
                          'Small authored set may overfit and cannot establish general chat ability.'] if args.mode == 'authored'
                          else ['Existing independently reviewed 42-row single-turn subset.',
                                'Control tests whether an equal dialogue-token budget without multi-turn histories has different effects.']))
    (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
