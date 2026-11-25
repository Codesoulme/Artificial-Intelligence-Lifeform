"""Clause-level evidence for selecting a speaker's requested memory field."""

from dataclasses import dataclass
import re

from scripts.spora_composition_address_cases import TRAIN_NAMES
from scripts.spora_field_authority_cases import PAIRS, TOPICS


SKIP, EXCLUDE, READ = range(3)
BOUNDARY = re.compile(r'[^，,。；;！？!?\n]+[，,。；;！？!?\n]*')


@dataclass(frozen=True)
class Task:
    question: str
    owner: str | None
    fields: tuple[str, ...]
    clauses: tuple[tuple[int, str, int], ...] = ()


def clauses(text):
    return [(match.start(), match.end()) for match in BOUNDARY.finditer(text)]


def training():
    for index, name in enumerate(TRAIN_NAMES):
        for first, second in PAIRS:
            for use, skip in ((first, second), (second, first)):
                a, b = TOPICS[use][index % 3], TOPICS[skip][(index + 1) % 3]
                yield Task(f'{name}的{b}别用存档的，只把之前登记的{a}写上。', name,
                           (use,), ((0, skip, EXCLUDE), (1, use, READ)))
                yield Task(f'把{name}记过的{a}写进消息，{b}我自己重新选。', name,
                           (use,), ((0, use, READ), (1, skip, EXCLUDE)))
                yield Task(f'{name}原来定过{a}和{b}，这次两个都另选。', name, (),
                           ((1, use, EXCLUDE), (1, skip, EXCLUDE)))
                yield Task(f'把{name}之前登记的{a}和{b}都报一下。', name,
                           (use, skip), ((0, use, READ), (0, skip, READ)))
                yield Task(f'我原先要查{name}登记的{a}，现在先不查{a}了。', name, (),
                           ((0, use, READ), (1, use, EXCLUDE)))
                yield Task(f'暂时别提{name}登记的{a}，现在想好了请把{a}查出来。', name,
                           (use,), ((0, use, EXCLUDE), (1, use, READ)))
        for field, aliases in TOPICS.items():
            topic = aliases[(index + 1) % 3]
            yield Task(f'替{name}把原来定好的{topic}补上。', name, (field,), ((0, field, READ),))
            yield Task(f'请解释“{name}登记的{topic}”这个说法。', name, (), ((0, field, EXCLUDE),))
            yield Task(f'给{name}再挑一种新的{topic}，原来那项别读。', name, (), ((0, field, EXCLUDE),))
            yield Task(f'我问的是殷晓连原先说的{topic}，不是{name}的{topic}。', name, (),
                       ((1, field, EXCLUDE),))
        yield Task(f'{name}今天有点疲惫，帮我写一句打气的话。', name, ())


HELD = (
    Task('赵茗初要拿什么就照旧记的说，碰头处我今天另外找。', '赵茗初', ('items',),
         ((0, 'items', READ), (1, 'places', EXCLUDE))),
    Task('赵茗初会面的老地方帮我核一下，至于物品这次不要报。', '赵茗初', ('places',),
         ((0, 'places', READ), (1, 'items', EXCLUDE))),
    Task('我不问何青玟在哪儿碰面，只提醒她把说好的东西捎来。', '何青玟', ('items',),
         ((0, 'places', EXCLUDE), (1, 'items', READ))),
    Task('何青玟带的东西我有数，原来约的碰头地儿能不能查给我？', '何青玟', ('places',),
         ((0, 'items', EXCLUDE), (1, 'places', READ))),
    Task('给许昀芝备过去常喝的那份，书名先一个字也别写。', '许昀芝', ('drinks',),
         ((0, 'drinks', READ), (1, 'books', EXCLUDE))),
    Task('许昀芝今天要尝新口味，借阅档里那本的名字还要告诉我。', '许昀芝', ('books',),
         ((0, 'drinks', EXCLUDE), (1, 'books', READ))),
    Task('写给顾绮言的短讯把以前约的去处和他登记的那本书都交代清楚。', '顾绮言', ('places', 'books'),
         ((0, 'places', READ), (0, 'books', READ))),
    Task('我改主意了，顾绮言的书名别往文案里放，碰头处仍按旧约定。', '顾绮言', ('places',),
         ((1, 'books', EXCLUDE), (2, 'places', READ))),
    Task('地点另挑不走旧安排，岑婉听原先说要带的那件东西仍写进提醒。', '岑婉听', ('items',),
         ((0, 'places', EXCLUDE), (1, 'items', READ))),
    Task('黎书然借的那册和她惯常点的饮料都照存档告诉我。', '黎书然', ('books', 'drinks'),
         ((0, 'books', READ), (0, 'drinks', READ))),
    Task('先说黎书然平常那杯吧，等等，那杯先不查了。', '黎书然', (),
         ((0, 'drinks', READ), (2, 'drinks', EXCLUDE))),
    Task('我本来不想查许昀芝借的书，想了想还是把她存的书名报一下。', '许昀芝', ('books',),
         ((0, 'books', EXCLUDE), (1, 'books', READ))),
    Task('我想问的是纪如一过去喝哪款，和黎书然的饮品无关。', '黎书然', (), ((1, 'drinks', EXCLUDE),)),
    Task('翟声然要带什么看他原来的记录就行，别把赵茗初那件写出来。', '赵茗初', (), ((1, 'items', EXCLUDE),)),
    Task('“岑婉听选的是哪件物品”这句话能不能说得更礼貌？', '岑婉听', (), ((0, 'items', EXCLUDE),)),
    Task('给顾绮言解释书页进水后该怎么处理。', '顾绮言', ()),
    Task('给许昀芝另外推荐两种饮料，先别查档。', '许昀芝', (), ((0, 'drinks', EXCLUDE),)),
    Task('何青玟和赵茗初各约在什么地方？', None, ()),
)
