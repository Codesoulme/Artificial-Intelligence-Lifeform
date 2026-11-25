"""Contrasts between requested, excluded and unmentioned memory fields."""

from dataclasses import dataclass

from scripts.spora_composition_address_cases import Request, TRAIN_NAMES, train_cases


@dataclass(frozen=True)
class LabeledRequest(Request):
    excluded: tuple[str, ...] = ()


TOPICS = {
    'drinks': ('饮品', '常点的那杯', '喜欢喝的那一种'),
    'places': ('会合地点', '原先约好的位置', '说好的集合处'),
    'items': ('要携带的用品', '记下的那件东西', '要拿的物件'),
    'books': ('登记的书名', '借过的那册读物', '记下的那本书'),
}
PAIRS = (
    ('drinks', 'books'), ('places', 'items'), ('places', 'books'), ('items', 'drinks'),
)
FORMS = (
    '给{name}写一句，{use}按之前的登记填，{skip}这回另外选。',
    '{name}的{skip}不用查，只给我原先登记的{use}。',
    '已经知道{name}的{skip}了，别写这一项，只补上已有记录里的{use}。',
    '关于{name}，我只确认{use}还是旧的，先别提{skip}。',
)


def training():
    for row in train_cases():
        yield LabeledRequest(row.question, row.owner, row.fields)
    for index, name in enumerate(TRAIN_NAMES):
        for first, second in PAIRS:
            for form_index, form in enumerate(FORMS):
                for use, skip in ((first, second), (second, first)):
                    choice = (index + form_index) % 3
                    prompt = form.format(name=name, use=TOPICS[use][choice], skip=TOPICS[skip][(choice + 1) % 3])
                    yield LabeledRequest(prompt, name, (use,), (skip,))
        for field, aliases in TOPICS.items():
            topic = aliases[index % 3]
            yield LabeledRequest(f'给{name}重新挑{topic}，原来的不采用。', name, (), (field,))


HELD = (
    LabeledRequest('顾绮言要带的书我手头有，书名不必写。帮我把跟他约好的会面地点找出来。', '顾绮言', ('places',), ('books',)),
    LabeledRequest('地点我另找，先把顾绮言说要带的那本写全，照以前的借阅记录。', '顾绮言', ('books',), ('places',)),
    LabeledRequest('赵茗初来见面要捎上什么？地点那一栏留空，只用上回确认带的东西。', '赵茗初', ('items',), ('places',)),
    LabeledRequest('写给赵茗初只说在哪碰头，照原先存的。带什么等会儿另外定。', '赵茗初', ('places',), ('items',)),
    LabeledRequest('何青玟已经把东西备齐了，别查清单。我只差早先说好的碰头地址。', '何青玟', ('places',), ('items',)),
    LabeledRequest('碰头处都清楚，回何青玟的时候只报登记的携带物。', '何青玟', ('items',), ('places',)),
    LabeledRequest('许昀芝爱喝哪一款？书的名字我有，只查她常喝的。', '许昀芝', ('drinks',), ('books',)),
    LabeledRequest('许昀芝喝的我想换个新品。借阅记录上的书名照原样给我。', '许昀芝', ('books',), ('drinks',)),
    LabeledRequest('黎书然借的那本叫什么？她的饮料别翻档，今天喝我新买的。', '黎书然', ('books',), ('drinks',)),
    LabeledRequest('黎书然的书不提了。我只要她平时点的饮品，查一下已有记录。', '黎书然', ('drinks',), ('books',)),
    LabeledRequest('替岑婉听把已确定的见面位置和需要携带的东西一起说出来。', '岑婉听', ('places', 'items')),
    LabeledRequest('告诉许昀芝哪本书已留好，再报上她先前登记的饮品，一条就行。', '许昀芝', ('books', 'drinks')),
    LabeledRequest('赵茗初的碰头地点和要带的物件都另选，给我两个建议。', '赵茗初', (), ('places', 'items')),
    LabeledRequest('想请黎书然尝个新口味，借阅的书也先不管。', '黎书然', (), ('drinks', 'books')),
    LabeledRequest('顾绮言祝我早安，回他一句简单的。', '顾绮言'),
    LabeledRequest('给何青玟解释木栈道和普通小桥有什么区别。', '何青玟'),
    LabeledRequest('翻译“岑婉听要带什么”，原句里的人名保留。', '岑婉听'),
    LabeledRequest('我现在问的是岑江澜先前挑的读物，不是岑婉听的。', None),
)
