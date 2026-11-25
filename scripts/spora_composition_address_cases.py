"""Address selection for natural requests without visible stored values."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    question: str
    owner: str | None
    fields: tuple[str, ...] = ()


TRAIN_NAMES = ('冬语晨', '尤诗蕾', '欧心诚', '魏千沂', '戚若楠', '邵静微')
FORMS = (
    ('给{name}写一条消息，按之前记的地方会合。', ('places',)),
    ('我跟{name}约的是哪个碰头点，麻烦说一下原来登记的那个。', ('places',)),
    ('提醒{name}把之前说要带的那件物品带上。', ('items',)),
    ('写给{name}的提示里请填上档案中那件要携带的东西。', ('items',)),
    ('按先前登记给{name}准备爱喝的那杯，并说出名称。', ('drinks',)),
    ('我好像忘了{name}习惯点什么饮料，拿已有记录核对一下。', ('drinks',)),
    ('给{name}写取书留言，书名从借阅记录取。', ('books',)),
    ('提醒{name}归还之前登记过的读物，消息里要写标题。', ('books',)),
    ('给{name}的微信里写清旧约定在哪集合、需要带哪个物件。', ('places', 'items')),
    ('用已经登记的信息告诉{name}在哪儿碰头和拿哪件东西。', ('places', 'items')),
    ('想约{name}在原定碰头点取之前说的读物。写清地点和书名。', ('places', 'books')),
    ('告诉{name}把那本已登记的书带来，去哪儿见也照档案写。', ('books', 'places')),
    ('替{name}留好他选的饮料和书。按原记录写清两样名字。', ('drinks', 'books')),
    ('给{name}说取读物时我会带饮料，两项都用记过的具体名字。', ('books', 'drinks')),
    ('查一下{name}的用品当前版。', ('items',)),
    ('请核对{name}的见面地点当前版。', ('places',)),
    ('说说{name}的饮品当前版。', ('drinks',)),
    ('告诉我{name}的书目当前版。', ('books',)),
    ('请帮我解释{name}这句话里“饮料”是什么意思，不用查登记。', ()),
    ('给{name}推荐一种新的饮料，我想送点不一样的东西。', ()),
    ('给{name}推荐一本适合入门的科普书。', ()),
    ('给{name}说一声辛苦了，写得自然一点。', ()),
    ('不用查{name}的会面地点，给我一个新场地的选择方法。', ()),
    ('{name}问纸书卷边了怎么办，给个建议。', ()),
    ('先别查{name}的用品，帮我理解什么叫日常用品。', ()),
    ('我只想润色“{name}的书叫什么”这几个字。', ()),
)


def train_cases():
    for name in TRAIN_NAMES:
        for form, fields in FORMS:
            yield Request(form.format(name=name), name, fields)


HELD = (
    Request('给赵茗初拟一句可直接发的话，把咱们说好的碰面地点和该捎上的东西都写进去。', '赵茗初', ('places', 'items')),
    Request('麻烦查赵茗初的用品当前版。', '赵茗初', ('items',)),
    Request('我先去找何青玟，把登记的碰头处发给我。', '何青玟', ('places',)),
    Request('告诉何青玟带上原来记下的那件东西，见面的地儿也照原安排填。', '何青玟', ('items', 'places')),
    Request('许昀芝想把她借阅的那本拿走，帮我把书的名字列一下，用存档的。', '许昀芝', ('books',)),
    Request('给许昀芝留她常喝的那一款，从原记录选就行。', '许昀芝', ('drinks',)),
    Request('我想告诉许昀芝书已经备好了，也会带她记过的饮料，两样都写全。', '许昀芝', ('books', 'drinks')),
    Request('给顾绮言写一句，让他拿上登记过的书，去已经约定的地方会合。', '顾绮言', ('books', 'places')),
    Request('顾绮言选中的那本读物叫啥？看一下存档。', '顾绮言', ('books',)),
    Request('岑婉听来时别忘了带约好的物件，告诉她具体带什么。', '岑婉听', ('items',)),
    Request('我不记得跟岑婉听说好了在哪儿见，只报之前登记的地点。', '岑婉听', ('places',)),
    Request('回黎书然一句，读物和她平常那杯都备好了，名字按档案填。', '黎书然', ('books', 'drinks')),
    Request('请核实黎书然的饮品当前版与书目当前版。', '黎书然', ('drinks', 'books')),
    Request('给顾绮言推荐一本关于野生动物的书，不是找他借过的书。', '顾绮言'),
    Request('陪许昀芝挑一种她没试过的饮料。', '许昀芝'),
    Request('把“何青玟要带什么”译一下。', '何青玟'),
    Request('赵茗初今天心情不好，写一句安慰她的话。', '赵茗初'),
    Request('帮岑婉听挑另一个碰头地点，之前定的先放一边。', '岑婉听'),
    Request('怎么把顾绮言家那本潮掉的书晾干？', '顾绮言'),
    Request('我还没选好，不用看黎书然登记的饮品，先说说红茶怎么冲。', '黎书然'),
    Request('给石小满发一句，提醒他带之前记下的用品。', None),
    Request('请告诉我黎书然和顾绮言两个人各借了哪一本。', None),
    Request('帮我解释这句话怎么加标点。', None),
)
