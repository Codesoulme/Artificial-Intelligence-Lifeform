"""Independent field requests, confusable uses and version-preservation checks."""

from scripts.spora_composition_address_cases import Request


FRESH = (
    Request('帮赵茗初把要随身带的那样东西写清楚，我要用双方讲妥的那一件。', '赵茗初', ('items',)),
    Request('我给赵茗初寄个便条，写上碰头是在哪儿、要捎什么，按原定计划。', '赵茗初', ('places', 'items')),
    Request('想把和何青玟约定的那个碰面的点发到群里，给我原先定下的地方。', '何青玟', ('places',)),
    Request('给何青玟说一下随身拿什么。地点在原话里已经写好，只需要提那件东西。', '何青玟', ('items',)),
    Request('许昀芝平常点的那杯给我留一份，要用她已经记下的口味。', '许昀芝', ('drinks',)),
    Request('给许昀芝那杯照旧，我还要在留言里写出她取的那册的标题，两项都按已有登记。', '许昀芝', ('drinks', 'books')),
    Request('我忘记顾绮言要带来一起读的那本叫什么了，帮我核一下记录。', '顾绮言', ('books',)),
    Request('麻烦把顾绮言和我碰头的旧约定填到消息里，这次别提他会带哪册。', '顾绮言', ('places',)),
    Request('岑婉听的碰头点照说好的，请在消息里也提醒她捎上原先定的那件。', '岑婉听', ('places', 'items')),
    Request('黎书然那册留的读物到底什么名字？照她的记录说，喝的先不问。', '黎书然', ('books',)),
    Request('我想给黎书然备她惯常喝的那一份，可以把档案里那个名字告诉我吗？', '黎书然', ('drinks',)),
    Request('黎书然问今晚喝什么，你帮我想一种她从没试过的。', '黎书然'),
    Request('给许昀芝出个主意，送一本新书，不考虑已借过的。', '许昀芝'),
    Request('翻译一下“赵茗初约好带什么”，这句只是翻译题。', '赵茗初'),
    Request('给岑婉听写句生日祝福，她最近挺忙的。', '岑婉听'),
    Request('何青玟说地点不用查。说说见面后怎么自然地开口打招呼。', '何青玟'),
    Request('先别用顾绮言登记的书，帮我看看普通借书流程。', '顾绮言'),
    Request('我要改约赵茗初，地点另挑一个，给两条挑地点的建议。', '赵茗初'),
    Request('麻烦给岑素明报一下原来准备带的那件东西。', None),
    Request('许昀芝和黎书然各自常喝的那份是什么？', None),
)


# Hold explicit history reads alongside natural current-record composition.
EXPLICIT = (
    ('请查赵茗初的用品首版。', '赵茗初', 'items', 0),
    ('请查顾绮言的书目上一版。', '顾绮言', 'books', 0),
    ('请查何青玟的见面地点当前版。', '何青玟', 'places', 1),
    ('请查黎书然的饮品当前版。', '黎书然', 'drinks', 1),
    ('请查黎书然的饮品首版还是现行版？', None, None, None),
    ('“请查顾绮言的书目上一版”请译成英文。', None, None, None),
    ('请查余景川的书目首版。', None, None, None),
)
