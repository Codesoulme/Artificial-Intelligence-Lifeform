"""Detect explicit text operations whose quoted content is data."""

import re


_QUOTED = re.compile(r'“[^”\n]+”|「[^」\n]+」|『[^』\n]+』|"[^"\n]+"')
_EDIT = re.compile(r'(?:^|[。！？；，,.!?;])\s*(?:请|帮我|麻烦你|只|仅|先)*'
                   r'(?:(?:改写|润色|翻译|转述)|(?:评价|评议|检查|判断)\s*(?:这句话?|这个问法|表达|措辞|语气|句子)|'
                   r'把\s*(?:换一种说法|换个说法)|这(?:句话?|个问法)(?:读起来)?(?:是否自然|是否通顺))')
_READ_AS_WELL = re.compile(r'(?:^|[。！？；，,.!?;]|再|然后|接着|最后|另外|同时|并且|还要|还请|并|也)\s*'
                          r'(?:请|帮我|实际|顺便)*(?:回答|作答|查询|查阅|查一下|读取|直接答|想问|想知道)')
_TOPIC_SWITCH = re.compile(r'^\s*(?:我们|先|再)?(?:换个话题|另起一个话题|说点别的)(?:[。！？；，,.!?;\s]|$)')


def treats_quotes_as_text(question):
    """Constrain the shortcut to quoted text and an outer editing request."""
    outer, quotes = _QUOTED.subn(' ', question)
    return bool(quotes and _EDIT.search(outer) and not _READ_AS_WELL.search(outer))


def starts_new_topic(question):
    return bool(_TOPIC_SWITCH.search(question))
