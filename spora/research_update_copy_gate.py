"""A typed, bounded value-copy gate for writer diagnostics."""

import re


_PATTERNS = {
    'day': re.compile(r'(?:\u5468|\u661f\u671f)[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u65e5\u5929]'),
    'time': re.compile(
        r'(?:\u4e0a\u5348|\u4e0b\u5348|\u665a\u4e0a|\u4e2d\u5348|\u65e9\u4e0a|\u6e05\u6668|\u508d\u665a|\u5348\u540e)'
        r'[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u96f6\u4e24]+\u70b9(?:\u534a)?'
    ),
    'quantity': re.compile(
        r'[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u96f6\u4e24]+'
        r'[\u5957\u7bb1\u5377\u518c\u4efd\u888b\u5305\u76d2\u672c\u5f20\u652f]'
    ),
    'direction': re.compile(r'(?:\u5411|\u5f80|\u671d)[\u4e1c\u5357\u897f\u5317]+'),
}

_FIELD_CUES = {
    'place': r'\u5730\u70b9|\u573a\u5730|\u5730\u5740|\u96c6\u5408(?:\u70b9|\u5904)?|\u78b0\u5934\u70b9|\u51fa\u53d1\u5730',
    'gear': r'\u88c5\u5907|\u51c6\u5907\u54c1|\u643a\u5e26(?:\u7684\u4e1c\u897f)?|\u7269\u54c1|\u4e1c\u897f|\u5907\u4e0a|\u5e26\u4e0a|\u6539\u5e26',
    'person': r'\u6536\u4ef6\u4eba|\u6536\u8d27\u4eba|\u59d3\u540d|\u4eba\u540d|\u540d\u5355|\u7b7e\u6536|\u4ea4\u7ed9|\u8ba9',
}
_CHANGE = r'\u6362\u6210|\u66f4\u65b0\u4e3a|\u53d8\u66f4\u6210|\u6539\u4e3a|\u6539\u6210|\u8c03\u6574\u4e3a|\u8c03\u81f3|\u6539\u5230|\u6362\u5230|\u6539\u5728|\u662f|\u4e3a'
_VALUE = r'[^\uff0c\u3002\uff1b,.;!?！？\s]+'
_PLACE_DIRECT = re.compile(
    r'(?:\u5728|\u5230|\u5f80|\u53bb|\u4ece|\u9001\u5230|\u5bc4\u5f80)'
    r'([^\uff0c\u3002\uff1b,.;!?！？\s]{2,14}?)'
    r'(?=(?:\u7684|\u4e3e\u884c|\u529e|\u53c2\u52a0|\u78b0\u9762|\u96c6\u5408|\u7b7e\u5230|\u89c1\u9762|\u9001|\u5bc4|\u53d1|\u7b49|\u4f1a\u5408|\u524d\u8fdb|\u4e4b\u540e|$))'
)
_PLACE_PREFIX = re.compile(r'^([^\uff0c\u3002\uff1b,.;!?！？]{4,14}?)(?=\u7684)')


class TypedValueCopyGate:
    """Overlay only complete typed spans that are explicitly present in a turn."""

    def __init__(self, fallback):
        self.fallback = fallback
        self.state = {}
        self.calls = 0

    def reset(self):
        self.state = {}
        self.calls = 0

    def __call__(self, text, margin=2.0):
        result = dict(self.fallback(text, margin))
        if self.calls < 2:
            for field, pattern in _PATTERNS.items():
                matches = list(pattern.finditer(text))
                if matches:
                    result[field] = matches[-1].group(0)
            for field, cue in _FIELD_CUES.items():
                pattern = re.compile(r'(?:' + cue + r').{0,12}?(?:' + _CHANGE + r')(' + _VALUE + r')')
                matches = list(pattern.finditer(text))
                if matches:
                    result[field] = matches[-1].group(1)
                    if field == 'person':
                        result[field] = re.sub(r'(?:\u7b7e\u5b57|\u7b7e\u6536)$', '', result[field])
                else:
                    for old in (self.state.get(field),):
                        if old and old in text:
                            marker = re.compile(r'(?:' + _CHANGE + r')(' + _VALUE + r')')
                            nearby = [match for match in marker.finditer(text) if match.start() >= text.find(old)]
                            if nearby:
                                result[field] = nearby[-1].group(1)
            place_matches = list(_PLACE_DIRECT.finditer(text))
            if place_matches:
                result['place'] = place_matches[-1].group(1)
            else:
                prefix = _PLACE_PREFIX.search(text)
                if prefix:
                    result['place'] = prefix.group(1)
            self.state.update(result)
        self.calls += 1
        return result
