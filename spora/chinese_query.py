"""Learned finite Chinese question addressing; answers live in neural memory.

Digit composition, finite address dispatch and response formatting are explicit
priors. No regex or keyword parser produces query labels at inference. This is
not an open-domain language generator or learned natural-language writing path.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


KINDS = ('current', 'history', 'source', 'reject')


def consistent_address(text, positions, digits, kind):
    """Validate neural pointers; never extract or correct an alternative address."""
    text=text.strip()
    groups=((positions[:2],digits[:2]),) if kind==0 else ((positions[:2],digits[:2]),(positions[2:],digits[2:]))
    used=set()
    for points,numbers in groups:
        active=[p for p in points if p>0]
        if not active or points != [0]*(len(points)-len(active))+active:
            return False
        if active != list(range(active[0],active[0]+len(active))) or used.intersection(active):
            return False
        used.update(active)
        for position,digit in zip(points,numbers):
            if position==0:
                if digit!=0:return False
            elif position>len(text) or text[position-1] not in '0123456789' or int(text[position-1])!=digit:
                return False
        start,end=active[0]-1,active[-1]
        if (start>0 and text[start-1] in '0123456789') or (end<len(text) and text[end] in '0123456789'):
            return False
    return True


@dataclass
class ChineseQueryConfig:
    characters: str
    width: int = 64
    layers: int = 2
    heads: int = 4
    context: int = 96
    confidence: float = .90

    def validate(self):
        if (not self.characters or len(set(self.characters)) != len(self.characters)
                or any(type(v) is not int or v <= 0 for v in (self.width, self.layers, self.heads, self.context))
                or self.width % self.heads or self.context > 256 or not .9 <= self.confidence < 1):
            raise ValueError('Invalid finite question configuration')


class ChineseQuery(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        self.character_to_token = {c:i+2 for i,c in enumerate(config.characters)}
        self.embedding = nn.Embedding(len(config.characters)+2, config.width, padding_idx=0)
        self.position = nn.Embedding(config.context, config.width)
        layer = nn.TransformerEncoderLayer(config.width, config.heads, config.width*2,
            dropout=0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.width)
        self.slots = nn.Parameter(torch.randn(6, config.width)*.1)
        self.intent = nn.Linear(config.width, 4)
        self.digit = nn.Linear(config.width, 10)

    def config_dict(self):
        return asdict(self.config)

    def encode(self, texts):
        rows = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError('Nonempty question required')
            text = text.strip()
            if len(text)+1 > self.config.context or any(c not in self.character_to_token for c in text):
                raise ValueError('Question outside supported character/length domain')
            rows.append([1]+[self.character_to_token[c] for c in text])
        if not rows:
            raise ValueError('Nonempty batch required')
        tokens = torch.zeros(len(rows), max(map(len,rows)), dtype=torch.long,
                             device=self.embedding.weight.device)
        for i,row in enumerate(rows): tokens[i,:len(row)] = torch.tensor(row, device=tokens.device)
        return tokens

    def forward(self, tokens):
        if tokens.ndim != 2 or tokens.dtype != torch.long or not 1 <= tokens.shape[1] <= self.config.context:
            raise ValueError('Finite integer token batch required')
        padding = tokens == 0
        x = self.embedding(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))
        x = self.norm(self.encoder(x, src_key_padding_mask=padding))
        attention = torch.einsum('bnd,sd->bsn', x, self.slots)/math.sqrt(self.config.width)
        attention = attention.masked_fill(padding[:,None], -torch.inf)
        selected = torch.einsum('bsn,bnd->bsd', attention.softmax(-1), x)
        return dict(intent=self.intent(x[:,0]), digits=self.digit(selected), attention=attention)

    @torch.no_grad()
    def parse(self, texts):
        output = self(self.encode(texts))
        intent_prob, kinds = output['intent'].softmax(-1).max(-1)
        digit_prob, digits = output['digits'].softmax(-1).max(-1)
        positions=output['attention'].argmax(-1).tolist()
        results = []
        for i,kind in enumerate(kinds.tolist()):
            values = digits[i].tolist()
            entity = values[0]*10+values[1]
            event = sum(d*10**(3-j) for j,d in enumerate(values[2:]))
            confidence = float(intent_prob[i])
            if kind != 3:
                confidence = min(confidence, float(digit_prob[i, :2 if kind == 0 else 6].min()))
            consistent = kind != 3 and consistent_address(texts[i],positions[i],values,kind)
            accepted = consistent and confidence >= self.config.confidence and entity < 64 and (kind == 0 or event < 1024)
            results.append(dict(kind=KINDS[kind], entity=entity if accepted else None,
                event=event if accepted and kind in (1,2) else None, confidence=confidence,
                accepted=accepted, pointer_consistent=consistent))
        return results


def attach_query(model, module):
    if hasattr(model, 'chinese_query'):
        raise ValueError('Question module already attached')
    if model.config.key_count != 64 or model.config.unknown != 16:
        raise ValueError('Question module requires the finite 64-key, 16-value individual')
    module.config.validate()
    if not all(bool(torch.isfinite(t).all()) for t in module.state_dict().values()):
        raise ValueError('Nonfinite question weights')
    model.chinese_query = module.to(next(model.parameters()).device).eval()
    return model


@torch.no_grad()
def answer_memory(model, state, question):
    if not hasattr(model, 'chinese_query'):
        raise ValueError('Individual has no trained Chinese query module')
    try:
        parsed = model.chinese_query.parse([question])[0]
    except ValueError as exc:
        return dict(status='rejected', text='目前只支持明确编号的有限记忆查询。', reason=str(exc))
    if not parsed['accepted']:
        return dict(status='rejected', parsed=parsed, text='没有可靠识别这条查询，请明确对象编号和查询类型。')
    kind, key, event = parsed['kind'], parsed['entity'], parsed['event']
    base = dict(parsed=parsed, value=None, source=None)
    if kind != 'current' and event >= state.cursor:
        return dict(base, status='unknown', text='未知：这个事件序号还没有记录。')
    if kind == 'source':
        from .internal_memory import read_provenance
        provenance = read_provenance(model, state, key, event) if state.internal is not None else {'source':None}
        source = provenance['source']
        names = {'direct':'直接观察', 'report':'转述', 'inference':'推测', 'simulation':'模拟'}
        return dict(base, status='known' if source else 'unknown', source=source, provenance=provenance,
                    text=names[source]+'。' if source else '未知：没有可靠的来源记录。')
    from .growing import query
    device = next(model.parameters()).device
    key_tensor, mode_tensor, time_tensor = [torch.tensor([v], device=device) for v in
        (key, int(kind=='history'), 0 if kind=='current' else event)]
    value = int(query(model, state, key_tensor, mode_tensor, time_tensor).argmax(-1))
    if value == model.config.unknown:
        return dict(base, status='unknown', text='未知：没有读到可靠的对应记忆。')
    return dict(base, status='known', value=value,
                text=f'对象{key}的当前值是{value}。' if kind=='current' else f'事件{event}中，对象{key}的记录值是{value}。')
