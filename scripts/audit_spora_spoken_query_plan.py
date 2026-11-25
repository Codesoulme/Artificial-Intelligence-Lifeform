"""Train a shared read/clarify/chat and version-address probe on frozen SPORA features."""

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import re
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_natural_lookup_cli import PAIRS, RECORDS, digest
from scripts.spora_version_chat import (FIELDS, VersionReplyActions, VersionStore, artifacts, base,
                                        compress_dimension, local_model, reply_to_route)
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_multi_version_route import resolve_address
from spora.research_version_store import MemoryAddress
from spora.research_version_timeline import VersionQuery


FIELDS_ORDER = tuple(FIELDS)
VERSIONS = (VersionQuery('current'), VersionQuery('previous'), VersionQuery('ordinal', 1))
CHAT, READ, ASK = range(3)
UNLABELED = -100
SEED = 60811
DEPTH = 12

TRAIN_NAMES = ('乔念双', '夏南明', '苏枫禾', '齐玥春', '栗桑宁', '钟檀知')
CAL_NAMES = ('唐舒微', '莫明秋', '温岚羽', '柯辰弦')
TRAIN_NOUNS = (('点的那杯', '买的饮料'), ('碰面的地址', '约见的位置'),
               ('带来的东西', '准备的那件物品'), ('挑的读物', '借阅的那册'))
CAL_NOUNS = ('喝的那种', '接头处', '要捎的那样', '选的书')
TRAIN_TIMES = (('现在', '刚换好之后'), ('换掉以前', '上一个版本里'), ('最早那次', '头回选的时候'))
CAL_TIMES = ('后来定下的', '上一回', '开头说的')
TRAIN_FRAMES = (
    ('{p}{v}说的{f}是什么？报一下。', '帮我翻译这句话：{p}{v}说的{f}是什么？',
     '{p}{v}说的{f}是什么？算了，这件事先不核对。'),
    ('{q}先放在一边，{p}{v}的{f}我想知道。',
     '把「{q}先放在一边，{p}{v}的{f}我想知道」改得客气些。',
     '{q}先放在一边，{p}{v}的{f}也先不看了。'),
    ('{p}这边别查，把{q}{v}的{f}告诉我。', '照抄原句，不用回答：{p}这边别查，把{q}{v}的{f}告诉我。',
     '{p}和{q}{v}的{f}都不看了，我要另定。'),
    ('{p}{v}的{f}是哪样？只问这个，{q}那件我另外确认。',
     '给这句加上标点：{p}{v}的{f}是哪样只问这个{q}那件我另外确认',
     '{p}{v}的{f}是哪样？先撤回这一问，{q}那件我也另外确认。'),
)
CAL_FRAMES = (
    ('{p}先搁着，我想弄清{q}{v}{f}究竟是哪一个。',
     '请用英文写出原话，不用查：{p}先搁着，我想弄清{q}{v}{f}究竟是哪一个。',
     '{p}先搁着，{q}{v}{f}也留到改天再问。'),
    ('给我说说{p}{v}{f}的名称。{q}那边我自己问。',
     '帮我改写原话：给我说说{p}{v}{f}的名称。{q}那边我自己问。',
     '先不问{p}{v}{f}的名称了。{q}那边我自己问。'),
)
ASK_TRAIN = (
    '之前说定的{f}，你能说一下是哪一个吗？我没写人名。',
    '我想核对{p}原先的安排，先帮我确认该问哪一项。',
    '{p}那份{f}是什么？我不知道要用现在的还是旧的。',
)
ASK_CAL = (
    '有人订过{f}，我把人忘了。能帮我找一下吗？',
    '{p}的事有一项我想弄清，具体问什么我还得想想。',
    '给我看{p}定的{f}吧，首个还是最后那个我还没想好。',
)
HELD_ASK = (
    '我忘了是谁说要喝那种了，能帮我核实一下吗？',
    '陶依晴那次定过一个东西，具体是哪一项我也不确定，怎么查？',
    '许雁书选的那本能核对一下吗？我还没弄清是要初选那本还是后来改选的。',
)
HELD_CHAT = ('推荐一本适合在火车上消磨时间的短篇小说。', '周六打算出门，可能下雨，我该带什么？')


def request_rows(names, nouns, times, frames, asks, *, other_owner):
    for frame_id, (positive, rewrite, cancel) in enumerate(frames):
        for index, p in enumerate(names):
            q = names[(index + 1) % len(names)]
            for field, aliases in enumerate(nouns):
                for version, phrases in enumerate(times):
                    args = dict(p=p, q=q, f=aliases[index % len(aliases)], v=phrases[index % len(phrases)])
                    request = positive.format(**args)
                    _, mentioned = anonymize(request, (p, q))
                    owner = q if frame_id in other_owner else p
                    yield request, (READ, mentioned.index(owner), field, version)
                    yield rewrite.format(**args), (CHAT, UNLABELED, UNLABELED, UNLABELED)
                    yield cancel.format(**args), (CHAT, UNLABELED, UNLABELED, UNLABELED)
    for index, p in enumerate(names):
        for aliases in nouns:
            for frame in asks:
                yield frame.format(p=p, f=aliases[index % len(aliases)]), (ASK, UNLABELED, UNLABELED, UNLABELED)


def anonymize(question, names):
    pattern = '|'.join(re.escape(name) for name in sorted(names, key=lambda name: (-len(name), name)))
    found = tuple(dict.fromkeys(re.findall(pattern, question))) if pattern else ()
    aliases = dict(zip(found, '甲乙'))
    if len(found) > 2:
        raise ValueError('The probe supports up to two named candidates')
    masked = re.sub(pattern, lambda match: aliases[match[0]], question) if found else question
    return f'{masked}\n接下来要做什么？', found


def vectors(extract, requests, names, features='last'):
    prompts, owners = zip(*(anonymize(question, names) for question in requests), strict=True)
    result = []
    for begin in range(0, len(prompts), 24):
        batch = prompts[begin:begin + 24]
        hidden = extract(batch)
        if features == 'user-span':
            rows = [hidden[row, :text.rfind('\n')].float().mean(dim=0) for row, text in enumerate(batch)]
        else:
            rows = [hidden[row, len(text) - 1].float() for row, text in enumerate(batch)]
        result.append(torch.stack(rows))
    return torch.cat(result), owners


def distinct_rows(rows, names):
    by_prompt = {}
    for question, labels in rows:
        normalized, _ = anonymize(question, names)
        if normalized in by_prompt and by_prompt[normalized][1] != labels:
            raise ValueError('The same normalized question has conflicting labels')
        by_prompt.setdefault(normalized, (question, labels))
    return tuple(by_prompt.values())


class QueryPlanProbe(nn.Module):
    classes = (3, 2, len(FIELDS_ORDER), len(VERSIONS))

    def __init__(self, width):
        super().__init__()
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 96), nn.GELU(), nn.Linear(96, sum(self.classes)))

    def forward(self, hidden):
        return self.head(hidden).split(self.classes, dim=-1)


def fit(features, targets):
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)
    probe = QueryPlanProbe(features.shape[-1])
    optimizer = torch.optim.AdamW(probe.parameters(), lr=.002)
    for _ in range(12):
        for index in torch.randperm(len(targets), generator=rng).split(48):
            all_logits = probe(features[index])
            labels = targets[index]
            loss = sum(F.cross_entropy(logits, labels[:, part], ignore_index=UNLABELED, reduction='sum')
                       / labels[:, part].ne(UNLABELED).sum().clamp_min(1)
                       for part, logits in enumerate(all_logits))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return probe.eval()


def predict(probe, features):
    with torch.inference_mode():
        maxima = [logits.softmax(-1).max(-1) for logits in probe(features)]
    return list(zip(*(list(zip(part.indices.tolist(), part.values.tolist(), strict=True)) for part in maxima), strict=True))


def planned(scored, owners, threshold):
    (action, _), (owner, _), (field, _), (version, _) = scored
    if action == READ and (owner >= len(owners) or min(score for _, score in scored) < threshold):
        action = ASK
    return (action, owner, field, version) if action == READ else (action, UNLABELED, UNLABELED, UNLABELED)


def pick_threshold(predictions, owners, labels):
    def quality(threshold):
        decisions = [planned(row, names, threshold) for row, names in zip(predictions, owners, strict=True)]
        mismatch = sum(row[0] == READ and row != target for row, target in zip(decisions, labels, strict=True))
        reads = sum(row[0] == READ and row == target for row, target in zip(decisions, labels, strict=True))
        return -mismatch, reads, -threshold
    return max((0., .55, .7, .8, .9, .95, 1.), key=quality)


@torch.inference_mode()
def evaluate(model, tokenizer, store, reader, extract, probe, threshold, representation, *,
             requests=None, records=RECORDS):
    catalog = store.catalog()
    names = {address.name for address in catalog}
    if requests is None:
        requests = ([(natural, READ, number) for number, _, natural, _ in PAIRS]
                    + [(other, CHAT, None) for _, _, _, other in PAIRS]
                    + [(text, ASK, None) for text in HELD_ASK] + [(text, CHAT, None) for text in HELD_CHAT])
    features, owners = vectors(extract, [text for text, _, _ in requests], names, representation)
    embedding = model.get_input_embeddings().weight
    codebook = F.normalize(embedding.index_select(1, compress_dimension(embedding.shape[1]).to(embedding.device)).float(), dim=-1)
    known_values = [row[-1] for row in records]
    rows, totals = [], {mode: Counter() for mode in ('raw', 'calibrated')}
    cached = {}
    for (text, action, number), scored, candidates in zip(requests, predict(probe, features), owners, strict=True):
        if any(term in text for term in known_values):
            raise ValueError('A held request contains a stored value')
        arms = {}
        for mode, cut in (('raw', 0.), ('calibrated', threshold)):
            proposed, owner, field, version = planned(scored, candidates, cut)
            record, output = None, None
            chosen_field = FIELDS_ORDER[field] if proposed == READ else None
            if chosen_field is not None:
                address = MemoryAddress(candidates[owner], chosen_field)
                record, _ = resolve_address(catalog, address, VERSIONS[version], date.today(), retain_same_day=True)
                if record is not None:
                    key = (text, record, chosen_field)
                    if key not in cached:
                        try:
                            _, result = reply_to_route(model, tokenizer, store, codebook, reader, text, [record],
                                                       labels=[FIELDS[chosen_field]])
                        except ValueError as error:
                            result = {'error': str(error)}
                        cached[key] = result
                    output = cached[key]
            expected_value = None if number is None else known_values[number - 1]
            shown = [term for term in known_values if output and term in output.get('answer', '')]
            correct = record is not None and record == number and shown == [expected_value]
            totals[mode].update(action_match=proposed == action, exact_read=correct,
                                wrong_read=record is not None and record != number, asks=proposed == ASK,
                                unnecessary_plan=proposed == READ and action != READ)
            arms[mode] = dict(action=proposed, owner=None if owner == UNLABELED else candidates[owner],
                              field=chosen_field, version=None if version == UNLABELED else version,
                              record=record, output=output, exact_read=correct)
        rows.append(dict(question=text, candidates=candidates, expected_action=action, expected_record=number,
                         scored=scored, arms=arms))
    return rows, totals


def main(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    reader_path, _, _, binding = artifacts(args.reader, args.normalizer)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    prior = json.loads((args.prior / 'result.json').read_text(encoding='utf-8'))
    archive = args.prior / 'recurrent' / 'versions.sqlite3'
    sealed = {p: digest(p) for p in (archive, Path(__file__), reader_path, args.recurrent[0],
                                    *(directory / 'attention.pt' for directory in args.recurrent[1:]))}
    if sealed[archive] != prior['recurrent']['archive_sha256']:
        raise ValueError('Expected the previous isolated recurrent test profile')
    train = distinct_rows(request_rows(TRAIN_NAMES, TRAIN_NOUNS, TRAIN_TIMES, TRAIN_FRAMES, ASK_TRAIN,
                                       other_owner={2}), TRAIN_NAMES)
    cal = distinct_rows(request_rows(CAL_NAMES, [(x,) for x in CAL_NOUNS], [(x,) for x in CAL_TIMES], CAL_FRAMES,
                                     ASK_CAL, other_owner={0}), CAL_NAMES)
    if {anonymize(q, TRAIN_NAMES)[0] for q, _ in train} & {anonymize(q, CAL_NAMES)[0] for q, _ in cal}:
        raise ValueError('Training and calibration overlap')
    with local_model(args.recurrent) as model, VersionStore(archive, binding) as store:
        extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=DEPTH)
        train_x, train_owners = vectors(extract, [q for q, _ in train], TRAIN_NAMES, args.features)
        if any(target[0] == READ and target[1] >= len(names) for (_, target), names in zip(train, train_owners, strict=True)):
            raise ValueError('A supervised owner could not be located in its sentence')
        probe = fit(train_x, torch.tensor([target for _, target in train]))
        cal_x, cal_owners = vectors(extract, [q for q, _ in cal], CAL_NAMES, args.features)
        cal_predictions = predict(probe, cal_x)
        labels = [target for _, target in cal]
        threshold = pick_threshold(cal_predictions, cal_owners, labels)
        cal_rows = [dict(question=q, gold=target, owners=names, scored=scored, planned=planned(scored, names, threshold))
                    for (q, target), names, scored in zip(cal, cal_owners, cal_predictions, strict=True)]
        state = torch.load(reader_path, map_location='cpu', weights_only=True)
        reader = VersionReplyActions(model.config.hidden_size, state['scaffold_tokens'], slots=2)
        reader.load_state_dict(state)
        rows, metrics = evaluate(model, tokenizer, store, reader.eval().to(model.device), extract, probe, threshold,
                                 args.features)
    if any(digest(path) != expected for path, expected in sealed.items()):
        raise ValueError('The audit changed a sealed artifact')
    checkpoint = args.output / 'probe.pt'
    atomic_save(probe.state_dict(), checkpoint)
    store_json(args.output / 'result.json', {
        'rows': rows, 'metrics': metrics, 'calibration': cal_rows, 'training': len(train), 'depth': DEPTH,
        'features': args.features,
        'actions': ('chat', 'read', 'ask'), 'threshold': threshold, 'probe_sha256': digest(checkpoint),
        'sources': {str(path): value for path, value in sealed.items()}, 'seconds': perf_counter() - started,
    })
    print(metrics, 'calibration threshold', threshold, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reader', type=Path, default=Path('runs/spora-dual-version-reply-v1'))
    parser.add_argument('--normalizer', type=Path, default=Path('runs/spora-version-normalizer-v1'))
    parser.add_argument('--features', default='last', choices=('last', 'user-span'))
    parser.add_argument('--recurrent', required=True, nargs=3, type=Path)
    main(parser.parse_args())
