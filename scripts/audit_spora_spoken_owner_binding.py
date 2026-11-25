"""Learn whether a mentioned person is the actual target of a spoken memory request."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from scripts import audit_spora_coordinate_fusion as fusion
from scripts import audit_spora_dense_joint_address as dense
from scripts import audit_spora_read_confirmation_action as action
from scripts import audit_spora_spoken_coordinate_clarification as coordinate
from scripts import audit_spora_spoken_query_plan as plan
from scripts.audit_spora_spoken_coordinate_clarification import ADDRESS, ASK_FIELD, ASK_OWNER, CHAT
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_version_dialogue_scope import PartialLookup
from spora.research_version_store import MemoryAddress


FIRST, SECOND, MISSING, WITHDRAWN = range(4)
TRAIN = (
    ('{a}和{b}都提到了。', '我得知道{p}{x}。', '我得知道另一人的{x}，名字我补。', '两人的{x}我会自行核对。'),
    ('先提{a}，后提{b}。', '这次想听{p}{x}，照已定的说。', '有个没提到的人，{x}我想核实。', '这两项先不看了。'),
    ('{a}和{b}的事分开。', '请你说说{p}{x}，我这边缺的正是这个。', '本次要查的人没说，先给你线索：{x}。', '这是我写给自己的提示，暂不需要你找。'),
    ('先前说的是{a}和{b}。', '说回{p}，{x}我现在要确认一下。', '说回另一位，{x}还要弄清，姓名我等会儿报。', '这几句只是举例，帮我拿掉问号。'),
    ('我和{a}、{b}有不同安排。', '打算按旧约办，{p}{x}？', '打算照旧办理，{x}？具体跟谁我先想想。', '先前安排留着，下回我再核对。'),
    ('{a}的安排和{b}别弄混。', '只想问{p}{x}，另外一位留着。', '这两位都解决了。还要问{x}，我先不报第三位。', '两位都解决了，接下来请另拟一条提醒。'),
    ('有个名单，上面有{a}和{b}。', '先帮我确认{p}{x}，照写下来的。', '我想确认{x}，与名单上两位无关，等下说对象。', '我稍后去问，不需要帮我确认。'),
    ('关于{a}和{b}，我思路有点乱。', '定下来的先给我报一下，{p}{x}？', '定下来的先核实一下，{x}？当事人我稍后告诉你。', '先放一下两边的事，今天只想润色文字。'),
    ('{a}的已办妥。', '不过还要核对{a}{x}，我怕弄混。', '不过另有一人，{x}？人名我等会给。', '办妥就好，暂时没有要核对的。'),
    ('{a}的我已经问过。', '为防出错，还是把{a}{x}报一遍。', '我还想问别人{x}。等我告诉你是谁。', '我会自己确认剩下的，不劳你找。'),
)
CAL = (
    ('提过{a}，也提过{b}。', '回到{p}这边，旧安排里{x}？', '还有一个没报姓名的人，想核实{x}。', '我有渠道弄清这些，先不用翻。'),
    ('先说{a}的进展已经清楚了。', '补充问一下{a}{x}，照旧的告诉我。', '剩下的是别人的。{x}我想知道，人名稍后给。', '剩下的以后自己再问。'),
)
FRESH = (
    (ADDRESS, 0, 'items', '夏宁瑶和夏宁遥的都到了。夏宁瑶让我收的是哪件来着？我核实下。'),
    (ADDRESS, 1, 'items', '夏宁瑶和夏宁遥的都到了。夏宁遥让我收的是哪件来着？我核实下。'),
    (ASK_OWNER, 2, 'items', '夏宁瑶和夏宁遥的都到了。还有一件前面说好我要收，先说是什么，我再报第三个人。'),
    (CHAT, None, None, '夏宁瑶和夏宁遥的都到了。我自己会核实剩下几件，你先别帮我找。'),
    (ADDRESS, 2, 'places', '郭回霜郭回双今天各有安排。先回想跟郭回霜商量在哪儿会面。'),
    (ADDRESS, 3, 'places', '郭回霜郭回双今天各有安排。先回想跟郭回双商量在哪儿会面。'),
    (ASK_OWNER, 1, 'places', '郭回霜郭回双今天各有安排。现在问的是跟第三个人说好在哪儿会面，名字我待会再给。'),
    (CHAT, None, None, '郭回霜郭回双今天各有安排。两个我都去打电话，你帮我想句问候。'),
    (ADDRESS, 1, 'drinks', '夏宁遥喝什么我还要确认下，尽管我已经把她那杯买好了。'),
    (ASK_OWNER, 0, 'drinks', '夏宁遥的那杯我已拿到。还有人让我带一杯，原先说带哪种？是谁我下一句说。'),
    (CHAT, None, None, '夏宁遥的那杯我已拿到。还有人也想要，我会自己问。'),
    (ADDRESS, 3, 'books', '郭回双先借的本来就还了。但第一次她借了哪本，我需要核对。'),
    (ASK_OWNER, 2, 'books', '郭回双的书我已经还。另一个人先借的那本是什么？姓名还没告诉你。'),
    (CHAT, None, None, '郭回双的书我已经还。接下来写一句话谢谢她。'),
    (ASK_FIELD, 0, 'books', '夏宁瑶先前留的那项我想翻出来。等我说是哪一类再找。'),
    (ASK_FIELD, 2, 'drinks', '我说郭回霜的资料，可能得调出来看一项。要哪项我还没讲。'),
)


def examples(frames, people, aliases):
    groups = []
    for index, (opening, request, missing, withdrawn) in enumerate(frames):
        for family in aliases:
            for item in family:
                a, b = people
                prefix = opening.format(a=a, b=b)
                owners = (b, a) if index % 2 else (a, b)
                if b not in prefix:
                    owners = (a,)
                group = [(prefix + request.format(p=owner, a=a, b=b, x=item), (a, b).index(owner)) for owner in owners]
                group += [(prefix + missing.format(x=item), MISSING), (prefix + withdrawn.format(x=item), WITHDRAWN)]
                groups.append(group)
    return groups


def features(extract, texts, people):
    states, spans, names = dense.align(extract, texts, people)
    mask = torch.tensor([[i < len(row) for i in range(2)] + [True, True] for row in names])
    return (states, spans, mask), names


class OwnerHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.global_context = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU())
        self.local_context = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU())
        self.named = nn.Linear(64, 1)
        self.unbound = nn.Linear(64, 2)

    def forward(self, state, spans, mask):
        context = self.global_context(state)
        named = self.named(F.gelu(context[:, None] + self.local_context(spans))).squeeze(-1)
        return torch.cat([named, self.unbound(context)], dim=-1).masked_fill(~mask, -torch.inf)


def fit(inputs, labels, pairs, *, contrast):
    torch.manual_seed(action.SEED)
    generator = torch.Generator().manual_seed(action.SEED)
    head = OwnerHead(inputs[0].shape[-1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    for _ in range(16):
        for batch in torch.randperm(len(labels), generator=generator).split(24):
            logits = head(*(item[batch] for item in inputs))
            loss = F.cross_entropy(logits, labels[batch])
            if contrast:
                left, right = pairs[batch].unbind(1)
                pos = head(*(item[left] for item in inputs))
                neg = head(*(item[right] for item in inputs))
                target = labels[left, None]
                change = pos.gather(1, target) - neg.gather(1, target)
                loss = loss + .25 * F.softplus(1 - change).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return head.eval()


def flatten(groups):
    rows, pairs = [], []
    for group in groups:
        start = len(rows)
        rows.extend(group)
        for index, (_, target) in enumerate(group):
            rival = len(group) - 2 if target < MISSING else 0
            pairs.append((start + index, start + rival))
    return rows, torch.tensor(pairs)


@torch.inference_mode()
def predict(head, inputs):
    confidence, labels = head(*inputs).softmax(-1).max(-1)
    return list(zip(labels.tolist(), confidence.tolist(), strict=True))


def calibration_threshold(predictions, examples):
    def quality(limit):
        correct = sum(label == gold and score >= limit
                      for (label, score), (_, gold) in zip(predictions, examples, strict=True))
        wrong = sum(label != gold and score >= limit
                    for (label, score), (_, gold) in zip(predictions, examples, strict=True))
        return correct - 2 * wrong, -wrong, -limit
    return max((.4, .5, .6, .7, .8, .9, .95, .99, 1.000001), key=quality)


def rewrite(proposal, decision, enrolled, catalog):
    if proposal is None or decision == WITHDRAWN:
        return None
    field = proposal.field
    if decision == MISSING:
        return PartialLookup(field=field) if field is not None else None
    owner = enrolled[decision]
    if field is None:
        return PartialLookup(owner=owner)
    full = MemoryAddress(owner, field)
    return full if full in catalog else None


def outcome(cases, names, predictions, threshold):
    total = Counter()
    for (kind, person, _, _), enrolled, (label, score) in zip(cases, names, predictions, strict=True):
        expected = (WITHDRAWN if kind == CHAT else MISSING if kind == ASK_OWNER
                    else enrolled.index(dense.NAMES[person]))
        total.update(raw=label == expected, high_correct=label == expected and score >= threshold,
                     high_wrong=label != expected and score >= threshold)
    return total


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    previous = json.loads((args.fusion / 'result.json').read_text(encoding='utf-8'))
    frozen = {Path(path).resolve(): checksum for path, checksum in previous['sources'].items()}
    archive = args.profile / 'profile' / 'versions.sqlite3'
    reader_file, _, writer_file, binding = action.artifacts(args.reader, args.normalizer)
    ptr_file = args.transfer / 'personal.pt'
    checkpoint, *layers = args.recurrent
    heads = (args.whole / 'action-19.pt', args.coordinate / 'action.pt')
    if any(frozen.get(path.resolve()) != action.digest(path) for path in (
            archive, reader_file, writer_file, ptr_file, checkpoint, *heads,
            *(folder / 'attention.pt' for folder in layers))):
        raise ValueError('Experiment requires the already audited checkpoint and heads')
    whole_limit = json.loads((args.whole / 'result.json').read_text(encoding='utf-8'))['results']['19']['cutoff']
    coordinate_limit = json.loads((args.coordinate / 'result.json').read_text(encoding='utf-8'))['threshold']
    args.output.mkdir(parents=True, exist_ok=False)
    train, pairs = flatten(examples(TRAIN, action.TRAIN_NAMES, dense.ALIASES))
    cal, _ = flatten(examples(CAL, action.CAL_NAMES, dense.CAL_ALIASES))
    held = {'regression': list(fusion.HELD), 'fresh': list(FRESH)}
    train_keys = {plan.anonymize(text, action.TRAIN_NAMES)[0] for text, _ in train}
    cal_keys = {plan.anonymize(text, action.CAL_NAMES)[0] for text, _ in cal}
    held_keys = {plan.anonymize(text, dense.NAMES)[0] for cases in held.values() for *_, text in cases}
    if train_keys & cal_keys or (train_keys | cal_keys) & held_keys:
        raise ValueError('Role supervision reuses held or calibration prompts')
    labels = torch.tensor([label for _, label in train])
    tokenizer = action.base.AutoTokenizer.from_pretrained(action.base.MODEL, local_files_only=True, trust_remote_code=False)
    settings = torch.load(writer_file, map_location='cpu', weights_only=True)
    arms = {}
    with action.local_model(args.recurrent) as model, action.VersionStore(archive, binding) as store:
        state = torch.load(reader_file, map_location='cpu', weights_only=True)
        reader = action.VersionReplyActions(model.config.hidden_size, state['scaffold_tokens'], slots=2)
        reader.load_state_dict(state)
        reader.eval().to(model.device)
        pointer = dense.JointPointer(model.config.hidden_size).eval()
        pointer.load_state_dict(torch.load(ptr_file, map_location='cpu', weights_only=True)['pointer'])
        extract = action.FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=19)
        early = action.FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=plan.DEPTH)
        training, _ = features(extract, [text for text, _ in train], action.TRAIN_NAMES)
        calibration, _ = features(extract, [text for text, _ in cal], action.CAL_NAMES)
        whole = nn.Sequential(nn.LayerNorm(2 * model.config.hidden_size), nn.Linear(2 * model.config.hidden_size, 64),
                              nn.GELU(), nn.Linear(64, 1))
        whole.load_state_dict(torch.load(heads[0], map_location='cpu', weights_only=True))
        whole.eval()
        targeted = coordinate.new_head(2 * model.config.hidden_size)
        targeted.load_state_dict(torch.load(heads[1], map_location='cpu', weights_only=True))
        targeted.eval()
        catalog = store.catalog()
        baselines, held_inputs = {}, {}
        for name, cases in held.items():
            questions = [text for *_, text in cases]
            held_inputs[name], mentioned = features(extract, questions, dense.NAMES)
            addresses, _ = action.rank(pointer, dense.align(early, questions, dense.NAMES), catalog)
            states = action.action_features(extract, questions, dense.NAMES)
            baselines[name] = {}
            for question, names, address, conf, (label, score) in zip(
                    questions, mentioned, addresses, action.predict(whole, states), coordinate.scores(targeted, states),
                    strict=True):
                full = address if conf >= whole_limit else None
                partial = coordinate.proposed(label, names, address) if score >= coordinate_limit else None
                baselines[name][question] = fusion.reconcile(partial, full, names)
        for arm, head in (('baseline', None), ('classification', fit(training, labels, pairs, contrast=False)),
                          ('paired', fit(training, labels, pairs, contrast=True))):
            results, threshold = {}, None
            if head is not None:
                marked = predict(head, calibration)
                threshold = calibration_threshold(marked, cal)
                atomic_save(head.state_dict(), args.output / f'{arm}.pt')
            for name, cases in held.items():
                proposals = baselines[name].copy()
                predictions = [] if head is None else predict(head, held_inputs[name])
                owners = [plan.anonymize(text, dense.NAMES)[1] for *_, text in cases]
                scores = None
                if head is not None:
                    scores = outcome(cases, owners, predictions, threshold)
                    for (*_, text), names, (label, confidence) in zip(cases, owners, predictions, strict=True):
                        if confidence >= threshold:
                            proposals[text] = rewrite(proposals[text], label, names, catalog)
                replay = coordinate.replay(store, model, tokenizer, reader, settings, proposals, cases=cases)
                results[name] = dict(replay, owner_labels=scores, predictions=predictions)
                print(arm, name, threshold, scores, replay['summary'], flush=True)
            arms[arm] = {'threshold': threshold, 'sets': results}
    if any(action.digest(path) != checksum for path, checksum in frozen.items()):
        raise RuntimeError('A frozen model artifact or archive changed')
    store_json(args.output / 'result.json', dict(arms=arms, train=len(train), calibration=len(cal),
               seconds=perf_counter() - started, sources={str(path): checksum for path, checksum in frozen.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fusion', type=Path)
    parser.add_argument('whole', type=Path)
    parser.add_argument('coordinate', type=Path)
    parser.add_argument('transfer', type=Path)
    parser.add_argument('profile', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--reader', type=Path, default=Path('runs/spora-dual-version-reply-v1'))
    parser.add_argument('--normalizer', type=Path, default=Path('runs/spora-version-normalizer-v1'))
    parser.add_argument('--recurrent', type=Path, nargs=3, required=True)
    main(parser.parse_args())
