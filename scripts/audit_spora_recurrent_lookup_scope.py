"""Compare learned per-turn memory scope with a current-message-only planner."""

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_paired_lookup_clarification import VALUES
from scripts.spora_version_chat import (
    FIELDS, VersionReplyActions, VersionStore, artifacts, base, compress_dimension, local_model,
    reply_to_route, resolve_address,
)
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_version_store import MemoryAddress
from spora.research_version_timeline import VersionQuery


CHAT, ASK, READ = range(3)  # READ + candidate position
VERSIONS = (VersionQuery('ordinal', 1), VersionQuery('current'), VersionQuery('previous'))
SEED = 32626

PEOPLE = {
    'train': (('苏岚川', 'books', '祝青遥', 'places'), ('袁和宁', 'items', '梁冬笙', 'drinks'),
              ('白未溪', 'places', '乔疏林', 'places'), ('文席月', 'drinks', '陆知禾', 'books')),
    'calibration': (('陶景清', 'items', '阮燕池', 'places'), ('连晓冬', 'drinks', '湛栖云', 'books')),
    'held': (('岑蕙安', 'books', '唐砚平', 'places'), ('叶澄依', 'items', '顾遥青', 'drinks'),
             ('顾遥青', 'books', '岑蕙安', 'books')),
}
# Each row shares its opening across changes, cancellations and ordinary remarks.
FRAMES = {
    'train': (
        ('我想核对{a}。', '继续刚才的。', '换成{b}，不查前一项。', '再加上{b}，原来的也查。',
         '我取消这次查阅。', '顺带说一句，{b}我自己另有安排。'),
        ('先帮我翻{a}的记录。', '这个查阅保持不变。', '前面那项撤掉，只留{b}。', '别漏掉{b}，两项都要。',
         '先不查了，等我明确后再开始。', '「请查{b}」只是举例，查阅仍旧。'),
        ('请找一下{a}留底的那项。', '先维持原要求。', '我改主意了，核对{b}。', '也核对{b}，第一项照查。',
         '先把这个请求撤回。', '说到{b}，我只是想到了，手头还是照原请求。'),
    ),
    'calibration': (
        ('请把{a}列到本次查阅里。', '原先选的照办。', '刚说的作废，我只要{b}。', '此外把{b}加进查阅。',
         '这次调阅暂时搁置。', '别把举例的{b}算成我的查询。'),
    ),
    'held': (
        ('我需要核实{a}，等下告诉你要哪次的。', '就沿用刚拍板的对象。', '那件算了，实际要看的是{b}。',
         '前一项继续，顺手把{b}也纳进来。', '到此打住，档案今天不翻了。',
         '提到{b}是为了说明情况，眼下这项照旧。'),
        ('有一份旧记录想找，事情是{a}。', '查阅范围不用动。', '不要管最先那条，接下来核查{b}。',
         '把{b}并进来，已有的那项保留。', '这条查询我收回了，先聊别的。',
         '「能核对{b}吗」是旁人的原话，我这边仍照原定。'),
    ),
}
QUESTIONS = {
    'train': (('首版。', '最早登记的那份。'), ('当前版。', '现在管用的记录。'),
              ('上一版。', '上次换下来的。'), ('把「查首版」这句话改得客气点。', '「查上一版」这个说法通顺吗？')),
    'calibration': (('给最早记的那条。',), ('现行版。',), ('用前一份。',), ('给「查现行版」换一种措辞。',)),
    'held': (('首版。', '按头一回存下的那份。'), ('当前版。', '取仍在执行的那条。'),
             ('上一版。', '给刚替换下去的记录。'), ('有人写了「首版」，帮我换个说法。', '只评价「取最新那份」这句话。')),
}


@dataclass(frozen=True)
class Episode:
    scope: str
    addresses: tuple[MemoryAddress, MemoryAddress]
    events: tuple[str, ...]
    question: str
    action: int
    version: int


def describe(address):
    return f'{address.name}的{FIELDS[address.field]}'


def episodes(split):
    result = []
    for owner, field, other, other_field in PEOPLE[split]:
        addresses = (MemoryAddress(owner, field), MemoryAddress(other, other_field))
        names = dict(zip(('a', 'b'), map(describe, addresses), strict=True))
        states = [('absent', (), ASK)]
        for opening, *updates in FRAMES[split]:
            start = opening.format(**names)
            states.extend((mode, (start, update.format(**names)), action)
                          for (mode, action), update in zip((('retain', READ), ('replace', READ + 1),
                              ('both', ASK), ('cancel', ASK), ('mention', READ)), updates, strict=True))
        for mode, history, action in states:
            for version, questions in enumerate(QUESTIONS[split]):
                for question in questions:
                    chosen = CHAT if version == len(VERSIONS) else action
                    result.append(Episode(mode, addresses, history, question, chosen,
                                          version if chosen >= READ else -1))
    return result


def features(extract, groups):
    def prompt(text, address):
        return f'{text}\n候选记录：{describe(address)}。'

    prompts = dict.fromkeys(prompt(text, address) for rows in groups.values() for row in rows
                            for text in (*row.events, row.question) for address in row.addresses)
    prompts = list(prompts)
    vectors = {}
    for start in range(0, len(prompts), 48):
        batch = prompts[start:start + 48]
        outputs = extract(batch)
        vectors.update((text, row[len(text) - 6:len(text)].float().mean(0))
                       for text, row in zip(batch, outputs, strict=True))

    def candidates(text, addresses):
        return torch.stack([vectors[prompt(text, address)] for address in addresses])

    result = {}
    for name, rows in groups.items():
        current = torch.stack([candidates(row.question, row.addresses) for row in rows])
        history = current.new_zeros(len(rows), 2, 2, current.shape[-1])  # episode, turn, candidate, feature
        valid = torch.zeros(len(rows), 2, dtype=torch.bool)
        for index, row in enumerate(rows):
            for turn, text in enumerate(row.events):
                history[index, turn] = candidates(text, row.addresses)
                valid[index, turn] = True
        result[name] = current, history, valid
    return result


class ScopePlanner(nn.Module):
    """Shared state and scoring per candidate, with no candidate-position parameter."""

    def __init__(self, width, size=64, *, supervise_scope=False):
        super().__init__()
        self.encode = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, size), nn.GELU())
        self.transition = nn.GRUCell(size, size)
        self.combine = nn.Sequential(nn.Linear(2 * size, size), nn.GELU())
        self.read = nn.Linear(size, 1)
        self.other = nn.Linear(size, 2)
        self.version = nn.Linear(size, len(VERSIONS))
        self.membership = nn.Linear(size, 1) if supervise_scope else None

    def observe(self, state, text_features):
        inputs = self.encode(text_features).flatten(0, 1)
        return self.transition(inputs, state.flatten(0, 1)).reshape_as(state)

    def advance(self, history, valid, *, recurrent):
        state = history.new_zeros(history.shape[0], history.shape[2], self.transition.hidden_size)
        if recurrent:
            for turn, present in zip(history.unbind(1), valid.unbind(1), strict=True):
                state = torch.where(present[:, None, None], self.observe(state, turn), state)
        return state

    def plan(self, current, state):
        query = self.encode(current)
        joint = self.combine(torch.cat((query, state), -1))
        pooled = joint.mean(1)
        return torch.cat((self.other(pooled), self.read(joint).squeeze(-1)), -1), self.version(pooled)

    def forward(self, current, history, valid, *, recurrent):
        return self.plan(current, self.advance(history, valid, recurrent=recurrent))


SCOPE_MEMBERS = {'absent': (0, 0), 'retain': (1, 0), 'replace': (0, 1),
                 'both': (1, 1), 'cancel': (0, 0), 'mention': (1, 0)}


def fit(data, rows, *, recurrent, steps, supervise_scope=False):
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)
    head = ScopePlanner(data[0].shape[-1], supervise_scope=supervise_scope)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    actions, versions = (torch.tensor([getattr(row, name) for row in rows]) for name in ('action', 'version'))
    members = torch.tensor([SCOPE_MEMBERS[row.scope] for row in rows], dtype=torch.float32)
    for _ in range(steps):
        batch = torch.randperm(len(rows), generator=rng)[:32]
        current, history, valid = (tensor[batch] for tensor in data)
        state = head.advance(history, valid, recurrent=recurrent)
        route, version = head.plan(current, state)
        loss = F.cross_entropy(route, actions[batch])
        version_loss = F.cross_entropy(version, versions[batch], ignore_index=-1, reduction='sum')
        loss += version_loss / actions[batch].ge(READ).sum().clamp_min(1)
        if supervise_scope:
            loss += F.binary_cross_entropy_with_logits(head.membership(state).squeeze(-1), members[batch])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return head.eval()


def decisions(logits, threshold, members=None):
    route, version = logits
    confidence, actions = route.softmax(-1).max(-1)
    actions = torch.where(actions.ge(READ) & confidence.lt(threshold), ASK, actions)
    if members is not None:
        chosen = members.gather(1, (actions - READ).clamp_min(0).unsqueeze(1)).squeeze(1)
        actions = torch.where(actions.ge(READ) & ~(chosen & members.sum(1).eq(1)), ASK, actions)
    return list(zip(actions.tolist(), version.argmax(-1).tolist(), strict=True))


def calibration(logits, rows, members=None):
    def score(threshold):
        valid, wrong = 0, 0
        for (action, version), row in zip(decisions(logits, threshold, members), rows, strict=True):
            valid += action == row.action and (action < READ or version == row.version)
            wrong += action >= READ and (action, version) != (row.action, row.version)
        return valid - 2 * wrong, -wrong, -threshold
    return max((0., .5, .65, .8, .9, .99, 1.), key=score)


def evaluate(rows, plan, catalog, store, model, tokenizer, codebook, reader, *, values=VALUES):
    today = date.today()
    results, counts = [], Counter()
    generated = {}
    for row, (action, version) in zip(rows, plan, strict=True):
        counts.update(total=1, active=row.action >= READ, full=action == row.action and (action < READ or version == row.version),
                      matching_address=action == row.action and row.action >= READ, planned_read=action >= READ,
                      unwanted_read=action >= READ and row.action < READ,
                      wrong_address=action >= READ and row.action >= READ and action != row.action,
                      wrong_version=action == row.action and action >= READ and version != row.version)
        selected, answer = None, None
        if action >= READ:
            address = row.addresses[action - READ]
            selected, reason = resolve_address(catalog, address, VERSIONS[version], today, retain_same_day=True)
            if reason != 'located':
                raise RuntimeError('An evaluated address or version was not in the synthetic archive')
            key = row.question, selected
            if key not in generated:
                _, output = reply_to_route(model, tokenizer, store, codebook, reader,
                                           f'请查{describe(address)}，只报名称。', [selected],
                                           labels=[FIELDS[address.field]])
                generated[key] = output['answer']
            answer = generated[key]
        expected = None
        if row.action >= READ:
            expected = resolve_address(catalog, row.addresses[row.action - READ], VERSIONS[row.version], today,
                                       retain_same_day=True)[0]
        counts['correct_read'] += (expected is not None and selected == expected and version == row.version
                                   and answer == f'{FIELDS[row.addresses[row.action - READ].field]}：{values[expected]}。')
        results.append(dict(scope=row.scope, history=row.events, question=row.question,
                            expected=[row.action, row.version], decision=[action, version],
                            expected_record=expected, selected=selected, answer=answer))
    return dict(summary=dict(counts), rows=results)


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    archive = args.prior / 'profile' / 'versions.sqlite3'
    prior = json.loads((args.prior / 'result.json').read_text(encoding='utf-8'))
    if base.sha(archive) != prior['archive_sha256']:
        raise ValueError('Archive does not match the completed CLI replay')
    reader_path, _, _, binding = artifacts(args.reader, args.normalizer)
    inputs = (archive, reader_path, args.recurrent[0], *(folder / 'attention.pt' for folder in args.recurrent[1:]))
    sources = {path: base.sha(path) for path in inputs}
    args.output.mkdir(parents=True, exist_ok=False)
    groups = {group: episodes(group) for group in PEOPLE}
    for row in groups['held']:
        if any(value in text for value in VALUES.values() for text in (*row.events, row.question)):
            raise ValueError('A stored value occurs in the visible evaluation text')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    results = {}
    with local_model(args.recurrent) as model, VersionStore(archive, binding) as store:
        catalog = store.catalog()
        state = torch.load(reader_path, weights_only=True, map_location='cpu')
        reader = VersionReplyActions(model.config.hidden_size, state['scaffold_tokens'], slots=2)
        reader.load_state_dict(state)
        reader.eval().to(model.device)
        embedding = model.get_input_embeddings().weight.detach()
        columns = compress_dimension(embedding.shape[1]).to(embedding.device)
        codebook = F.normalize(embedding.index_select(1, columns).float(), dim=-1)
        data = features(FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer), groups)
        arms = (('scope_supervised', True, True),) if args.scope_supervision else (
            ('current_only', False, False), ('recurrent', True, False))
        for name, recurrent, supervised in arms:
            head = fit(data['train'], groups['train'], recurrent=recurrent, steps=args.steps, supervise_scope=supervised)
            path = args.output / f'{name}.pt'
            atomic_save(head.state_dict(), path)
            head.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'), strict=True)
            with torch.inference_mode():
                cal_current, cal_history, cal_valid = data['calibration']
                cal_state = head.advance(cal_history, cal_valid, recurrent=recurrent)
                cal_members = head.membership(cal_state).squeeze(-1).ge(0) if supervised else None
                cut = calibration(head.plan(cal_current, cal_state), groups['calibration'], cal_members)
                current, history, valid = data['held']
                state = head.advance(history, valid, recurrent=recurrent)
                plan = head.plan(current, state)
                swapped = head(current.flip(1), history.flip(2), valid, recurrent=recurrent)
                torch.testing.assert_close(plan[0], torch.cat((swapped[0][:, :READ], swapped[0][:, READ:].flip(1)), -1))
                torch.testing.assert_close(plan[1], swapped[1])
                members = head.membership(state).squeeze(-1).ge(0) if supervised else None
                results[name] = evaluate(groups['held'], decisions(plan, cut, members), catalog, store, model, tokenizer,
                                         codebook, reader)
                results[name].update(threshold=cut, head_sha256=base.sha(path), uncalibrated=decisions(plan, 0.))
                if recurrent:
                    blank = torch.zeros_like(state)
                    erased = head.plan(current, blank)
                    erased_members = head.membership(blank).squeeze(-1).ge(0) if supervised else None
                    results[name]['after_erasing_state'] = decisions(erased, cut, erased_members)
                if supervised:
                    results[name]['predicted_members'] = members.tolist()
                    results[name]['without_threshold'] = decisions(plan, 0., members)
            print(name, results[name]['summary'], flush=True)
    if any(base.sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('An input model or the archive changed')
    store_json(args.output / 'result.json', dict(results=results, counts={key: len(value) for key, value in groups.items()},
               steps=args.steps, sources={str(path): digest for path, digest in sources.items()}, seconds=perf_counter() - started))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('prior', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--reader', type=Path, default=Path('runs/spora-dual-version-reply-v1'))
    cli.add_argument('--normalizer', type=Path, default=Path('runs/spora-version-normalizer-v1'))
    cli.add_argument('--recurrent', type=Path, nargs=3, required=True)
    cli.add_argument('--steps', default=240, type=int)
    cli.add_argument('--scope-supervision', action='store_true')
    args = cli.parse_args()
    if args.steps < 1:
        cli.error('--steps must be positive')
    main(args)
