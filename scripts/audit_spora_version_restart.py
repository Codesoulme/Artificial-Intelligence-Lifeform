"""Persist confirmed events in one process and replay mixed dialogue in another."""

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_fast_weight_lexical_write import base, compress_dimension, SOURCE
from scripts.audit_spora_version_clarification import action_head
from scripts.audit_spora_version_coverage import write_fresh
from scripts.audit_spora_version_reply_actions import atomic_reply
from scripts.audit_spora_version_roles import clause_features, flat_features, role_predictions
from scripts.audit_spora_version_timeline import TODAY, day
from scripts.audit_spora_versioned_memory_generation import read_versions
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.identity_fingerprints import IdentityFingerprints
from spora.research_ttt_ntp import LowRankFastWeight
from spora.research_version_clarification import (Intent, VersionAccessHead, VersionDecision, resolve_version_turn,
                                                  scope_version_intent, start_version_turn, use_explicit_read_plan)
from spora.research_version_roles import ABSTAIN, VersionRoleHead
from spora.research_version_scope import addressed_subject
from spora.research_version_store import MemoryAddress, VersionStore
from spora.research_versioned_memory_pointer import VersionReplyActions


WEEKDAY = MemoryAddress('安若帆', '饮品', '工作日')
WEEKEND = MemoryAddress('安若帆', '饮品', '周末')
NEAR_NAME = MemoryAddress('安若凡', '饮品')
PLACE = MemoryAddress('安若帆', '碰面地点')
EVENTS = {
    'weekday-first': (WEEKDAY, '桂花豆乳', 2, 2),
    'weekend': (WEEKEND, '青提苏打', 6, 6),
    'near-name': (NEAR_NAME, '糯米拿铁', 6, 6),
    'place': (PLACE, '银杏亭', 7, 7),
    'weekday-correction': (WEEKDAY, '麦香奶茶', 9, 9),
}
INITIAL = (
    ('write', 'weekday-first'), ('write', 'weekend'), ('write', 'near-name'), ('write', 'place'),
    ('说一个开会前放松的方法。', None), ('如果只有十秒呢？', None),
    ('工作日。安若帆的饮品写的是什么？请查当前版。', 'weekday-first'),
    ('write', 'weekday-correction'),
    ('工作日。安若帆的饮品写的是什么？请查当前版。', 'weekday-correction'),
    ('换个话题，怎么让泡好的茶凉得更快？', None),
)
RESTARTED = (
    ('工作日。安若帆的饮品写的是什么？请查当前版。', 'weekday-correction'),
    ('先别查这条，等我过会儿再说。工作日。安若帆的饮品写的是什么？', None),
    ('就查第一版吧。', None),
    ('帮我判断后面这句话是否通顺。工作日。安若帆的饮品写的是什么？', None),
    ('工作日。安若帆的饮品，2026-10-05有效的是什么？', 'weekday-first'),
    ('周末。安若帆的饮品，当前写的是什么？', 'weekend'),
    ('安若凡的饮品，当前写的是什么？', 'near-name'),
    ('安若帆的碰面地点，当前写的是什么？', 'place'),
    ('安若帆的饮品，当前写的是什么？', None),
    ('我想核对，但得先问我是哪一版。工作日。安若帆的饮品写的是什么？', None),
    ('就查第一版吧。', 'weekday-first'),
    ('再换个话题，写一句下雨天出门的提醒。', None),
    ('短一点，也别提下雨两个字。', None),
    ('安若舟的饮品，当前写的是什么？', None),
)


def catalog_address(question, catalog, *, explicit=False):
    if not catalog:
        return None
    names = tuple(sorted({address.name for address in catalog}))
    fingerprints = IdentityFingerprints(len(names))
    for i, name in enumerate(names):
        fingerprints.bind(i, name)
    fields = {i: tuple({address.field for address in catalog if address.name == name}) for i, name in enumerate(names)}
    subject = addressed_subject(question, fingerprints, fields, colloquial=True)
    if subject is None:
        return None
    candidates = [entry for entry in catalog if entry.name == names[subject] and addressed_subject(
                  question, fingerprints, {subject: (entry.field,)}, colloquial=True) == subject]
    def condition_matches(condition):
        if not explicit:
            return condition in question
        return re.search(rf'(?:^|[。；，;,.])\s*(?:在)?{re.escape(condition)}(?:时)?(?=[。；，、;,.])', question)

    scoped = [entry for entry in candidates if entry.condition and condition_matches(entry.condition)]
    choices = scoped if scoped else [entry for entry in candidates if not entry.condition]
    return choices[0] if len(choices) == 1 else None


class Replay:
    def __init__(self, store, model, tokenizer, settings, reader_state, version_state, access, *, explicit=False):
        self.store, self.model, self.tokenizer, self.settings = store, model, tokenizer, settings
        width = model.config.hidden_size
        self.action = action_head(width).eval()
        self.action.load_state_dict(access['action'], strict=True)
        self.access = VersionAccessHead(width).eval()
        self.access.load_state_dict(access['gate'], strict=True)
        self.cutoff = access['cutoff']
        self.explicit = explicit
        self.version = VersionRoleHead(width).eval()
        self.version.load_state_dict(version_state['expanded']['weights'], strict=True)
        self.reader = VersionReplyActions(width, reader_state['scaffold_tokens']).eval().to('cuda')
        self.reader.load_state_dict(reader_state, strict=True)
        embedding = model.get_input_embeddings().weight.detach()
        self.columns = compress_dimension(embedding.shape[1]).to(embedding.device)
        self.codebook = F.normalize(embedding.index_select(1, self.columns).float(), dim=-1)
        self.history, self.pending = [], None

    def confirm_write(self, event, address, value, effective, observed):
        # The user's confirmed fields provide the boundaries. Only the produced factors are stored.
        records = write_fresh(self.model, self.tokenizer, self.columns, layer=self.settings['layers'][0],
                              eta=self.settings['eta'], ridge=self.settings['ridge'],
                              events={'confirmed': ((address.name, value),)},
                              facts={'confirmed': f'登记{address.field}。{{name}}的值是{{value}}。'})
        keys, values = records[address.name][0]
        number = self.store.append(event, address, effective, observed, LowRankFastWeight(keys, values))
        self.history.clear()
        self.pending = None
        return number

    @torch.inference_mode()
    def start(self, question, address, versions):
        if self.explicit and address is None:
            return VersionDecision(Intent.OTHER), {'reason': 'unresolved_address'}
        vectors, mask = clause_features(self.model, self.tokenizer, [question])
        features = flat_features(vectors, mask)
        raw = Intent(int(self.action(features).argmax()))
        score = float(self.access(features).sigmoid().item())
        intent = scope_version_intent(raw, permits_lookup=raw != Intent.ASK or score >= self.cutoff)
        plan = role_predictions(self.version, vectors, mask)[0]['plan']
        timeline = tuple((entry.effective, entry.observed) for entry in versions)
        applied = use_explicit_read_plan(question, plan) if self.explicit and intent == Intent.READ else plan
        decision = start_version_turn(question, 0 if address is not None else None, timeline, TODAY, ABSTAIN, applied, intent)
        return decision, {'intent': raw.name, 'score': score, 'plan': plan, 'applied_plan': applied}

    @torch.inference_mode()
    def turn(self, question, *, read_disabled=False):
        catalog = self.store.catalog()
        decision, detail = None, {}
        if self.pending is not None:
            address, pending = self.pending
            versions = catalog.get(address, ())
            timelines = {0: tuple((item.effective, item.observed) for item in versions)}
            self.pending = None
            decision = resolve_version_turn(pending, question, timelines, TODAY)
            detail['continuation'] = decision.action.name
        if decision is None or decision.action == Intent.OTHER:
            address = catalog_address(question, catalog, explicit=self.explicit)
            versions = catalog.get(address, ())
            decision, initial = self.start(question, address, versions)
            detail.update(initial)
        selected = None
        value = torch.empty(0, dtype=torch.long, device=self.columns.device)
        if decision.route is not None and not read_disabled:
            selected = versions[decision.route[1]].number
            state = self.store.read(selected)
            value = read_versions({address.name: [(state.keys, state.values)]}, self.codebook)[address.name][0]
        prior = tuple(self.history)
        if decision.pending is not None:
            self.pending = address, decision.pending
            output = {'answer': decision.pending.question, 'token_ids': []}
        elif decision.action == Intent.DEFER:
            output = {'answer': '好，等你想查时再说。', 'token_ids': []}
        else:
            output = atomic_reply(self.model, self.tokenizer, self.reader, question, value, history=prior)
        self.history = [*self.history, {'role': 'user', 'content': question},
                        {'role': 'assistant', 'content': output['answer']}][-4:]
        return {'question': question, 'prior_messages': prior, 'address': None if address is None else asdict(address),
                'action': decision.action.name, 'selected': selected, 'evidence': detail, **output}


def grade(row, expected, numbers, eos):
    expected_value = EVENTS[expected][1] if expected is not None else None
    hits = [event for event, spec in EVENTS.items() if spec[1] in row['answer']]
    matched = row['selected'] == numbers.get(expected)
    return {**row, 'expected': expected, 'route_correct': matched, 'mentioned': hits,
            'value_correct': bool(expected_value is not None and matched and hits == [expected]
                                  and row['token_ids'] and row['token_ids'][-1] == eos)}


def totals(rows):
    counts = Counter()
    for row in rows:
        memory = row['expected'] is not None
        counts.update({'memory_turns': memory, 'correct_memory_routes': memory and row['route_correct'],
                       'correct_memory_answers': row['value_correct'], 'no_read_turns': not memory,
                       'unexpected_reads': not memory and row['selected'] is not None,
                       'remembered_value_in_no_read_turn': not memory and bool(row['mentioned'])})
    return counts


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    writing = args.phase == 'write'
    explicit = args.phase == 'recall-explicit'
    if writing:
        args.output.mkdir(parents=True, exist_ok=False)
    database = args.output / 'versions.sqlite3'
    settings_path = SOURCE / 'adapters.pt'
    sources = {str(path): base.sha(path) for path in (args.reader, args.version, args.access, settings_path,
               base.MODEL / 'model.safetensors', base.MODEL / 'tokenizer.json', base.MODEL / 'tokenizer_config.json')}
    binding = sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    path = args.output / (args.phase + '.json')
    if path.exists():
        raise FileExistsError(path)
    first = None if writing else json.loads((args.output / 'write.json').read_text(encoding='utf-8'))
    if first is not None and (first['sources'] != sources or first['database_sha'] != base.sha(database)):
        raise ValueError('The previous process did not leave the expected bank and sources')
    model = load_full_streamed()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    numbers, rows, writes, ablation = {}, [], [], None
    with VersionStore(database, binding, create=writing) as store:
        chat = Replay(store, model, tokenizer, torch.load(settings_path, map_location='cpu', weights_only=True),
                      torch.load(args.reader, map_location='cpu', weights_only=True)['atomic'],
                      torch.load(args.version, map_location='cpu', weights_only=True),
                      torch.load(args.access, map_location='cpu', weights_only=True), explicit=explicit)
        if first is not None:
            numbers = first['numbers']
            if chat.history or chat.pending is not None or os.getpid() == first['pid']:
                raise ValueError('Restart must start with fresh conversation state')
        eos = tokenizer.eos_token_id
        cases = INITIAL if writing else RESTARTED
        if explicit:
            cases += (('请查首版。工作日。安若帆的饮品写的是什么？', 'weekday-first'),
                      ('请核对当前版。周末。安若帆的饮品写的是什么？', 'weekend'),
                      ('并非周末。安若帆的饮品写的是什么？请查当前版。', None),
                      ('工作日或周末。安若帆的饮品写的是什么？请查当前版。', None))
        for question, expected in cases:
            if question == 'write':
                address, value, effective, observed = EVENTS[expected]
                number = chat.confirm_write(expected, address, value, day(effective), day(observed))
                numbers[expected] = number
                writes.append({'event': expected, 'number': number, 'value': value})
                continue
            answer = grade(chat.turn(question), expected, numbers, eos)
            rows.append(answer)
            print(answer['action'], answer['selected'], answer['answer'], flush=True)
            if first is not None and len(rows) == 1:
                active = rows[0]
                if active['prior_messages'] or any(spec[1] in question for spec in EVENTS.values()):
                    raise ValueError('The causal test requires an empty history and no answer in the question')
                chat.history.clear()
                off = grade(chat.turn(question, read_disabled=True), None, numbers, eos)
                chat.history = [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': active['answer']}]
                ablation = {'same_question': off['question'] == active['question'], 'off': off, 'on_correct': active['value_correct']}
    bank_sha = base.sha(database)
    if first is not None and bank_sha != first['database_sha']:
        raise ValueError('Read-only replay changed the saved bank')
    if any(base.sha(Path(path)) != digest for path, digest in sources.items()):
        raise ValueError('A frozen source changed')
    store_json(path, {'sources': sources, 'database_sha': bank_sha, 'database_bytes': database.stat().st_size,
                      'pid': os.getpid(), 'numbers': numbers, 'writes': writes, 'rows': rows, 'summary': totals(rows),
                      'empty_history_comparison': ablation, 'seconds': perf_counter() - started,
                      'peak_bytes': torch.cuda.max_memory_allocated()})
    print(json.dumps(totals(rows)), flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('reader', 'version', 'access', 'output'):
        cli.add_argument(name, type=Path)
    cli.add_argument('--phase', choices=('write', 'recall', 'recall-explicit'), required=True)
    main(cli.parse_args())
