"""Interactive DeepSeek chat with a dedicated, versioned SPORA memory profile."""

import argparse
from contextlib import ExitStack, contextmanager
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import sys
from types import SimpleNamespace
from uuid import uuid4

import torch
from torch.nn import functional as F

from scripts.audit_spora_active_time import TIME
from scripts.audit_spora_dialogue_version_scope import TemporalProposals
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_dual_version_reply import FIELDS, reply as version_reply
from scripts.audit_spora_dual_version_routing import reply_to_route
from scripts.audit_spora_fast_weight_lexical_write import SOURCE, base, compress_dimension, tokens
from scripts.audit_spora_typed_composition import SLOTS, arrangement
from scripts.audit_spora_version_coverage import write_fresh
from scripts.audit_spora_versioned_memory_generation import read_versions
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_ttt_read_adapter_pilot import generate
from spora.awake_chat import awake_chat_lines
from spora.integrated_checkpoint import load_integrated_runtime
from spora.research_multi_version_route import resolve_address
from spora.research_recurrent_decode import load_recurrent_replacements
from spora.research_ttt_ntp import LowRankFastWeight
from spora.research_version_dialogue_scope import (PartialLookup, advance_scope, ambiguous_partial_reply, archive_lookup_canceled,
                                                   continue_partial_lookup, dialogue_targets, eligible_spoken_lookup, pending_after_lookup,
                                                   pending_followup_targets, resolve_spoken_proposal, start_partial_lookup)
from spora.research_version_store import MemoryAddress, VersionStore
from spora.research_version_timeline import parse_version_query
from spora.research_versioned_memory_pointer import VersionReplyActions


FIELD_IDS = {label: field for field, label in FIELDS.items()}
NAME = re.compile(r'[\u3400-\u9fff]{2,12}')
VALUE = re.compile(r'[\w ·-]{1,24}')
CANCEL = re.compile(r'^(?:先别查(?:了)?|先不查(?:了)?|不查了|算了|先暂停)(?:[，,。；;！!\s]|$)')
DEFER_UNTIL = re.compile(r'(?:稍后|过会儿|等会儿|改天|下次)再(?:问|查|说)[。！？!?]*$')
HELP = ('/记 姓名 | 字段 | 短值 [| 生效日期 YYYY-MM-DD]。字段可用饮品、见面地点、用品、书目。'
        '查询时可说「请查林语的饮品当前版」，再问「那上一版呢」。'
        '/清空对话 清除本轮对话和指代范围。/退出 结束。')
COMPOSE_HELP = ('/成句 姓名 | 字段版次 | 字段版次 | 时间 | 写作要求。'
                '也可先查两项后说「按这两项写一句消息」。'
                '可组合见面地点和用品、见面地点和书目、书目和饮品。')
COMPOSE_PAIRS = (
    {'places', 'items'}, {'places', 'books'}, {'books', 'drinks'},
)
COMPOSE_FIELD = re.compile(r'(饮品|见面地点|用品|书目)\s*(当前版|首版|上一版|第[一二三四五六七八九1-9]版)')
COMPOSE_REFERENCE = re.compile(r'^(?:那就|那|接着|麻烦|请|就)?(?:按|用|根据)(?:刚(?:刚)?查到的)?这两(?:条记录|项记录|条|项)')
COMPOSE_ACTION = re.compile(r'(?:写|拟)(?:出|好|成)?(?:一条|一句|个|条|句)?(?:微信|短讯|短信|便条|通知|消息|提醒|邀约)')
COMPOSE_CANCEL = re.compile(r'先别|不用写|不要写|还没定|还未确定|待定|取消|算了|换(?:个|一个)?(?:话题|问题)|改查')
CONFIRMED_TIME = re.compile(r'(?:时间(?:是|定(?:在|为)|改(?:到|成))|(?:就)?约在)\s*(' + TIME.pattern + r')(?=[，,。！？!?；;\s]|$)')
BARE_TIME = re.compile(r'(?:时间(?:是|定(?:在|为)|改(?:到|成))\s*)?(' + TIME.pattern + r')[。！？!?]?')
AWAKE_RECALL = re.compile(
    r'(?:请)?(?:再说一遍刚才查到的(?:那项|记录)?|把刚才查到的(?:那项|记录)?再说一遍'
    r'|刚才查到的(?:那项|记录)?是什么)[。！？!?]?'
)
AWAKE_HELP = '/status 查看持续状态，/relax 放松，/focus 专注，/sleep 暂停，/wake 唤醒。'


def write_request(raw, today):
    parts = [part.strip() for part in raw.split('|')]
    if len(parts) not in (3, 4):
        raise ValueError('请用 /记 姓名 | 字段 | 短值 [| YYYY-MM-DD]。')
    name, field, value = parts[:3]
    if not NAME.fullmatch(name):
        raise ValueError('姓名须为 2 到 12 个中文字符。同名记录可使用不同显示名。')
    if field not in FIELD_IDS:
        raise ValueError('字段请选择饮品、见面地点、用品或书目。')
    if not VALUE.fullmatch(value):
        raise ValueError('请用简短的名称或数值，可含空格、间隔号和连字符。')
    try:
        effective = date.fromisoformat(parts[3]) if len(parts) == 4 else today
        if len(parts) == 4 and effective.isoformat() != parts[3]:
            raise ValueError('Noncanonical date')
    except ValueError as error:
        raise ValueError('日期请用 YYYY-MM-DD。') from error
    return MemoryAddress(name, FIELD_IDS[field]), value, effective


def compose_request(raw):
    parts = [part.strip() for part in raw.split('|', 4)]
    if len(parts) != 5 or not all(parts) or not NAME.fullmatch(parts[0]):
        raise ValueError(COMPOSE_HELP)
    name, first, second, when, question = parts
    targets = []
    for field in (first, second):
        match = COMPOSE_FIELD.fullmatch(field)
        if not match:
            raise ValueError('请明确写出每个字段的版次，例如「见面地点当前版」。')
        label, version = match.groups()
        targets.append((MemoryAddress(name, FIELD_IDS[label]), parse_version_query(version)))
    if {address.field for address, _ in targets} not in COMPOSE_PAIRS:
        raise ValueError(COMPOSE_HELP)
    if not VALUE.fullmatch(when) or len(question) > 512:
        raise ValueError('请给出一个简短的明确时间及不超过 512 字的写作要求。')
    return targets, when, question


def compose_followup(question):
    return bool(COMPOSE_REFERENCE.match(question) and COMPOSE_ACTION.search(question)
                and not COMPOSE_CANCEL.search(question) and not re.search(r'[“”「」]|翻译|润色', question))


def confirmed_time(question, *, answer_only=False):
    if COMPOSE_CANCEL.search(question) or re.search(r'或者|还是|暂定|如果|可能|不确定|未定|备选|[“”「」]', question):
        return None
    if answer_only:
        match = BARE_TIME.fullmatch(question)
        return match[1] if match else None
    choices = CONFIRMED_TIME.findall(question)
    candidates = [match.group() for match in TIME.finditer(question)]
    return choices[0] if len(choices) == 1 and candidates == choices else None


def clarify(catalog, targets, reasons, today):
    duplicate = len({item.address for item in targets if item.address}) < sum(item.address is not None for item in targets)
    if len(targets) > 2 or (duplicate and 'located' in reasons):
        return '请分开查阅，每次明确指定一项或两项记录。'
    for item, reason in zip(targets, reasons, strict=True):
        address = item.address
        if address is None:
            return '我还没确定要查谁的哪项记录。请告诉我姓名和事项。'
        label = FIELDS[address.field]
        if reason == 'unknown_address' or address not in catalog:
            return f'我没有找到{address.name}的{label}记录。请核对姓名或事项。'
        if reason == 'missing_version':
            return f'{address.name}的{label}找不到这一版。你可以指定当前版或首版。'
        if reason == 'unresolved_scope':
            active = [entry for entry in catalog[address] if entry.effective <= today and entry.observed <= today]
            if not active:
                return f'{address.name}的{label}目前没有生效的记录。'
            if len(active) == 1:
                return f'{address.name}的{label}目前只有一版。你可以说「查当前版」。'
            return f'{address.name}的{label}要查当前版、首版，还是上一版？'
    return '请分别给出要查的人物、事项和版次。'


def deferred_lookup(question):
    canceled = CANCEL.match(question)
    if not canceled:
        return False
    remainder = question[canceled.end():].strip('，,。；;！! ')
    return not remainder or bool(DEFER_UNTIL.search(remainder))


def ask_lookup_coordinate(partial):
    if partial.owner is None and partial.field is not None:
        return f'请告诉我要查谁的{FIELDS[partial.field]}。'
    if partial.owner is None:
        return '请告诉我要查谁的哪项记录。'
    if partial.field is None:
        return f'要查{partial.owner}的哪项记录？可回答饮品、见面地点、用品或书目。'
    return f'要查{partial.owner}的{FIELDS[partial.field]}哪一版？可回答当前版、首版或上一版。'


def ask_spoken_approval(address):
    return {'kind': 'clarify', 'answer': (f'你是要查{address.name}的{FIELDS[address.field]}吗？'
                                          '确认后可以说「是，上一版」或「是」，也可直接改口。'),
            'selected': [], 'reasons': ['unconfirmed_address']}


class VersionChat:
    def __init__(self, store, model, tokenizer, reader, propose, settings, *, ordinary_tokens, composer=None,
                 typed_lookup=False, awake=None, spoken=None):
        self.store, self.model, self.tokenizer, self.reader = store, model, tokenizer, reader
        self.propose, self.settings, self.ordinary_tokens = propose, settings, ordinary_tokens
        self.composer = composer
        self.typed_lookup = typed_lookup
        self.awake = awake
        self.spoken = spoken
        embedding = model.get_input_embeddings().weight.detach()
        self.columns = compress_dimension(embedding.shape[1]).to(embedding.device)
        self.codebook = F.normalize(embedding.index_select(1, self.columns).float(), dim=-1)
        self.reset_dialogue()

    def reset_dialogue(self):
        if self.awake is not None:
            self.awake[1]()
        self.history, self.scope, self.pending = [], (), None
        self.partial_lookup = None
        self.spoken_pending = None
        self.compose_selection, self.compose_pending = (), None

    @torch.inference_mode()
    def save(self, raw, today):
        address, value, effective = write_request(raw, today)
        pieces = tokens(self.tokenizer, value)
        if not 1 <= len(pieces) <= 4 or set(pieces) & set(self.tokenizer.all_special_ids):
            raise ValueError('这个值太长，请用四个分词以内的短名称。')
        bank = write_fresh(self.model, self.tokenizer, self.columns, layer=self.settings['layers'][0],
                           eta=self.settings['eta'], ridge=self.settings['ridge'],
                           events={address.field: ((address.name, value),)},
                           facts={address.field: '\n{value}'}, minimum_tokens=1)
        if read_versions(bank, self.codebook)[address.name][0].tolist() != pieces:
            raise ValueError('写入前的读回校验未通过，请换一个短名称。')
        keys, values = bank[address.name][0]
        number = self.store.append(uuid4().hex, address, effective, today, LowRankFastWeight(keys, values))
        self.reset_dialogue()
        return {'kind': 'saved', 'answer': f'已登记{address.name}的{FIELDS[address.field]}。', 'record': number}

    @torch.inference_mode()
    def compose(self, raw, today):
        if self.composer is None:
            raise ValueError('启动时请用 --composition 指定成句模型目录。')
        if self.awake is not None:
            self.awake[1]()
        self.scope, self.pending = (), None
        self.partial_lookup = None
        self.spoken_pending = None
        self.compose_selection, self.compose_pending = (), None
        targets, when, question = compose_request(raw)
        catalog = self.store.catalog()
        decisions = [resolve_address(catalog, address, query, today, retain_same_day=True)
                     for address, query in targets]
        if any(reason != 'located' for _, reason in decisions):
            answer = next(f'未找到{address.name}的{FIELDS[address.field]}的指定版次。'
                          for (address, _), (_, reason) in zip(targets, decisions, strict=True)
                          if reason != 'located')
            return {'kind': 'clarify', 'answer': answer, 'selected': []}
        return self.compose_selected(question, when, tuple(
            (address, number) for (address, _), (number, _) in zip(targets, decisions, strict=True)
        ))

    def compose_selected(self, question, when, selection):
        if when not in question:
            question += f'时间是{when}。'
        when_ids = tokens(self.tokenizer, when)
        if not 1 <= len(when_ids) <= 4 or self.tokenizer.decode(when_ids) != when:
            raise ValueError('时间须能按原文编码为一至四个词元。')
        selected = [number for _, number in selection]
        bank = {}
        for number in selected:
            state = self.store.read(number)
            bank[number] = [(state.keys, state.values)]
        values = read_versions(bank, self.codebook)
        fields = [(address.field, values[number][0]) for address, number in selection]
        memory = arrangement([*fields, ('time', torch.tensor(when_ids, device=self.model.device))])
        output = version_reply(self.model, self.tokenizer, self.composer, question, memory,
                               require_coverage=True, cover_present_slots=True, max_new_tokens=72)
        copied = {SLOTS[item['slot']] for item in output['copied']}
        expected = {field for field, _ in fields} | {'time'}
        if copied != expected or output['token_ids'][-1:] != [self.tokenizer.eos_token_id]:
            return {'kind': 'clarify', 'answer': '这次没有生成完整消息，请核对字段和写作要求。', 'selected': selected}
        return {'kind': 'composed', 'answer': output['answer'], 'selected': selected, 'copied': sorted(copied)}

    def compose_from_dialogue(self, question):
        selection, pending = self.compose_selection, self.compose_pending
        self.compose_selection, self.compose_pending = (), None
        if pending is not None and (when := confirmed_time(question, answer_only=True)) is not None:
            selection, request = pending
            return self.compose_selected(request, when, selection)
        if not compose_followup(question):
            return None
        self.scope, self.pending = (), None
        self.partial_lookup = None
        self.spoken_pending = None
        if not selection:
            return {'kind': 'clarify', 'answer': '请先查阅同一人的两项记录，再告诉我要写什么。', 'selected': []}
        owner = selection[0][0].name
        fields = '和'.join(FIELDS[address.field] for address, _ in selection)
        request = f'请给{owner}{question}。请用已查到的{fields}。'
        when = confirmed_time(question)
        if when is None:
            self.compose_pending = selection, request
            return {'kind': 'clarify', 'answer': '请确认一个具体时间，例如「周五傍晚」。', 'selected': []}
        return self.compose_selected(request, when, selection)

    def ask_for_coordinate(self, partial):
        self.partial_lookup = partial
        return {'kind': 'clarify', 'answer': ask_lookup_coordinate(partial),
                'selected': [], 'reasons': ['incomplete_address']}

    @torch.inference_mode()
    def turn(self, question, today):
        if len(question) > 512:
            raise ValueError('每条消息最多 512 个字符。')
        if deferred_lookup(question):
            self.reset_dialogue()
            return {'kind': 'defer', 'answer': '好，等你想查时再说。', 'selected': [], 'reasons': []}
        if archive_lookup_canceled(question):
            self.reset_dialogue()
        if self.awake is not None:
            recent = self.awake[1]()
            if AWAKE_RECALL.fullmatch(question):
                return self.recall_awake(recent)
        if self.composer is not None:
            composed = self.compose_from_dialogue(question)
            if composed is not None:
                return composed
        catalog = self.store.catalog()
        enrolled = {address.name for address in catalog}
        proposal, self.spoken_pending = self.spoken_pending, None
        approved = None
        if proposal is not None and eligible_spoken_lookup(question):
            decision, approved = resolve_spoken_proposal(question, proposal, enrolled, FIELDS)
            if decision == 'awaiting_confirmation':
                self.spoken_pending = proposal
                return ask_spoken_approval(proposal)
            if decision == 'declined':
                return {'kind': 'clarify', 'answer': '请重新告诉我要查谁的哪项记录。', 'selected': [],
                        'reasons': ['address_declined']}
        previous, self.partial_lookup = self.partial_lookup, None
        if previous and (slot := ambiguous_partial_reply(question, enrolled, FIELDS)):
            self.partial_lookup = previous
            label = {'owner': '姓名', 'field': '字段', 'query': '版次'}[slot]
            return {'kind': 'clarify', 'answer': f'请确认这次要查的{label}，一次选一个。',
                    'selected': [], 'reasons': ['ambiguous_address']}
        partial = approved or (continue_partial_lookup(question, previous, enrolled, FIELDS) if previous else None)
        if partial is None and self.pending is None:
            partial = start_partial_lookup(question, enrolled, FIELDS)
        if partial is not None:
            self.scope, self.pending = (), None
            target = partial.target()
            if target is None:
                return self.ask_for_coordinate(partial)
            targets = (target,)
        else:
            targets = (dialogue_targets(question, enrolled, FIELDS, self.scope) if self.pending is None else
                       pending_followup_targets(question, enrolled, FIELDS, self.pending))
        if len(targets) == 1 and targets[0].source == 'unresolved_confirmed_reference':
            self.history, self.scope, self.pending = [], (), None
            return self.ask_for_coordinate(PartialLookup(query=targets[0].query))
        if not targets and self.spoken is not None and proposal is None and eligible_spoken_lookup(question):
            suggested = self.spoken(question, enrolled)
            if isinstance(suggested, PartialLookup):
                owner, field = suggested.owner, suggested.field
                supported = ((owner in enrolled and field is None)
                             or (owner is None and any(address.field == field for address in catalog)))
                if supported and suggested.query is None:
                    self.history, self.scope, self.pending = [], (), None
                    return self.ask_for_coordinate(suggested)
            elif suggested in catalog:
                self.history, self.scope, self.pending = [], (), None
                self.spoken_pending = suggested
                return ask_spoken_approval(suggested)
        proposals = iter(self.propose([item.evidence for item in targets
                                       if item.evidence is not None and item.address is not None]))
        decisions = []
        for item in targets:
            query = next(proposals) if item.evidence is not None and item.address is not None else item.query
            decisions.append(resolve_address(catalog, item.address, query, today, retain_same_day=True))
        route = [number for number, _ in decisions]
        reasons = tuple(reason for _, reason in decisions)
        selected = []
        if not targets:
            kind = 'chat'
            output = generate(self.model, self.tokenizer, question, history=self.history, max_new_tokens=self.ordinary_tokens)
            answer = output['answer']
        elif None in route or len(route) > 2 or len(set(route)) != len(route):
            kind = 'clarify'
            answer = clarify(catalog, targets, reasons, today)
        else:
            kind = 'read'
            labels = None
            read_question = question
            confirmed = len(targets) == 1 and targets[0].source == 'confirmed'
            if self.typed_lookup or confirmed or self.awake is not None:
                addresses = [target.address for target in targets]
                multi_person = len({address.name for address in addresses}) > 1
                labels = [f'{address.name}的{FIELDS[address.field]}' if multi_person else FIELDS[address.field]
                          for address in addresses]
            if confirmed:
                target = targets[0]
                version = {'current': '当前版', 'previous': '上一版'}.get(target.query.kind)
                if version is None:
                    version = f'第{target.query.coordinate}版'
                read_question = f'请查{target.address.name}的{FIELDS[target.address.field]}{version}。'
            selected, output = reply_to_route(self.model, self.tokenizer, self.store, self.codebook, self.reader,
                                              read_question, route, history=self.history, labels=labels)
            answer = output['answer']
            if self.awake is not None and len(selected) == 1:
                observe, _, bind = self.awake
                field = targets[0].address.field
                observe(tuple(float(candidate == field) for candidate in FIELDS))
                bind(selected[0], field)
            if output['token_ids'] and output['token_ids'][-1] == self.tokenizer.eos_token_id:
                answer = answer.rstrip('，,、 ')
                if answer and answer[-1] not in '。！？.!?':
                    answer += '。'
        self.scope = advance_scope(self.scope, targets, selected)
        self.pending = pending_after_lookup(catalog, targets, selected, reasons)
        if (self.composer is not None and len(self.scope) == len(selected) == 2
                and self.scope[0].name == self.scope[1].name
                and {address.field for address in self.scope} in COMPOSE_PAIRS):
            self.compose_selection = tuple(zip(self.scope, selected, strict=True))
        self.history = [*self.history, {'role': 'user', 'content': question},
                        {'role': 'assistant', 'content': answer}][-4:]
        return {'kind': kind, 'answer': answer, 'selected': selected, 'reasons': list(reasons)}

    def recall_awake(self, recent):
        self.reset_dialogue()
        if recent is None:
            return {'kind': 'clarify', 'answer': '请告诉我要查谁的哪项记录和版次。', 'selected': []}
        number, field, _, _ = recent
        label = FIELDS[field]
        selected, output = reply_to_route(self.model, self.tokenizer, self.store, self.codebook, self.reader,
                                          f'请把刚才查到的{label}只报名称。', [number], labels=[label])
        return {'kind': 'read', 'answer': output['answer'], 'selected': selected, 'reasons': ['awake_recall']}

    def dispatch(self, line):
        if line == '/帮助':
            help_text = [HELP]
            if self.composer is not None:
                help_text.append(COMPOSE_HELP)
            if self.awake is not None:
                help_text.append(AWAKE_HELP)
            if self.spoken is not None:
                help_text.append('口语查阅会先询问人物和事项。确认后可回复「是，上一版」。')
            return {'kind': 'help', 'answer': '\n'.join(help_text)}
        if line == '/清空对话':
            self.reset_dialogue()
            return {'kind': 'reset', 'answer': '已清空本轮对话，记忆记录已保留。'}
        if line.startswith('/记 '):
            return self.save(line[3:], date.today())
        if line.startswith('/成句 '):
            return self.compose(line[4:], date.today())
        if line.startswith('/'):
            raise ValueError('命令未识别。输入 /帮助 查看用法。')
        return self.turn(line, date.today())


def artifacts(reader_dir, normalizer_dir):
    reader, normalizer, writer = reader_dir / 'head.pt', normalizer_dir / 'normalizers.pt', SOURCE / 'adapters.pt'
    reader_report = json.loads((reader_dir / 'result.json').read_text(encoding='utf-8'))
    normalizer_report = json.loads((normalizer_dir / 'guarded.json').read_text(encoding='utf-8'))
    paths = (base.MODEL / 'model.safetensors', base.MODEL / 'tokenizer.json', base.MODEL / 'tokenizer_config.json', writer)
    sources = {path.name: base.sha(path) for path in paths}
    expected = {Path(path).name: digest for path, digest in reader_report['sources'].items()}
    reader_digest = base.sha(reader)
    if (sources['model.safetensors'] != base.WEIGHT_SHA or any(expected.get(file) != digest for file, digest in sources.items())
            or reader_digest != reader_report['head_sha256']
            or reader_digest not in (digest for path, digest in normalizer_report['sources'].items() if Path(path).name == 'head.pt')
            or base.sha(normalizer) != normalizer_report['normalizer_sha256']):
        raise ValueError('本机模型、记忆读出与口语核验文件不匹配。')
    binding = sha256(json.dumps({'codec': 'lexical-fast-weights-v1', 'sources': sources}, sort_keys=True).encode()).hexdigest()
    return reader, normalizer, writer, binding


def composition_head(directory, width, device):
    report = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
    candidate = directory / 'typed_time.pt'
    if base.sha(candidate) != report['head_sha256']['typed_time']:
        raise ValueError('成句模型与评测文件不匹配。')
    (control_path, control_digest), = ((path, digest) for path, digest in report['sources'].items()
                                     if Path(path).name == 'result.json')
    control = Path(control_path)
    if base.sha(control) != control_digest:
        raise ValueError('成句模型的底座评测文件不匹配。')
    parent = json.loads(control.read_text(encoding='utf-8'))['sources']
    expected_sources = {'model.safetensors': base.WEIGHT_SHA, 'tokenizer.json': base.sha(base.MODEL / 'tokenizer.json')}
    for name, digest in expected_sources.items():
        if not any(Path(path).name == name and value == digest for path, value in parent.items()):
            raise ValueError('成句模型的底座或分词器不匹配。')
    state = torch.load(candidate, map_location='cpu', weights_only=True)
    head = VersionReplyActions(width, state['scaffold_tokens'], slots=len(SLOTS))
    head.load_state_dict(state)
    return head.eval().to(device)


@contextmanager
def local_model(recurrent):
    if recurrent is None:
        yield load_full_streamed()
        return
    from scripts.benchmark_spora_recurrent_dialogue import extend_positions, using

    checkpoint, *layers = recurrent
    model = load_full_streamed(stage_output_projection=True)
    rotary = model.model.rotary_emb
    runtime = load_integrated_runtime(checkpoint, model, base.MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    limit = 2048
    if decoder.position_table_max_position is None or decoder.position_table_max_position > limit:
        raise ValueError('循环版需要经验证的位置表。')
    if decoder.position_table_max_position < limit:
        extend_positions(decoder, rotary, limit)
    replacements = load_recurrent_replacements(decoder, layers, base.sha(checkpoint))
    if tuple(replacements) != (19, 22):
        raise ValueError('请按顺序提供第 19、22 层循环注意力权重。')
    torch.cuda.empty_cache()
    model.lm_head.to(model.device)

    def check_position(_module, inputs):
        if int(inputs[1].amax()) >= limit:
            raise ValueError('循环版对话超出 2048 词元上限，请清空对话或缩短消息。')

    hook = model.model.rotary_emb.register_forward_pre_hook(check_position)
    try:
        with using(replacements, 'recurrent'):
            yield model
    finally:
        hook.remove()


def main(args):
    torch.set_num_threads(2)
    interactive = sys.stdin.isatty() and not args.json
    if interactive:
        print('正在校验并加载本机模型，首次打开可能需要约一分钟。', flush=True)
    reader_path, normalizer_path, writer_path, binding = artifacts(args.reader, args.normalizer)
    with local_model(args.recurrent) as model:
        run_chat(args, model, reader_path, normalizer_path, writer_path, binding, interactive)


def spoken_lookup_planner(folder, recurrent, model, tokenizer):
    from scripts.audit_spora_spoken_query_plan import DEPTH, QueryPlanProbe, predict, proposed_address, vectors

    report = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    candidate = folder / 'public.pt'
    identifiers = {Path(path).resolve(): digest for path, digest in report['sources'].items()}
    sources = [recurrent[0], *(path / 'attention.pt' for path in recurrent[1:])]
    if (base.sha(candidate) != report['results']['public']['probe_sha256']
            or any(identifiers.get(path.resolve()) != base.sha(path) for path in sources)):
        raise ValueError('口语查阅模型与循环骨干不匹配。')
    cutoff = report.get('address_proposal', {}).get('threshold', report['results']['public']['threshold'])
    if not isinstance(cutoff, (int, float)) or not 0 <= cutoff <= 1.000001:
        raise ValueError('口语查阅阈值不合法。')
    head = QueryPlanProbe(model.config.hidden_size).eval()
    head.load_state_dict(torch.load(candidate, map_location='cpu', weights_only=True))
    extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=DEPTH)

    def propose(question, enrolled):
        if not any(name in question for name in enrolled):
            return None
        # Multiple mentioned people are scored by the same head used in the audit.
        if sum(name in question for name in enrolled) > 2:
            return None
        feature, (names,) = vectors(extract, [question], enrolled, 'user-span')
        (scored,) = predict(head, feature)
        return proposed_address(scored, names, cutoff)

    return propose


def run_chat(args, model, reader_path, normalizer_path, writer_path, binding, interactive):
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    width = model.config.hidden_size
    state = torch.load(reader_path, map_location='cpu', weights_only=True)
    reader = VersionReplyActions(width, state['scaffold_tokens'], slots=2)
    reader.load_state_dict(state)
    composer = composition_head(args.composition, width, model.device) if args.composition is not None else None
    extractor = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=len(model.model.layers))
    proposer = TemporalProposals(extractor, width, torch.load(normalizer_path, map_location='cpu', weights_only=True)['final_layer'])
    spoken = (spoken_lookup_planner(args.spoken_lookup, args.recurrent, model, tokenizer)
              if args.spoken_lookup is not None else None)

    def propose(evidence):
        if args.recurrent is None:
            return proposer(evidence)
        return [proposer([text])[0] for text in evidence]

    settings = torch.load(writer_path, map_location='cpu', weights_only=True)
    args.profile.mkdir(parents=True, exist_ok=True)
    database = args.profile / 'versions.sqlite3'
    with VersionStore(database, binding, create=not database.exists()) as store, ExitStack() as resources:
        awake = None
        lines = terminal_lines(interactive)
        if args.awake_state is not None:
            path = args.awake_state.resolve()
            recurrent = ([args.recurrent[0], *(folder / 'attention.pt' for folder in args.recurrent[1:])]
                         if args.recurrent else [])
            protected = (database, args.awake_core, reader_path, normalizer_path, writer_path,
                         base.MODEL / 'model.safetensors', *recurrent)
            if args.spoken_lookup is not None:
                protected += (args.spoken_lookup / 'public.pt', args.spoken_lookup / 'result.json')
            if path in {file.resolve() for file in protected}:
                raise ValueError('在线状态需要使用独立文件。')
            individual = {'codec': binding, 'profile': str(database.resolve()),
                          'backbone': [base.sha(file) for file in recurrent]}

            def report(status):
                if args.json:
                    print(json.dumps({'kind': 'awake', 'status': status}, ensure_ascii=False), flush=True)
                else:
                    print(f"孢子> {status['mode']}，已执行 {status['neural_steps']} 次内部更新。", flush=True)

            lines, observe, consume, bind = resources.enter_context(awake_chat_lines(
                model, sys.stdin, path, args.awake_core, individual, report, topic_labels=FIELDS,
                clear_topics_on_sleep=True, core_device='cpu',
            ))
            awake = (observe, consume, bind)
        chat = VersionChat(store, model, tokenizer, reader.eval().to(model.device), propose, settings,
                           ordinary_tokens=args.reply_tokens, composer=composer, typed_lookup=args.recurrent is not None,
                           awake=awake, spoken=spoken)
        if interactive:
            backend = '循环注意力版' if args.recurrent else '原始注意力版'
            print(f'孢子版本记忆已就绪。当前使用{backend}。输入 /帮助 查看用法。', flush=True)
        for raw in lines:
            line = raw.strip()
            if line == '/退出':
                break
            if not line:
                continue
            try:
                result = chat.dispatch(line)
            except (ValueError, sqlite3.Error) as error:
                result = {'kind': 'error', 'answer': str(error)}
            print(json.dumps(result, ensure_ascii=False) if args.json else f"孢子> {result['answer']}", flush=True)


def terminal_lines(interactive):
    while True:
        try:
            yield input('你> ' if interactive else '')
        except (EOFError, KeyboardInterrupt):
            return


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--profile', type=Path, default=Path('runs/spora-version-chat-local'))
    cli.add_argument('--reader', type=Path, default=Path('runs/spora-dual-version-reply-v1'))
    cli.add_argument('--normalizer', type=Path, default=Path('runs/spora-version-normalizer-v1'))
    cli.add_argument('--composition', type=Path, help='Enable the audited typed composition head')
    cli.add_argument('--recurrent', nargs=3, type=Path, metavar=('CHECKPOINT', 'LAYER19', 'LAYER22'),
                     help='Load the integrated recurrent SPORA checkpoint and both layer directories')
    cli.add_argument('--awake-state', type=Path, help='Run and persist the online CPU state')
    cli.add_argument('--awake-core', type=Path, default=Path('runs/p24-awake/core-seed7.pt'))
    cli.add_argument('--spoken-lookup', type=Path, help='Propose a memory address for explicit confirmation')
    cli.add_argument('--reply-tokens', type=int, default=72)
    cli.add_argument('--json', action='store_true', help='Emit one JSON record per stdin line')
    options = cli.parse_args()
    if not 1 <= options.reply_tokens <= 256:
        cli.error('--reply-tokens must be between 1 and 256')
    if options.spoken_lookup is not None and options.recurrent is None:
        cli.error('--spoken-lookup requires the supported recurrent weights')
    try:
        main(options)
    except (OSError, ValueError, KeyError, sqlite3.Error) as error:
        cli.exit(1, f'孢子未能启动。{error}\n')
