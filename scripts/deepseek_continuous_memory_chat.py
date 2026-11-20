"""Interactive chat and explicit writes for the continuous-memory research checkpoint."""

import argparse
import json
import os
import sys
from collections import deque
from contextlib import ExitStack, nullcontext
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from deepseek_aligned_memory_calibration import load_adapter
from deepseek_aligned_memory_dialogue import (
    EARLY_TIMING_THRESHOLD, OUT as ADAPTER_RUN, RANK, THRESHOLD, generate_memory_suffix, generate_with_memory,
)
from deepseek_memory_attribute_router import CLARIFY, MISSING
from deepseek_memory_attribute_span import KEYS, Route, route_action
from deepseek_memory_cosine_readout import cosine_bias, cosine_slot_bias, projected_codebook
from deepseek_memory_dialogue_context import OUT as DIALOGUE_RUN, PromptPositionLimit, features_for, route
from deepseek_memory_field_decoding import FIELDS, OUT as FIELD_RUN
from deepseek_memory_field_transfer import OUT as TRANSFER_RUN
from deepseek_memory_inheritance_gate import OUT as INHERITANCE_GATE_RUN
from deepseek_memory_inherited_turn import OUT as INHERITED_RUN
from deepseek_memory_turn_priority import OUT as PRIORITY_RUN, gate_probabilities
from deepseek_output_aligned_memory import MODEL, SEED, SLOTS, SOURCE, encode, load_integrated, read, read_bank, sha, torch, write_bank
from spora.awake_chat import awake_chat_lines
from spora.awake_recall_intent import SUBJECT_SCOPE_CONFLICT, OrdinaryIntent, RecallIntent, SameSubjectIntent, SelfNameIntent
from spora.continuous_bank_store import (
    bank_revision, commit_banks, read_archived_slots, read_bank_tensor, read_banks, read_recent_snapshots, read_slots_as_of,
)
from spora.durable import RevisionConflict, pack
from spora.event_ledger import EventLedger
from spora.language_training import atomic_save
from spora.online_time import OnlineSession, OnlineState
from spora.research_continuous_observation import (
    ADDRESSES, ELLIPTICAL_QUERY, can_register, inherited_query_subject, parse_observation, query_subject, split_history_query,
)
from spora.structure_transfer import SPORALayerReplacement
from spora.working_checkpoint_lock import lock_working_checkpoint
from spora.world_time import TimeWorld

ENTITIES = tuple(ADDRESSES.values())
ASK_SUBJECT = '请明确你想查询谁的记录。'
WORKING_BANKS = TRANSFER_RUN / 'working-banks.pt'
LEGACY_WORKING_FORMAT = 'spora-continuous-chat-banks-v1'
WORKING_FORMAT = 'spora-continuous-chat-banks-v2'
PAGED_FORMAT = 'spora-continuous-chat-banks-v3'
SEED_FILLED = frozenset((entity, field, version) for entity in ENTITIES for field in FIELDS for version in (0, 1))
UTC_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
DEMO_WORLD_ACTIONS = ((0., 0.), (.5, 0.), (-.5, 0.), (0., .5), (0., -.5))
ORDINARY_TURNS = 4
ORDINARY_NEW_TOKENS = 64


def ordinary_context_ids(tokenizer, history, question, cap, current_ids):
    turns = list(history)[-ORDINARY_TURNS:]
    while turns:
        messages = [dict(role=role, content=content) for user, answer in turns
                    for role, content in (('user', user), ('assistant', answer))]
        messages.append(dict(role='user', content=question))
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(prompt + '</think>\n\n', add_special_tokens=False)
        if len(ids) <= cap:
            return ids, len(turns)
        turns.pop(0)
    return current_ids, 0


def parse_as_of(text):
    try:
        value = datetime.fromisoformat(text)
        if value.tzinfo is None:
            raise ValueError('Timezone required')
        return (value - UTC_EPOCH) // timedelta(microseconds=1)
    except (OverflowError, ValueError) as error:
        raise argparse.ArgumentTypeError('Use an ISO 8601 timestamp with Z or a UTC offset') from error


class InvalidMemoryValue(ValueError):
    """The submitted value cannot fit the continuous-memory slots."""


class MemoryUpdateRejected(ValueError):
    """A candidate failed verification before the memory transaction."""


def uses_database(path):
    return path.suffix.lower() == '.sqlite3'


def matches_memory_suffix(tokens, bias, eos):
    return (1 < len(tokens) <= len(bias) and tokens[-1] == eos
            and bias[:len(tokens)].argmax(dim=-1).tolist() == tokens)


def bank_payload(banks, seed_sha, addresses, filled, pages, page_size):
    payload = dict(format=WORKING_FORMAT, seed_sha256=seed_sha, banks=banks,
                   addresses=addresses, filled=sorted(filled))
    if pages:
        payload.update(format=PAGED_FORMAT, pages=list(pages), page_size=page_size)
    return payload


def load_working_banks(path, seed_banks, seed_sha, *, lazy=False):
    tracked = {}
    if not path.exists():
        return seed_banks, None, dict(ADDRESSES), SEED_FILLED, (), None, tracked
    if uses_database(path):
        payload, digest, tracked = read_banks(path.resolve(), load_tensors=not lazy)
    else:
        digest = sha(path)
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if sha(path) != digest:
            raise RevisionConflict('Continuous memory changed during load; retry')
    checkpoint_format = payload.get('format')
    if checkpoint_format not in (LEGACY_WORKING_FORMAT, WORKING_FORMAT, PAGED_FORMAT) or payload.get('seed_sha256') != seed_sha:
        raise ValueError('Working banks have incompatible provenance')
    banks = payload['banks']
    if set(banks) != set(seed_banks):
        raise ValueError('Working banks have incompatible fields or versions')
    for key, original in seed_banks.items():
        bank = banks[key]
        if bank is None and lazy and tracked:
            if key[1] == 0 and tracked[0, *key] != pack(original)[-1]:
                raise ValueError('Invalid original bank digest')
        elif (bank.shape != original.shape or bank.dtype != original.dtype or not bool(torch.isfinite(bank).all())
              or (key[1] == 0 and not torch.equal(bank, original))):
            raise ValueError('Invalid working bank')
    addresses, filled = dict(ADDRESSES), SEED_FILLED
    pages, page_size = (), None
    if checkpoint_format != LEGACY_WORKING_FORMAT:
        registered = payload['addresses']
        if any(registered.get(name) != key for name, key in ADDRESSES.items()):
            raise ValueError('Seed subjects cannot change address')
        for name, key in registered.items():
            if name in ADDRESSES:
                continue
            if not can_register(name, addresses) or type(key) is not int or key < 0 or key in addresses.values():
                raise ValueError('Invalid registered subject')
            addresses[name] = key
        filled = frozenset(map(tuple, payload['filled']))
        allowed = SEED_FILLED | {(entity, field, 1) for entity in addresses.values() for field in FIELDS}
        if not SEED_FILLED <= filled <= allowed:
            raise ValueError('Invalid field availability')
    if checkpoint_format == PAGED_FORMAT:
        pages, page_size = tuple(payload['pages']), payload['page_size']
        if type(page_size) is not int or page_size <= max(ENTITIES) or not pages:
            raise ValueError('Invalid memory page layout')
        used = {entity // page_size for entity in addresses.values()}
        if used != set(range(len(pages) + 1)):
            raise ValueError('Registered addresses and memory pages do not match')
        for index, page in enumerate(pages, start=1):
            fields = {field for entity, field, version in filled if entity // page_size == index}
            if set(page) != fields:
                raise ValueError('Paged memory fields and occupancy do not match')
            for field, bank in page.items():
                if bank is None and lazy and tracked:
                    continue
                template = seed_banks[field, 1]
                if bank.shape != template.shape or bank.dtype != template.dtype or not bool(torch.isfinite(bank).all()):
                    raise ValueError('Invalid paged bank')
    return banks, digest, addresses, filled, pages, page_size, tracked


class ContinuousMemoryChat:
    def __init__(self, decoder, tokenizer, site, banks, styles, timing, columns, contextual, direct, inheritance, setup,
                 *, working_path, working_sha, seed_sha, addresses, filled, pages=(), saved_page_size=None,
                 inherited_router=None, gate_cutoff=None, bank_digests=None, bank_specs=None):
        self.decoder, self.tokenizer, self.site, self.banks = decoder, tokenizer, site, banks
        self.styles, self.timing, self.columns = styles, timing, columns
        self.contextual, self.direct, self.inheritance = contextual, direct, inheritance
        self.inherited_router = inherited_router
        self.recall_intent = None
        self.same_subject_intent = None
        self.ordinary_intent = None
        self.subject_hops = 1
        self.self_name = None
        self.gate_cutoff = setup['gate_cutoff'] if gate_cutoff is None else gate_cutoff
        self.cutoff = setup['cutoffs']['priority']
        self.working_path, self.working_sha, self.seed_sha = working_path, working_sha, seed_sha
        self.bank_digests = bank_digests if bank_digests is not None else {}
        self.bank_specs = bank_specs
        self.page_size = site.current_basis.shape[0]
        if saved_page_size is not None and saved_page_size != self.page_size:
            raise ValueError('Saved page size does not match the address basis')
        if max(addresses.values()) >= self.page_size * (1 + len(pages)):
            raise ValueError('Registered subject is outside the address basis')
        self.addresses, self.filled = addresses, filled
        self.pages = pages
        layers = decoder.base.model.layers
        self.position_limit = min((layer.max_length for layer in layers if isinstance(layer, SPORALayerReplacement)),
                                  default=decoder.base.config.max_position_embeddings)
        weight = decoder.base.lm_head.weight
        basis = torch.linalg.qr(torch.randn(weight.shape[1], RANK, generator=torch.Generator().manual_seed(SEED))).Q
        self.codes, self.scale = projected_codebook(weight, basis.to(weight.device))

    def _read_tokens(self, bank, entity):
        return self._read_address_tokens(bank, (entity,))[0]

    def _local_address(self, entity):
        slot = entity % self.page_size
        return self.site.current_basis[slot:slot + 1]

    def bank_for(self, entity, field, version):
        page = entity // self.page_size
        if page == 0:
            bank = self.banks.get((field, version))
        else:
            bank = self.pages[page - 1].get(field) if version == 1 else None
        key = page, field, version
        if bank is None and key in self.bank_digests:
            bank = read_bank_tensor(self.working_path, key, self.working_sha, self.bank_digests[key])
            if (bank.shape, bank.dtype) != self.bank_specs[field, version]:
                raise ValueError('Stored bank does not match the model layout')
        return bank

    def _read_address_tokens(self, bank, entities):
        results = []
        # Bound the temporary vocabulary scores independently of address capacity.
        for start in range(0, len(entities), 8):
            group = entities[start:start + 8]
            slots = torch.stack([read_bank(self.site, bank, self._local_address(key))[:, :RANK]
                                 for key in group]).to(self.codes.device)
            directions = torch.nn.functional.normalize(slots.flatten(0, 1), dim=-1)
            tokens = (directions @ self.codes.T).argmax(dim=-1).reshape(len(group), SLOTS)
            results.extend(tokens.masked_fill(slots.norm(dim=-1) < .2, -1).tolist())
        return results

    def _check_revision(self):
        if uses_database(self.working_path):
            current = bank_revision(self.working_path)
        else:
            current = sha(self.working_path) if self.working_path.exists() else None
        if current != self.working_sha:
            raise RevisionConflict('Continuous memory changed; reload before writing')

    def _save(self, banks, addresses, filled, pages, *, changed=(), archive=None, recorded=None, repairs=()):
        payload = bank_payload(banks, self.seed_sha, addresses, filled, pages, self.page_size)
        if uses_database(self.working_path):
            self.working_sha, self.bank_digests, _ = commit_banks(
                self.working_path, payload, self.working_sha, self.bank_digests, changed=changed, archive=archive,
                recorded=recorded, repairs=repairs)
            banks = dict.fromkeys(banks)
            pages = tuple(dict.fromkeys(page) for page in pages)
        else:
            atomic_save(payload, self.working_path)
            self.working_sha = sha(self.working_path)
        self.banks, self.addresses, self.filled = banks, addresses, filled
        self.pages = pages

    def register(self, name, *, expand=False):
        name = name.strip()
        with lock_working_checkpoint(self.working_path):
            self._check_revision()
            if name in self.addresses:
                return dict(action='already_registered', entity=self.addresses[name], name=name)
            if not can_register(name, self.addresses):
                return dict(action='defer', reason='invalid_or_overlapping_name')
            used = set(self.addresses.values())
            capacity = self.page_size * (1 + len(self.pages))
            entity = next((key for key in range(capacity) if key not in used), None)
            pages = self.pages
            if entity is None:
                if not expand:
                    return dict(action='defer', reason='no_free_address')
                entity, pages = capacity, (*pages, {})
            self._save(self.banks, {**self.addresses, name: entity}, self.filled, pages)
        return dict(action='registered', entity=entity, name=name)

    @torch.inference_mode()
    def remember(self, entity, field, value):
        if entity not in self.addresses.values() or field not in FIELDS:
            raise ValueError('Unknown experimental address or field')
        tokens = encode(self.tokenizer, value.strip())
        if not 1 <= len(tokens) < SLOTS or set(tokens) & set(self.tokenizer.all_special_ids):
            raise InvalidMemoryValue(f'Value must have 1 to {SLOTS - 1} non-special tokens')
        tokens.append(self.tokenizer.eos_token_id)
        page = entity // self.page_size
        record = entity, field, 1
        expected = tokens + [-1] * (SLOTS - len(tokens))
        with lock_working_checkpoint(self.working_path):
            self._check_revision()
            bank = self.bank_for(entity, field, 1)
            if bank is None:
                shape, dtype = self.bank_specs[field, 1]
                bank = torch.zeros(shape, dtype=dtype)
            if record in self.filled and self._read_tokens(bank, entity) == expected:
                return dict(action='already_stored', entity=entity, field=field)
            target = torch.zeros_like(bank[:, :, 0])
            target[:len(tokens), :RANK] = self.codes[tokens].cpu()
            candidate = write_bank(self.site, bank, self._local_address(entity), target)
            if self._read_tokens(candidate, entity) != expected:
                raise MemoryUpdateRejected('Candidate matrix did not reproduce the written value')
            neighbors = sorted(other for other, other_field, version in self.filled
                               if other != entity and other_field == field and version == 1 and other // self.page_size == page)
            if self._read_address_tokens(candidate, neighbors) != self._read_address_tokens(bank, neighbors):
                raise MemoryUpdateRejected('Candidate matrix changed another occupied address')
            updated = dict(self.banks)
            pages = list(self.pages)
            if page == 0:
                updated[field, 1] = candidate
            else:
                pages[page - 1] = {**pages[page - 1], field: candidate}
            archive = None
            recorded = None
            if uses_database(self.working_path):
                address = self._local_address(entity)
                recorded = entity, field, read_bank(self.site, candidate, address)[:, :RANK]
                if record in self.filled:
                    archive = entity, field, read_bank(self.site, bank, address)[:, :RANK]
            self._save(updated, self.addresses, self.filled | {record}, tuple(pages),
                       changed=((page, field, 1),), archive=archive, recorded=recorded)
        return dict(action='stored', entity=entity, field=field)

    def observe(self, statement, *, entity=None):
        candidate = parse_observation(statement, self.addresses)
        if candidate is None:
            return dict(action='defer', reason='unsupported_or_ambiguous_statement')
        subject, field, value = candidate
        if entity is not None and entity != subject:
            return dict(action='defer', reason='subject_mismatch')
        try:
            return self.remember(subject, field, value)
        except InvalidMemoryValue:
            return dict(action='defer', reason='value_does_not_fit')
        except MemoryUpdateRejected:
            return dict(action='defer', reason='matrix_verification_failed')

    @torch.inference_mode()
    def verify_recent_memory(self, limit, *, repair=False):
        if not uses_database(self.working_path):
            raise ValueError('Replay verification requires a SQLite memory file')
        with lock_working_checkpoint(self.working_path) if repair else nullcontext():
            return self._verify_recent_memory(limit, repair)

    def _verify_recent_memory(self, limit, repair):
        if self.working_sha is None:
            self._check_revision()
            return dict(action='replay_verified', checked=0, mismatches=[], revision=None)
        snapshots = read_recent_snapshots(self.working_path, self.working_sha, limit)
        mismatches = []
        targets = {}
        eos = self.tokenizer.eos_token_id
        for entity, field, revision, slots in snapshots:
            evidence = dict(entity=entity, field=field, write_revision=revision)
            if entity not in self.addresses.values() or (entity, field, 1) not in self.filled:
                mismatches.append(dict(evidence, reason='address_missing'))
                continue
            if slots.shape != (SLOTS, RANK) or slots.dtype != self.bank_specs[field, 1][1]:
                mismatches.append(dict(evidence, reason='snapshot_layout'))
                continue
            if not bool(torch.isfinite(slots).all()):
                mismatches.append(dict(evidence, reason='snapshot_not_finite'))
                continue
            bias = cosine_slot_bias(slots, self.codes, self.scale)
            present = slots.norm(dim=-1) >= .2
            reference = bias.argmax(dim=-1).masked_fill(~present.to(bias.device), -1).tolist() if bias is not None else []
            end = reference.index(eos) + 1 if eos in reference else 0
            special = set(self.tokenizer.all_special_ids)
            if (end < 2 or any(token < 0 or token in special for token in reference[:end - 1])
                    or any(token != -1 for token in reference[end:])):
                mismatches.append(dict(evidence, reason='snapshot_missing_value'))
                continue
            bank = self.bank_for(entity, field, 1)
            actual = self._read_tokens(bank, entity) if bank is not None else []
            if actual != reference:
                mismatches.append(dict(evidence, reason='current_value_differs'))
                if repair and bank is not None and bool(torch.isfinite(bank).all()):
                    targets[entity, field] = (revision, slots, reference)
        self._check_revision()
        outcome = dict(action='replay_mismatch' if mismatches else 'replay_verified', checked=len(snapshots),
                       mismatches=mismatches, revision=self.working_sha)
        if not repair or not mismatches:
            return outcome
        if len(targets) != len(mismatches):
            return dict(outcome, action='defer', reason='unusable_replay_evidence')
        try:
            self._repair_replayed_memory(targets)
        except MemoryUpdateRejected:
            return dict(outcome, action='defer', reason='repair_verification_failed')
        return dict(outcome, action='replay_repaired', repaired=len(targets), revision=self.working_sha)

    def _repair_replayed_memory(self, targets):
        groups = {}
        for (entity, field), item in targets.items():
            groups.setdefault((entity // self.page_size, field), {})[entity] = item
        banks = dict(self.banks)
        pages = [dict(page) for page in self.pages]
        for (page, field), records in groups.items():
            original = self.bank_for(next(iter(records)), field, 1)
            repaired = original
            for entity, (_, slots, _) in records.items():
                target = torch.zeros_like(original[:, :, 0])
                target[:, :RANK] = slots
                repaired = write_bank(self.site, repaired, self._local_address(entity), target)
            neighbors = sorted(entity for entity, relation, version in self.filled
                               if relation == field and version == 1 and entity // self.page_size == page
                               and entity not in records)
            if (any(self._read_tokens(repaired, entity) != record[2] for entity, record in records.items())
                    or self._read_address_tokens(repaired, neighbors) != self._read_address_tokens(original, neighbors)):
                raise MemoryUpdateRejected('Replay repair did not preserve the bank values')
            if page == 0:
                banks[field, 1] = repaired
            else:
                pages[page - 1][field] = repaired
        changed = tuple((page, field, 1) for page, field in groups)
        repairs = tuple((entity, field, item[0]) for (entity, field), item in targets.items())
        self._save(banks, self.addresses, self.filled, tuple(pages), changed=changed, repairs=repairs)

    def _answer_self_name(self, prompt, confirm_recent_read):
        bias = cosine_bias(self.site, self.self_name.name_bank, self._local_address(0), self.codes, self.scale)
        result = dict(response=CLARIFY, action='clarify', reason='self_name_unconfirmed', field='self_name',
                      confidence=None, inherits=False, memory_start=None, entity=None, version=1,
                      subject_reason='self_name')
        if confirm_recent_read:
            result['confirmed_recent_read'] = False
        if bias is not None:
            ids = torch.tensor([prompt], device=bias.device)
            available = self.position_limit - len(prompt)
            tokens = generate_memory_suffix(self.decoder, self.tokenizer, ids, 2 * bias, min(SLOTS, available))
            if matches_memory_suffix(tokens, bias, self.tokenizer.eos_token_id):
                result.update(response=self.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                              action='read', reason='read', memory_start=0)
        return result

    @torch.inference_mode()
    def answer(self, question, entity=None, version=1, history=None, *, timing_threshold=THRESHOLD, back=0, as_of=None,
               topic_observer=None, consume_topic=None, bind_topic=None, ordinary_history=()):
        if (entity is not None and entity not in self.addresses.values()) or version not in (0, 1):
            raise ValueError('Unknown experimental address or snapshot')
        if type(back) is not int or not 0 <= back <= 2**63 - 1 or (back and (version != 1 or not uses_database(self.working_path))):
            raise ValueError('Historical recall requires a SQLite checkpoint and version 1')
        if as_of is not None and (back or version != 1 or not uses_database(self.working_path)):
            raise ValueError('Timestamp recall requires a version-1 SQLite query without --back')
        recent_read = consume_topic() if consume_topic is not None else None
        as_of_utc = (UTC_EPOCH + timedelta(microseconds=as_of)).isoformat() if as_of is not None else None
        question, requested_back = split_history_query(question)
        history_issue = None
        if requested_back is None:
            history_issue = 'invalid_history_query'
        elif requested_back:
            if as_of is not None or (back and requested_back != back):
                history_issue = 'conflicting_history_offsets'
            elif version != 1 or not uses_database(self.working_path):
                history_issue = 'history_requires_sqlite'
            else:
                back = requested_back
        if history:
            history, _ = split_history_query(history)
        row = dict(key='input', question=question, history=history, target_turn=0)
        try:
            features = features_for(self.decoder, self.tokenizer, row, False, position_limit=self.position_limit - 1)
        except PromptPositionLimit:
            return dict(response='这条消息太长，请拆成几段发给我。', action='clarify', reason='position_limit', field=None,
                        confidence=None, inherits=False, memory_start=None, entity=entity, version=version,
                        subject_reason='no_subject')
        if (self.self_name is not None and entity is None and version == 1 and not back and as_of is None
                and history_issue is None and self.self_name.targets_self(question) and self.self_name(*features[:2])):
            return self._answer_self_name(features[3], consume_topic is not None)
        probability, = gate_probabilities(self.inheritance, [row], {'input': features})
        inherits = bool(history) and probability >= self.gate_cutoff
        subject_reason = 'specified_subject'
        if entity is None:
            entity, subject_reason = query_subject(question, self.addresses)
            if entity is not None and history:
                previous, _ = query_subject(history, self.addresses)
                if previous != entity:
                    inherits = False
        elliptical = False
        learned_repeat = False
        scope_conflict = False
        if consume_topic is not None and subject_reason == 'no_subject':
            elliptical = ELLIPTICAL_QUERY.fullmatch(question.strip()) is not None
            scope_conflict = self.same_subject_intent is not None and SUBJECT_SCOPE_CONFLICT.search(question) is not None
            learned_repeat = self.recall_intent is not None and self.recall_intent(*features[:2])
        eligible_binding = (recent_read is not None and version == 1 and not back and as_of is None
                            and history_issue is None and not scope_conflict)
        use_topic = eligible_binding and ((elliptical and probability >= self.gate_cutoff) or learned_repeat)
        use_subject = False
        if use_topic:
            entity, selected, confidence, _ = recent_read
            subject_reason = 'awake_recent_read'
            inherits = False
            predicted = dict(selected=selected, confidence=confidence)
        else:
            if inherits:
                try:
                    features = features_for(self.decoder, self.tokenizer, row, True, position_limit=self.position_limit - 1)
                except PromptPositionLimit:
                    inherits = False
            if subject_reason == 'no_subject' and inherits:
                entity = inherited_query_subject(question, history, self.addresses)
                if entity is not None:
                    subject_reason = 'inherited_subject'
            router = self.contextual if inherits else self.direct
            route_features = features
            if inherits and self.inherited_router is not None:
                router = self.inherited_router
                route_features = features_for(self.decoder, self.tokenizer, dict(question=history), False)
            predicted, = route(router, [row], {'input': route_features})
            if (eligible_binding and entity is None and subject_reason == 'no_subject' and self.same_subject_intent is not None
                    and route_action(predicted, self.cutoff) == 'read' and self.same_subject_intent(*features[:2])):
                entity = recent_read[0]
                subject_reason = 'awake_same_subject'
                use_subject = True
            if topic_observer is not None and not (elliptical or learned_repeat):
                topic_observer(predicted['probabilities'])
        selected = predicted['selected']
        action = route_action(predicted, self.cutoff)
        if (action != 'ordinary' and self.ordinary_intent is not None and subject_reason == 'no_subject'
                and not (inherits or elliptical or learned_repeat or requested_back or back or scope_conflict)
                and as_of is None and history_issue is None and self.ordinary_intent(*features[:2])):
            selected, action = 'ordinary', 'ordinary'
        start = None
        confirmed = False
        context_turns = 0
        reason = action
        if history_issue is not None:
            action, reason, response = 'clarify', history_issue, CLARIFY
        elif (elliptical or learned_repeat) and entity is None:
            action, reason, response = 'clarify', 'awake_topic_unavailable', ASK_SUBJECT
        elif entity is None and (requested_back or as_of is not None or action in ('read', 'missing') or subject_reason != 'no_subject'):
            action, reason, response = 'clarify', subject_reason, ASK_SUBJECT
        elif (requested_back or as_of is not None) and action == 'ordinary':
            action, reason, response = 'clarify', 'history_field_unresolved', CLARIFY
        elif action == 'read' and (entity, selected, version) not in self.filled:
            action, reason, response = 'missing', 'unwritten_field', MISSING
        elif action == 'read':
            missing_reason = 'empty_matrix'
            if back or as_of is not None:
                archive = None
                if self.working_sha is not None:
                    if as_of is None:
                        archive = read_archived_slots(self.working_path, entity, selected, self.working_sha, back)
                    else:
                        archive = read_slots_as_of(self.working_path, entity, selected, self.working_sha, as_of)
                if archive is not None and (archive.shape != (SLOTS, RANK) or archive.dtype != self.bank_specs[selected, 1][1]):
                    raise ValueError('Archived memory shape or dtype differs from the model layout')
                bias = cosine_slot_bias(archive, self.codes, self.scale) if archive is not None else None
                missing_reason = 'history_unavailable'
            else:
                bias = cosine_bias(self.site, self.bank_for(entity, selected, version), self._local_address(entity),
                                   self.codes, self.scale)
            if bias is None:
                action, reason, response = 'missing', missing_reason, MISSING
            elif use_topic or use_subject:
                ids = torch.tensor([features[3]], device=bias.device)
                available = self.position_limit - len(features[3])
                tokens = generate_memory_suffix(self.decoder, self.tokenizer, ids, 2 * bias, min(SLOTS, available))
                if matches_memory_suffix(tokens, bias, self.tokenizer.eos_token_id):
                    start = 0
                    response = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    if (use_subject and bind_topic is not None and recent_read[3] > 0
                            and selected != recent_read[1]):
                        confirmed = bind_topic(entity, selected, recent_read[3])
                else:
                    action, reason, response = 'clarify', 'awake_read_unconfirmed', CLARIFY
            else:
                tokens, start = generate_with_memory(self.decoder, self.tokenizer, features[3], self.styles[selected],
                                                      self.columns, self.timing, 2 * bias,
                                                      max_new_tokens=min(32, self.position_limit - len(features[3])),
                                                      timing_threshold=timing_threshold)
                if start is None:
                    action, reason, response = 'clarify', 'injection_not_triggered', CLARIFY
                else:
                    if (bind_topic is not None and subject_reason == 'named_subject' and version == 1 and not back
                            and as_of is None and matches_memory_suffix(tokens[start:], bias, self.tokenizer.eos_token_id)):
                        confirmed = bind_topic(entity, selected, self.subject_hops)
                    response = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    if requested_back:
                        response = f'{requested_back} 次更正前的记录是「{response}」。'
                    elif as_of_utc is not None:
                        response = f'截至 {as_of_utc} 的记录是「{response}」。'
        elif action == 'ordinary':
            prompt, context_turns = ordinary_context_ids(self.tokenizer, ordinary_history, question,
                                                        self.position_limit - ORDINARY_NEW_TOKENS, features[3])
            available = self.position_limit - len(prompt)
            ids = torch.tensor([prompt], device=self.codes.device)
            tokens = self.decoder.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                           max_new_tokens=min(ORDINARY_NEW_TOKENS, available),
                                           pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
            response = self.tokenizer.decode(tokens[0, ids.shape[1]:], skip_special_tokens=True).strip()
        else:
            response = MISSING if action == 'missing' else CLARIFY
        result = dict(response=response, action=action, reason=reason, field=selected, confidence=predicted['confidence'],
                      inherits=inherits, memory_start=start, entity=entity, version=version, subject_reason=subject_reason)
        if back:
            result['corrections_back'] = back
        if as_of_utc is not None:
            result['as_of_utc'] = as_of_utc
        if consume_topic is not None:
            result['confirmed_recent_read'] = confirmed
        if action == 'ordinary':
            result['context_turns'] = context_turns
        return result


def load_chat(working_path=WORKING_BANKS, *, use_inherited_router=False, use_retrained_gate=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if use_retrained_gate and not use_inherited_router:
        raise ValueError('The retrained gate requires the inherited field router')
    paths = dict(adapter=ADAPTER_RUN / 'candidate.pt', styles=FIELD_RUN / 'field-adapters.pt',
                 context=DIALOGUE_RUN / 'router.pt', priority=PRIORITY_RUN / 'routing.pt',
                 routes=PRIORITY_RUN / 'routes.json', banks=TRANSFER_RUN / 'banks.pt')
    expected = {paths['adapter']: read(ADAPTER_RUN / 'result.json')['candidate_sha256']}
    for name in ('styles', 'context', 'priority', 'routes', 'banks'):
        path = paths[name]
        expected[path] = read(path.parent / 'result.json')['artifact_sha256'][path.name]
    if use_inherited_router:
        paths['previous'] = INHERITED_RUN / 'previous-router.pt'
        previous_digests = read(INHERITED_RUN / 'result.json')['artifact_sha256']
        previous_plan = INHERITED_RUN / 'plan.json'
        if sha(previous_plan) != previous_digests[previous_plan.name]:
            raise ValueError('Inherited router plan digest mismatch')
        expected[previous_plan] = previous_digests[previous_plan.name]
        expected[paths['previous']] = previous_digests[paths['previous'].name]
    gate_result = None
    if use_retrained_gate:
        paths['gate'] = INHERITANCE_GATE_RUN / 'gates.pt'
        gate_plan = INHERITANCE_GATE_RUN / 'plan.json'
        gate_result = read(INHERITANCE_GATE_RUN / 'result.json')
        digests = gate_result['artifact_sha256']
        expected.update({path: digests[path.name] for path in (paths['gate'], gate_plan)})
    model_files = (SOURCE, *(MODEL / name for name in ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json')))
    plan_path = TRANSFER_RUN / 'plan.json'
    if sha(plan_path) != read(TRANSFER_RUN / 'result.json')['artifact_sha256'][plan_path.name]:
        raise ValueError(f'Artifact digest mismatch: {plan_path}')
    provenance = read(plan_path)['protected']
    expected.update({path: provenance[str(path)] for path in model_files})
    if use_inherited_router:
        sources = read(previous_plan)['sources']
        if any(sources.get(str(path)) != expected[path] for path in model_files):
            raise ValueError('Inherited router uses incompatible model files')
    for path, digest in expected.items():
        if sha(path) != digest:
            raise ValueError(f'Artifact digest mismatch: {path}')
    if use_retrained_gate:
        sources = read(gate_plan)['sources']
        if any(sources.get(str(path)) != expected[path] for path in (*model_files, paths['previous'], paths['priority'])):
            raise ValueError('Retrained gate uses incompatible model or router files')
    working_path = Path(working_path).resolve()
    if working_path in {path.resolve() for path in (*expected, plan_path)}:
        raise ValueError('Working memory requires its own checkpoint path')
    seed_banks = torch.load(paths['banks'], weights_only=True)
    banks, working_sha, addresses, filled, pages, page_size, tracked = load_working_banks(
        working_path, seed_banks, expected[paths['banks']], lazy=uses_database(working_path))
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    load_device = 'cpu' if os.name == 'nt' else 'cuda'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
              dtype=torch.float16, attn_implementation='sdpa', device_map={'': load_device}, low_cpu_mem_usage=True)
    base = base.to('cuda').eval()
    decoder, organism, _ = load_integrated(SOURCE, base, MODEL, individual_device='cpu')
    decoder.eval().requires_grad_(False)
    width = base.lm_head.weight.shape[1]
    _, _, timing, columns = load_adapter(torch.load(paths['adapter'], weights_only=True), width)
    checkpoint = torch.load(paths['styles'], weights_only=True)
    styles = {}
    for field in FIELDS:
        style = torch.nn.Sequential(torch.nn.Linear(width, 24), torch.nn.Tanh(), torch.nn.Linear(24, len(columns)))
        style.load_state_dict(checkpoint[field]['style'])
        styles[field] = style.eval().requires_grad_(False).to('cuda')
    contextual = Route(width, localized=True).eval()
    contextual.load_state_dict(torch.load(paths['context'], weights_only=True))
    priority = torch.load(paths['priority'], weights_only=True)
    direct = Route(width, localized=True).eval()
    direct.load_state_dict(priority['direct'])
    inheritance = torch.nn.Linear(width, 1).eval()
    gate_state = (torch.load(paths['gate'], weights_only=True)['tail'] if use_retrained_gate else priority['inheritance'])
    inheritance.load_state_dict(gate_state)
    previous = None
    if use_inherited_router:
        previous = Route(width, localized=True).eval()
        previous.load_state_dict(torch.load(paths['previous'], weights_only=True))
    chat = ContinuousMemoryChat(decoder, tokenizer, organism.memories[0], banks, styles, timing, columns,
                                contextual, direct, inheritance, read(paths['routes']), working_path=working_path,
                                working_sha=working_sha, seed_sha=expected[paths['banks']], inherited_router=previous,
                                gate_cutoff=None if gate_result is None else gate_result['cutoffs']['tail'],
                                addresses=addresses, filled=filled, pages=pages, saved_page_size=page_size,
                                bank_digests=tracked, bank_specs={key: (bank.shape, bank.dtype) for key, bank in seed_banks.items()})
    chat.organism = organism
    chat.recall_fingerprint = {path.name: expected[path] for path in model_files}
    chat.awake_individual = dict(model_sha256=expected[MODEL / 'model.safetensors'], source_sha256=expected[SOURCE],
                                 memory_path=str(working_path), seed_sha256=expected[paths['banks']])
    return chat


def import_checkpoint(source, destination):
    if uses_database(source) or not uses_database(destination) or not source.is_file():
        raise ValueError('Import requires an existing .pt checkpoint and a new .sqlite3 destination')
    seed = TRANSFER_RUN / 'banks.pt'
    digest = read(TRANSFER_RUN / 'result.json')['artifact_sha256'][seed.name]
    if sha(seed) != digest:
        raise ValueError('Seed bank digest mismatch')
    banks, _, addresses, filled, pages, page_size, _ = load_working_banks(source, torch.load(seed, weights_only=True), digest)
    payload = bank_payload(banks, digest, addresses, filled, pages, page_size)
    with lock_working_checkpoint(destination):
        if destination.exists():
            raise FileExistsError(destination)
        revision, _, size = commit_banks(destination, payload, None, {})
    return dict(action='imported', revision=revision, subjects=len(addresses), tensor_bytes_written=size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entity', type=int)
    parser.add_argument('--version', type=int, choices=(0, 1), default=1)
    parser.add_argument('--back', type=int, default=0, help='Read this many SQLite corrections back for the selected field')
    parser.add_argument('--as-of', type=parse_as_of, help='Read the latest SQLite observation at a timestamp with timezone')
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument('--question')
    operation.add_argument('--remember', nargs=2, metavar=('FIELD', 'VALUE'))
    operation.add_argument('--observe', metavar='STATEMENT')
    operation.add_argument('--preview-observation', metavar='STATEMENT')
    operation.add_argument('--register', metavar='NAME')
    operation.add_argument('--import-checkpoint', type=Path, metavar='SOURCE_PT')
    operation.add_argument('--verify-replay', type=int, metavar='WRITES', help='Check a bounded suffix of SQLite memory writes')
    operation.add_argument('--repair-replay', type=int, metavar='WRITES', help='Repair verified discrepancies from SQLite snapshots')
    parser.add_argument('--expand', action='store_true', help='Allocate one memory page when registration reaches capacity')
    parser.add_argument('--memory-file', type=Path, default=WORKING_BANKS)
    parser.add_argument('--history', help='Preceding user turn')
    parser.add_argument('--awake-state', type=Path, help='Persist the experimental awake core during stdin chat')
    parser.add_argument('--sleep-replay', type=int, metavar='WRITES', help='Verify recent writes when entering sleep')
    parser.add_argument('--sleep-world', type=Path, help='Load a bounded P11 world model for synthetic sleep simulations')
    parser.add_argument('--sleep-ledger', type=Path, help='Save sleep replay and simulation evidence in a local event ledger')
    parser.add_argument('--awake-topic-recall', action='store_true', help='Use verified recent reads for bounded follow-ups')
    parser.add_argument('--awake-recall-intent-head', type=Path, help='Load the experimental repeat-intent classifier')
    parser.add_argument('--awake-subject-intent-head', type=Path, help='Load the experimental same-person classifier')
    parser.add_argument('--awake-subject-hops', type=int, choices=(1, 2, 3), default=1,
                        help='Maximum consecutive verified same-person reads')
    parser.add_argument('--self-name-state', type=Path, help='Load an experimental self-name matrix for chat')
    parser.add_argument('--ordinary-intent-head', type=Path, help='Load the experimental ordinary-request classifier')
    parser.add_argument('--migrate-backbone', action='store_true', help='Use fused SPORA blocks for the remaining Qwen2 layers')
    parser.add_argument('--inherited-router', action='store_true', help='Use the experimental previous-turn field classifier')
    parser.add_argument('--retrained-gate', action='store_true', help='Use the retrained inheritance decision')
    parser.add_argument('--earlier-readout', action='store_true', help='Use the experimental earlier memory handoff')
    parser.add_argument('--json', action='store_true', help='Include routing and read diagnostics')
    args = parser.parse_args()
    if args.expand and args.register is None:
        parser.error('--expand requires --register')
    replay_budget = args.repair_replay if args.repair_replay is not None else args.verify_replay
    if replay_budget is not None and (not 1 <= replay_budget <= 256 or not uses_database(args.memory_file)):
        parser.error('Replay requires a SQLite file and a budget from 1 to 256 writes')
    if args.repair_replay is not None and (args.version != 1 or args.back or args.as_of is not None):
        parser.error('--repair-replay requires the current memory version')
    if args.sleep_replay is not None and (args.awake_state is None or not uses_database(args.memory_file)
                                          or not 1 <= args.sleep_replay <= 256):
        parser.error('--sleep-replay requires --awake-state, a SQLite file and a budget from 1 to 256 writes')
    if args.sleep_world is not None and args.awake_state is None:
        parser.error('--sleep-world requires --awake-state')
    if args.sleep_ledger is not None and (args.sleep_replay is None and args.sleep_world is None):
        parser.error('--sleep-ledger needs a sleep replay or world model')
    observation = args.observe if args.observe is not None else args.preview_observation
    if args.awake_state is not None and (args.question is not None or observation is not None or args.remember
                                         or args.register is not None or args.import_checkpoint is not None
                                         or replay_budget is not None):
        parser.error('--awake-state is available during stdin chat')
    if args.awake_topic_recall and args.awake_state is None:
        parser.error('--awake-topic-recall requires --awake-state')
    if args.awake_recall_intent_head is not None and not args.awake_topic_recall:
        parser.error('--awake-recall-intent-head requires --awake-topic-recall')
    if args.awake_subject_intent_head is not None and not args.awake_topic_recall:
        parser.error('--awake-subject-intent-head requires --awake-topic-recall')
    if args.awake_subject_hops > 1 and args.awake_subject_intent_head is None:
        parser.error('--awake-subject-hops above 1 requires --awake-subject-intent-head')
    if not 0 <= args.back <= 2**63 - 1 or (args.back and (
            args.version != 1 or not uses_database(args.memory_file) or observation is not None
            or args.remember or args.register is not None or args.import_checkpoint is not None)):
        parser.error('--back requires a version-1 SQLite query')
    if args.as_of is not None and (args.back or args.version != 1 or not uses_database(args.memory_file)
                                   or observation is not None or args.remember or args.register is not None
                                   or args.import_checkpoint is not None):
        parser.error('--as-of requires a version-1 SQLite query without --back')
    if args.remember and args.entity is None:
        parser.error('--entity is required for field-based writes')
    if args.observe is not None and args.version != 1:
        parser.error('--observe updates version 1')
    if args.remember and (args.version != 1 or args.remember[0] not in FIELDS):
        parser.error('--remember updates version 1 and requires drink, work, or place')
    if args.retrained_gate and not args.inherited_router:
        parser.error('--retrained-gate requires --inherited-router')
    if args.import_checkpoint is not None:
        print(json.dumps(import_checkpoint(args.import_checkpoint.resolve(), args.memory_file.resolve())))
        return
    if observation is not None:
        addresses = ADDRESSES
        if args.memory_file.exists():
            seed = TRANSFER_RUN / 'banks.pt'
            digest = read(TRANSFER_RUN / 'result.json')['artifact_sha256'][seed.name]
            if sha(seed) != digest:
                raise ValueError('Seed bank digest mismatch')
            _, _, addresses, _, _, _, _ = load_working_banks(
                args.memory_file, torch.load(seed, weights_only=True), digest, lazy=uses_database(args.memory_file))
        candidate = parse_observation(observation, addresses)
        if candidate is None:
            print(json.dumps(dict(action='defer', reason='unsupported_or_ambiguous_statement')))
            return
        entity, field, value = candidate
        if args.entity is not None and args.entity != entity:
            print(json.dumps(dict(action='defer', reason='subject_mismatch')))
            return
        if args.preview_observation is not None:
            print(json.dumps(dict(action='preview', entity=entity, field=field, value=value), ensure_ascii=False))
            return
    chat = load_chat(args.memory_file, use_inherited_router=args.inherited_router, use_retrained_gate=args.retrained_gate)
    if args.migrate_backbone:
        migrated = chat.decoder.migrate_pretrained_blocks()
        print(f'Migrated {len(migrated)} pretrained blocks to fused SPORA projections.', file=sys.stderr, flush=True)
    if replay_budget is not None:
        result = chat.verify_recent_memory(replay_budget, repair=args.repair_replay is not None)
        print(json.dumps(result, ensure_ascii=False))
        return
    if args.register is not None:
        print(json.dumps(chat.register(args.register, expand=args.expand), ensure_ascii=False))
        return
    if args.observe is not None:
        if chat.addresses.get(next(name for name, key in addresses.items() if key == entity)) != entity:
            raise RevisionConflict('Subject registry changed during observation')
        result = chat.observe(observation, entity=args.entity)
        print(json.dumps(result, ensure_ascii=False))
        return
    if args.entity is not None and args.entity not in chat.addresses.values():
        parser.error('--entity must refer to a registered subject')
    if args.remember:
        print(json.dumps(chat.remember(args.entity, *args.remember), ensure_ascii=False))
        return
    timing_threshold = EARLY_TIMING_THRESHOLD if args.earlier_readout else THRESHOLD
    world_session = None
    world_digest = None
    ledger = None
    world_observation_event = None
    last_simulation = None
    simulation_event = None
    if args.sleep_world is not None:
        world_path = args.sleep_world.resolve()
        if world_path in (chat.working_path, args.awake_state.resolve()):
            parser.error('--sleep-world requires a separate model file')
        world_bytes = world_path.read_bytes()
        chat.organism.time_world, _ = TimeWorld.load(BytesIO(world_bytes))
        world_digest = sha256(world_bytes).hexdigest()
        world_session = OnlineSession(chat.organism.time_world, OnlineState())
    if args.ordinary_intent_head is not None:
        ordinary_path = args.ordinary_intent_head.resolve()
        if ordinary_path == chat.working_path or (args.awake_state is not None and ordinary_path == args.awake_state.resolve()):
            parser.error('--ordinary-intent-head needs a separate file from memory and awake state')
        width = chat.decoder.base.lm_head.weight.shape[1]
        chat.ordinary_intent = OrdinaryIntent.load(ordinary_path, width, chat.recall_fingerprint)
    if args.self_name_state is not None:
        name_path = args.self_name_state.resolve()
        if name_path == chat.working_path or (args.awake_state is not None and name_path == args.awake_state.resolve()):
            parser.error('--self-name-state needs a separate file from memory and awake state')
        width = chat.decoder.base.lm_head.weight.shape[1]
        chat.self_name = SelfNameIntent.load_name(name_path, width, chat.recall_fingerprint, chat.bank_specs['drink', 1])
        chat.organism.self_name = chat.self_name
        chat.awake_individual['self_name_sha256'] = sha(name_path)

    def record_sleep_event(kind, payload, source, event_id=None):
        if ledger is not None:
            revision = chat.working_sha
            ledger.append(kind, dict(payload, memory_revision=revision), event_id=event_id or uuid4().hex,
                          source=source, state_version=revision if isinstance(revision, int) else 0)

    def answer_lines(lines, topic_observer=None, consume_topic=None, bind_topic=None):
        nonlocal world_observation_event
        history = args.history
        ordinary_history = deque(maxlen=ORDINARY_TURNS)
        for line in lines:
            question = line.strip()
            if not question:
                continue
            command = question.split(maxsplit=1)
            if world_session is not None and command[0] == '/world-plan':
                try:
                    values = [float(value) for value in command[1].split()] if len(command) == 2 else []
                    suggestion = world_session.choose_imagined_plan(last_simulation, values)
                except ValueError:
                    suggestion = dict(kind='invalid_world_goal')
                if suggestion['kind'] == 'plan_suggestion':
                    suggestion['action'] = DEMO_WORLD_ACTIONS[suggestion['plan'] - 1]
                    record_sleep_event('world.plan_suggestion', dict(suggestion, simulation_event=simulation_event,
                                                                     world_sha256=world_digest), 'model_simulation')
                print(json.dumps(dict(world_plan=suggestion), ensure_ascii=False), flush=True)
                continue
            if world_session is not None and command[0] == '/world-check':
                try:
                    plan, step, *values = command[1].split() if len(command) == 2 else ()
                    comparison = world_session.compare_imagination(last_simulation, int(plan), int(step),
                                                                    [float(value) for value in values])
                except ValueError:
                    comparison = dict(kind='invalid_world_outcome')
                if comparison['kind'] == 'comparison':
                    record_sleep_event('world.outcome_check', dict(comparison, simulation_event=simulation_event,
                                                                  world_sha256=world_digest), 'synthetic_outcome_input')
                print(json.dumps(dict(world_check=comparison), ensure_ascii=False), flush=True)
                continue
            if world_session is not None and command[0] == '/world':
                if consume_topic is not None:
                    consume_topic()
                history = None
                ordinary_history.clear()
                try:
                    numbers = [float(token) for token in command[1].split()] if len(command) == 2 else []
                    world_session.observe_world(numbers[:3], numbers[3:5], numbers[5:], 'synthetic')
                except ValueError:
                    world = dict(action='defer', reason='expected_supported_world_observation')
                else:
                    world = dict(action='world_observed', source='synthetic', count=world_session.state.world_count)
                    if ledger is not None:
                        world_observation_event = uuid4().hex
                        record_sleep_event('world.observation', dict(values=numbers, count=world['count'], world_sha256=world_digest),
                                           'synthetic_input', world_observation_event)
                print(json.dumps(world, ensure_ascii=False), flush=True)
                continue
            if args.question is None and command[0] in ('/observe', '/记住'):
                if consume_topic is not None:
                    consume_topic()
                history = None
                ordinary_history.clear()
                try:
                    if args.version != 1 or args.back or args.as_of is not None:
                        result = dict(action='defer', reason='read_only_query_mode')
                    else:
                        result = chat.observe(command[1] if len(command) == 2 else '', entity=args.entity)
                except RevisionConflict:
                    result = dict(action='defer', reason='reload_required')
                text = {
                    'stored': '已更新这项记录。',
                    'already_stored': '这项记录已经存在。',
                    'defer': '未写入，请用已登记姓名说明一项明确的记录。',
                }[result['action']]
                if result.get('reason') == 'reload_required':
                    text = '记忆文件已被其他进程更新，请重启聊天后重试。'
                elif result.get('reason') == 'read_only_query_mode':
                    text = '当前是历史查询模式，请重新启动当前版本会话后写入。'
                elif result.get('reason') == 'matrix_verification_failed':
                    text = '写入候选未通过记忆校验，本次未保存。'
                print(json.dumps(result, ensure_ascii=False) if args.json else text, flush=True)
                continue
            result = chat.answer(question, args.entity, args.version, history, timing_threshold=timing_threshold,
                                 back=args.back, as_of=args.as_of, topic_observer=topic_observer,
                                 consume_topic=consume_topic, bind_topic=bind_topic, ordinary_history=ordinary_history)
            if result['action'] == 'ordinary':
                ordinary_history.append((question, result['response']))
            else:
                ordinary_history.clear()
            print(json.dumps(result, ensure_ascii=False) if args.json else result['response'], flush=True)
            history = None if args.awake_topic_recall else question

    if args.question is not None:
        answer_lines((args.question,))
    elif args.awake_state is not None:
        state_path = args.awake_state.resolve()
        trained = SOURCE.parents[1] / 'p24-awake' / 'core-seed7.pt'
        if state_path in (chat.working_path, trained.resolve()) or state_path.suffix.lower() != '.pt':
            parser.error('--awake-state needs a separate .pt path')
        if args.awake_recall_intent_head is not None:
            head_path = args.awake_recall_intent_head.resolve()
            if state_path == head_path:
                parser.error('--awake-state needs a path separate from the recall intent head')
            width = chat.decoder.base.lm_head.weight.shape[1]
            chat.recall_intent = RecallIntent.load(head_path, width, chat.recall_fingerprint)
        if args.awake_subject_intent_head is not None:
            head_path = args.awake_subject_intent_head.resolve()
            if state_path == head_path:
                parser.error('--awake-state needs a path separate from the subject intent head')
            width = chat.decoder.base.lm_head.weight.shape[1]
            chat.same_subject_intent = SameSubjectIntent.load(head_path, width, chat.recall_fingerprint)
            chat.subject_hops = args.awake_subject_hops

        def report(status):
            print(json.dumps(dict(awake=status), ensure_ascii=False), flush=True)

        def replay_on_sleep():
            nonlocal last_simulation, simulation_event
            result = {}
            if world_session is not None:
                result['imagination'] = world_session.imagine([[action] * 4 for action in DEMO_WORLD_ACTIONS], [.25] * 4)
                if result['imagination']['kind'] == 'simulation':
                    result['imagination']['actions'] = DEMO_WORLD_ACTIONS
                    last_simulation = result['imagination']
                    simulation_event = uuid4().hex
                    record_sleep_event('world.imagination', dict(last_simulation, evidence_event=world_observation_event,
                                                                  world_sha256=world_digest), 'model_simulation', simulation_event)
            if args.sleep_replay is not None:
                try:
                    result['replay'] = chat.verify_recent_memory(args.sleep_replay)
                except RevisionConflict:
                    result['replay'] = dict(action='defer', reason='reload_required')
                record_sleep_event('memory.replay', result['replay'], 'snapshot_verification')
            return result

        print('Awake controls: /status /relax /focus /sleep /wake /observe', file=sys.stderr, flush=True)
        with ExitStack() as resources:
            if args.sleep_ledger is not None:
                ledger_path = args.sleep_ledger.resolve()
                if ledger_path.suffix.lower() not in ('.sqlite', '.sqlite3') or ledger_path == chat.working_path:
                    parser.error('--sleep-ledger needs a separate SQLite file')
                resources.enter_context(lock_working_checkpoint(ledger_path))
                opener = EventLedger.open if ledger_path.exists() else EventLedger.create
                ledger = resources.enter_context(opener(ledger_path))
                ledger.verify()
                individual = dict(chat.awake_individual, awake_path=str(state_path))
                first = ledger.connection.execute('SELECT kind, payload FROM events ORDER BY seq LIMIT 1').fetchone()
                if first is None:
                    ledger.append('sleep.individual', individual, event_id=uuid4().hex)
                elif first[0] != 'sleep.individual' or json.loads(first[1]) != individual:
                    raise ValueError('Sleep ledger belongs to a different chat individual')
            lines, observe_topic, consume_topic, bind_topic = resources.enter_context(awake_chat_lines(
                chat.organism, sys.stdin, state_path, trained, chat.awake_individual, report,
                topic_labels=KEYS, clear_topics_on_sleep=args.awake_topic_recall,
                on_sleep=replay_on_sleep if args.sleep_replay is not None or world_session is not None else None,
            ))
            if args.awake_topic_recall:
                answer_lines(lines, observe_topic, consume_topic, bind_topic)
            else:
                answer_lines(lines, observe_topic)
    else:
        if sys.stdin.isatty():
            print('Enter one question per line. Ctrl+Z then Enter exits on Windows.', file=sys.stderr, flush=True)
        answer_lines(sys.stdin)


if __name__ == '__main__':
    main()
