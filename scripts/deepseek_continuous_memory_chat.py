"""Interactive chat and explicit writes for the continuous-memory research checkpoint."""

import argparse
import json
import sys
from pathlib import Path

from deepseek_aligned_memory_calibration import load_adapter
from deepseek_aligned_memory_dialogue import EARLY_TIMING_THRESHOLD, OUT as ADAPTER_RUN, RANK, THRESHOLD, generate_with_memory
from deepseek_memory_attribute_router import CLARIFY, MISSING
from deepseek_memory_attribute_span import Route, route_action
from deepseek_memory_cosine_readout import cosine_bias, projected_codebook
from deepseek_memory_dialogue_context import OUT as DIALOGUE_RUN, features_for, route
from deepseek_memory_field_decoding import FIELDS, OUT as FIELD_RUN
from deepseek_memory_field_transfer import OUT as TRANSFER_RUN
from deepseek_memory_inheritance_gate import OUT as INHERITANCE_GATE_RUN
from deepseek_memory_inherited_turn import OUT as INHERITED_RUN
from deepseek_memory_turn_priority import OUT as PRIORITY_RUN, gate_probabilities
from deepseek_output_aligned_memory import MODEL, SEED, SLOTS, SOURCE, encode, load_integrated, read, read_bank, sha, torch, write_bank
from spora.durable import RevisionConflict
from spora.language_training import atomic_save
from spora.research_continuous_observation import ADDRESSES, can_register, inherited_query_subject, parse_observation, query_subject
from spora.working_checkpoint_lock import lock_working_checkpoint

ENTITIES = tuple(ADDRESSES.values())
ASK_SUBJECT = '请明确你想查询谁的记录。'
WORKING_BANKS = TRANSFER_RUN / 'working-banks.pt'
LEGACY_WORKING_FORMAT = 'spora-continuous-chat-banks-v1'
WORKING_FORMAT = 'spora-continuous-chat-banks-v2'
SEED_FILLED = frozenset((entity, field, version) for entity in ENTITIES for field in FIELDS for version in (0, 1))


class InvalidMemoryValue(ValueError):
    """The submitted value cannot fit the continuous-memory slots."""


def load_working_banks(path, seed_banks, seed_sha):
    if not path.exists():
        return seed_banks, None, dict(ADDRESSES), SEED_FILLED
    digest = sha(path)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if sha(path) != digest:
        raise RevisionConflict('Continuous memory changed during load; retry')
    if payload.get('format') not in (LEGACY_WORKING_FORMAT, WORKING_FORMAT) or payload.get('seed_sha256') != seed_sha:
        raise ValueError('Working banks have incompatible provenance')
    banks = payload['banks']
    if set(banks) != set(seed_banks):
        raise ValueError('Working banks have incompatible fields or versions')
    for key, original in seed_banks.items():
        bank = banks[key]
        if (bank.shape != original.shape or bank.dtype != original.dtype or not bool(torch.isfinite(bank).all())
                or (key[1] == 0 and not torch.equal(bank, original))):
            raise ValueError('Invalid working bank')
    addresses, filled = dict(ADDRESSES), SEED_FILLED
    if payload['format'] == WORKING_FORMAT:
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
    return banks, digest, addresses, filled


class ContinuousMemoryChat:
    def __init__(self, decoder, tokenizer, site, banks, styles, timing, columns, contextual, direct, inheritance, setup,
                 *, working_path, working_sha, seed_sha, addresses, filled, inherited_router=None, gate_cutoff=None):
        self.decoder, self.tokenizer, self.site, self.banks = decoder, tokenizer, site, banks
        self.styles, self.timing, self.columns = styles, timing, columns
        self.contextual, self.direct, self.inheritance = contextual, direct, inheritance
        self.inherited_router = inherited_router
        self.gate_cutoff = setup['gate_cutoff'] if gate_cutoff is None else gate_cutoff
        self.cutoff = setup['cutoffs']['priority']
        self.working_path, self.working_sha, self.seed_sha = working_path, working_sha, seed_sha
        if max(addresses.values()) >= site.current_basis.shape[0]:
            raise ValueError('Registered subject is outside the address basis')
        self.addresses, self.filled = addresses, filled
        weight = decoder.base.lm_head.weight
        basis = torch.linalg.qr(torch.randn(weight.shape[1], RANK, generator=torch.Generator().manual_seed(SEED))).Q
        self.codes, self.scale = projected_codebook(weight, basis.to(weight.device))

    def _read_tokens(self, bank, entity):
        address = self.site.current_basis[entity:entity + 1]
        slots = read_bank(self.site, bank, address).to(self.codes.device)[:, :RANK]
        scores = torch.nn.functional.normalize(slots, dim=-1) @ self.codes.T
        return scores.argmax(dim=-1).masked_fill(slots.norm(dim=-1) < .2, -1).tolist()

    def _check_revision(self):
        current = sha(self.working_path) if self.working_path.exists() else None
        if current != self.working_sha:
            raise RevisionConflict('Continuous memory changed; reload before writing')

    def _save(self, banks, addresses, filled):
        atomic_save(dict(format=WORKING_FORMAT, seed_sha256=self.seed_sha, banks=banks,
                         addresses=addresses, filled=sorted(filled)), self.working_path)
        self.working_sha = sha(self.working_path)
        self.banks, self.addresses, self.filled = banks, addresses, filled

    def register(self, name):
        name = name.strip()
        with lock_working_checkpoint(self.working_path):
            self._check_revision()
            if name in self.addresses:
                return dict(action='already_registered', entity=self.addresses[name], name=name)
            if not can_register(name, self.addresses):
                return dict(action='defer', reason='invalid_or_overlapping_name')
            used = set(self.addresses.values())
            entity = next((key for key in range(self.site.current_basis.shape[0]) if key not in used), None)
            if entity is None:
                return dict(action='defer', reason='no_free_address')
            self._save(self.banks, {**self.addresses, name: entity}, self.filled)
        return dict(action='registered', entity=entity, name=name)

    @torch.inference_mode()
    def remember(self, entity, field, value):
        if entity not in self.addresses.values() or field not in FIELDS:
            raise ValueError('Unknown experimental address or field')
        tokens = encode(self.tokenizer, value.strip())
        if not 1 <= len(tokens) < SLOTS or set(tokens) & set(self.tokenizer.all_special_ids):
            raise InvalidMemoryValue(f'Value must have 1 to {SLOTS - 1} non-special tokens')
        tokens.append(self.tokenizer.eos_token_id)
        bank = self.banks[field, 1]
        record = entity, field, 1
        expected = tokens + [-1] * (SLOTS - len(tokens))
        with lock_working_checkpoint(self.working_path):
            self._check_revision()
            if record in self.filled and self._read_tokens(bank, entity) == expected:
                return dict(action='already_stored', entity=entity, field=field)
            target = torch.zeros_like(bank[:, :, 0])
            target[:len(tokens), :RANK] = self.codes[tokens].cpu()
            candidate = write_bank(self.site, bank, self.site.current_basis[entity:entity + 1], target)
            if self._read_tokens(candidate, entity) != expected:
                raise ValueError('Candidate matrix did not reproduce the written value')
            for other, other_field, version in self.filled:
                if other != entity and other_field == field and version == 1:
                    if self._read_tokens(candidate, other) != self._read_tokens(bank, other):
                        raise ValueError('Candidate matrix changed another occupied address')
            updated = dict(self.banks)
            updated[field, 1] = candidate
            self._save(updated, self.addresses, self.filled | {record})
        return dict(action='stored', entity=entity, field=field)

    @torch.inference_mode()
    def answer(self, question, entity=None, version=1, history=None, *, timing_threshold=THRESHOLD):
        if (entity is not None and entity not in self.addresses.values()) or version not in (0, 1):
            raise ValueError('Unknown experimental address or snapshot')
        row = dict(key='input', question=question, history=history, target_turn=0)
        features = features_for(self.decoder, self.tokenizer, row, False)
        probability, = gate_probabilities(self.inheritance, [row], {'input': features})
        inherits = bool(history) and probability >= self.gate_cutoff
        subject_reason = 'specified_subject'
        if entity is None:
            entity, subject_reason = query_subject(question, self.addresses)
            if entity is not None and history:
                previous, _ = query_subject(history, self.addresses)
                if previous != entity:
                    inherits = False
            elif subject_reason == 'no_subject' and inherits:
                entity = inherited_query_subject(question, history, self.addresses)
                if entity is not None:
                    subject_reason = 'inherited_subject'
        router = self.contextual if inherits else self.direct
        if inherits:
            features = features_for(self.decoder, self.tokenizer, row, True)
        route_features = features
        if inherits and self.inherited_router is not None:
            router = self.inherited_router
            route_features = features_for(self.decoder, self.tokenizer, dict(question=history), False)
        predicted, = route(router, [row], {'input': route_features})
        selected = predicted['selected']
        action = route_action(predicted, self.cutoff)
        start = None
        reason = action
        if entity is None and (action in ('read', 'missing') or subject_reason != 'no_subject'):
            action, reason, response = 'clarify', subject_reason, ASK_SUBJECT
        elif action == 'read' and (entity, selected, version) not in self.filled:
            action, reason, response = 'missing', 'unwritten_field', MISSING
        elif action == 'read':
            address = self.site.current_basis[entity:entity + 1]
            bias = cosine_bias(self.site, self.banks[selected, version], address, self.codes, self.scale)
            if bias is None:
                action, reason, response = 'missing', 'empty_matrix', MISSING
            else:
                tokens, start = generate_with_memory(self.decoder, self.tokenizer, features[3], self.styles[selected],
                                                      self.columns, self.timing, 2 * bias,
                                                      timing_threshold=timing_threshold)
                if start is None:
                    action, reason, response = 'clarify', 'injection_not_triggered', CLARIFY
                else:
                    response = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        elif action == 'ordinary':
            ids = torch.tensor([features[3]], device=self.codes.device)
            tokens = self.decoder.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                           max_new_tokens=64, pad_token_id=self.tokenizer.pad_token_id,
                                           eos_token_id=self.tokenizer.eos_token_id)
            response = self.tokenizer.decode(tokens[0, ids.shape[1]:], skip_special_tokens=True).strip()
        else:
            response = MISSING if action == 'missing' else CLARIFY
        return dict(response=response, action=action, reason=reason, field=selected, confidence=predicted['confidence'],
                    inherits=inherits, memory_start=start, entity=entity, version=version, subject_reason=subject_reason)


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
    banks, working_sha, addresses, filled = load_working_banks(working_path, seed_banks, expected[paths['banks']])
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
              dtype=torch.float16, attn_implementation='sdpa', device_map={'': 'cuda'}, low_cpu_mem_usage=True).eval()
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
    return ContinuousMemoryChat(decoder, tokenizer, organism.memories[0], banks, styles, timing, columns,
                                contextual, direct, inheritance, read(paths['routes']), working_path=working_path,
                                working_sha=working_sha, seed_sha=expected[paths['banks']], inherited_router=previous,
                                gate_cutoff=None if gate_result is None else gate_result['cutoffs']['tail'],
                                addresses=addresses, filled=filled)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entity', type=int)
    parser.add_argument('--version', type=int, choices=(0, 1), default=1)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument('--question')
    operation.add_argument('--remember', nargs=2, metavar=('FIELD', 'VALUE'))
    operation.add_argument('--observe', metavar='STATEMENT')
    operation.add_argument('--preview-observation', metavar='STATEMENT')
    operation.add_argument('--register', metavar='NAME')
    parser.add_argument('--memory-file', type=Path, default=WORKING_BANKS)
    parser.add_argument('--history', help='Preceding user turn')
    parser.add_argument('--inherited-router', action='store_true', help='Use the experimental previous-turn field classifier')
    parser.add_argument('--retrained-gate', action='store_true', help='Use the retrained inheritance decision')
    parser.add_argument('--earlier-readout', action='store_true', help='Use the experimental earlier memory handoff')
    parser.add_argument('--json', action='store_true', help='Include routing and read diagnostics')
    args = parser.parse_args()
    observation = args.observe if args.observe is not None else args.preview_observation
    if args.remember and args.entity is None:
        parser.error('--entity is required for field-based writes')
    if args.observe is not None and args.version != 1:
        parser.error('--observe updates version 1')
    if args.remember and (args.version != 1 or args.remember[0] not in FIELDS):
        parser.error('--remember updates version 1 and requires drink, work, or place')
    if args.retrained_gate and not args.inherited_router:
        parser.error('--retrained-gate requires --inherited-router')
    if observation is not None:
        addresses = ADDRESSES
        if args.memory_file.exists():
            seed = TRANSFER_RUN / 'banks.pt'
            digest = read(TRANSFER_RUN / 'result.json')['artifact_sha256'][seed.name]
            if sha(seed) != digest:
                raise ValueError('Seed bank digest mismatch')
            _, _, addresses, _ = load_working_banks(args.memory_file, torch.load(seed, weights_only=True), digest)
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
    if args.register is not None:
        print(json.dumps(chat.register(args.register), ensure_ascii=False))
        return
    if args.observe is not None:
        if chat.addresses.get(next(name for name, key in addresses.items() if key == entity)) != entity:
            raise RevisionConflict('Subject registry changed during observation')
        try:
            result = chat.remember(entity, field, value)
        except InvalidMemoryValue:
            result = dict(action='defer', reason='value_does_not_fit')
        print(json.dumps(result, ensure_ascii=False))
        return
    if args.entity is not None and args.entity not in chat.addresses.values():
        parser.error('--entity must refer to a registered subject')
    if args.remember:
        print(json.dumps(chat.remember(args.entity, *args.remember), ensure_ascii=False))
        return
    history = args.history
    timing_threshold = EARLY_TIMING_THRESHOLD if args.earlier_readout else THRESHOLD
    lines = sys.stdin if args.question is None else (args.question,)
    if args.question is None and sys.stdin.isatty():
        print('Enter one question per line. Ctrl+Z then Enter exits on Windows.', file=sys.stderr, flush=True)
    for line in lines:
        question = line.strip()
        if not question:
            continue
        result = chat.answer(question, args.entity, args.version, history, timing_threshold=timing_threshold)
        print(json.dumps(result, ensure_ascii=False) if args.json else result['response'], flush=True)
        history = question


if __name__ == '__main__':
    main()
