"""Local DeepSeek/SPORA route-and-answer entry point.

The router is experimental: it uses learned frozen-feature gates and refuses
ambiguous or unsupported requests. It never fabricates a memory address from
an external answer table.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_learned_name_span import single_span_mask
from scripts.deepseek_named_binding_probe import pooled, predict
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_composed_router_answer_eval import _memory_args
from scripts.deepseek_spora_memory_answer_distill import SOURCE_CHECKPOINT
from scripts.deepseek_spora_recalibrate_name_router import OUTPUT as ROUTER
from scripts.deepseek_spora_gate_fresh_intent_audit import KNOWN
from scripts.deepseek_named_binding_migrate import OUTPUT as NAMED
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_temporal_intent_router import TRAIN
from scripts.spora_event_scan_selector import select_event
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.current_memory_status import current_memory_status
from spora.growing import load, query
from spora.integrated_checkpoint import load_integrated, replacement_feature_variant_sha256
from spora.named_address import address_basis_sha256, feature_files_sha256
from spora.internal_memory import observe_sourced
from spora.observation_candidate import (ObservationCandidate, calibrated_write_guard,
                                          write_guard)
from spora.memory_output import MemoryOutputAdapter
from spora.structure_transfer import SPORALayerBlend, SPORALayerReplacement
from spora.working_checkpoint_lock import lock_working_checkpoint
from spora.durable import RevisionConflict
from spora.natural_write_heads import load_natural_write_heads
from spora.structured_text_memory import (StructuredTextMemoryCodec, StructuredTextMemoryState,
                                          StructuredTextWriteGate)
from spora.structured_text_output import (StructuredFactSuffixDecoder, StructuredMemoryTextDecoder,
                                          contains_fact_tokens)

CHECKPOINT = ROOT/'runs/deepseek-spora-memory-output-integrated-v5-20261111/individual-with-output-adapter.pt'
WORKING_CHECKPOINT = ROOT/'runs/spora-runtime-individual/individual.pt'
TEMPORAL = ROOT/'runs/deepseek-spora-temporal-gate-train-20261111/temporal.pt'
READ_INTENT_RECOVERY_FLOOR = 2.0
INTENT = ROOT/'runs/deepseek-spora-read-intent-gate-v7-20261112'
ENTITY = ROOT/'runs/deepseek-spora-single-entity-gate-v2-20261111'
MEMORY_DOMAIN_ANCHORS = ('饮品', '饮料', '喝', '饮用')
PAST_TIME_MARKERS = (
    '以前', '过去', '原来', '原先', '之前', '曾经', '历史', '先前',
    '早先', '较早', '最初', '早期', '改口以前', '变更前', '更正前',
    '修改前', '更改前', '变动前', '变更以前', '更正之前',
)
CURRENT_TIME_MARKERS = ('现在', '目前', '当下', '此刻', '眼下', '当前', '最新')
LATER_TIME_MARKERS = (
    '后来', '之后', '以后', '更新后', '变更以后', '后续', '修改完成后', '将来',
    '后来的', '更新完成后', '更正以后', '再往后', '再往后的',
)


def constrain_temporal_intent(question, predicted):
    """Require explicit temporal evidence before selecting a historical page."""
    if not isinstance(question, str) or predicted not in (0, 1, 2, 3):
        raise ValueError('Temporal constraint inputs are invalid')
    has_past = any(marker in question for marker in PAST_TIME_MARKERS)
    has_later = any(marker in question for marker in LATER_TIME_MARKERS)
    if has_past and has_later:
        return 3
    if has_past:
        return 1
    if has_later:
        return 2
    if any(marker in question for marker in CURRENT_TIME_MARKERS):
        return 0
    if predicted in (1, 2):
        return 0
    return predicted


def _supported_beverage_evidence(logits, value_count):
    """Only finite-symbol values with a defined beverage vocabulary can be used."""
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape != (1, value_count)
            or not bool(torch.isfinite(logits).all())):
        raise ValueError('Invalid native memory output evidence')
    return int(logits.argmax(-1)[0]) < len(WORDS)


def _prepare_memory_read(organism, state, key, mode, event_time):
    """Derive layer read arguments and output evidence from one selection.

    This couples the inputs, but does not prove that the neural reader and
    output evidence agree, or that the selected event is factually correct.
    """
    if mode not in ('current', 'past', 'later'):
        raise ValueError('Unsupported memory mode')
    if mode == 'current' and event_time is not None:
        raise ValueError('Current memory must not select a historical event')
    if mode != 'current' and (type(event_time) is not int or event_time < 0):
        raise ValueError('Historical memory requires an event time')
    event = 0 if event_time is None else event_time
    args = _memory_args(organism, state, {'key': key, 'intent': mode, 'event_time': event})
    device = next(organism.parameters()).device
    keys = torch.tensor([key], dtype=torch.long, device=device)
    modes = torch.zeros_like(keys) if mode == 'current' else torch.ones_like(keys)
    times = torch.full_like(keys, event)
    return args, query(organism, state, keys, modes, times)


class SporaChat:
    def __init__(self, base, tokenizer, decoder, organism, state, gates, working_sha256=None,
                 natural_write_heads=None):
        self.base, self.tokenizer, self.decoder = base, tokenizer, decoder
        self.organism, self.state = organism, state
        self._working_sha256 = working_sha256
        self.natural_write_heads = natural_write_heads
        self.router, self.entity_data, self.intent_data, self.temporal, self.temporal_data = gates
        # Research-only per-instance override; the default remains the module policy.
        self.read_intent_recovery_floor = float(READ_INTENT_RECOVERY_FLOOR)
        self.bases = torch.stack([F.normalize(site.current_basis.detach().cpu().float(), dim=-1)
                                  for site in organism.memories])

    def _replacements(self):
        layers = self.decoder.base.model.layers
        if any(isinstance(layer, SPORALayerBlend) for layer in layers):
            raise ValueError('Blended structural layers cannot be persisted')
        return {index: layer for index, layer in enumerate(layers)
                if isinstance(layer, SPORALayerReplacement)}

    def _structured_codec(self):
        structured = self.state.structured_text
        if structured is None:
            return StructuredTextMemoryCodec(
                self.base.model.embed_tokens.weight.detach().cpu(),
                memory_width=self.organism.config.memory_width,
                identity_slots=3, fact_slots=5,
                identity_slot_width=15, fact_slot_width=9,
                seed=29071, eos_token_id=self.tokenizer.eos_token_id)
        return StructuredTextMemoryCodec.from_config(
            self.base.model.embed_tokens.weight.detach().cpu(), structured.codec_config)

    @torch.no_grad()
    def observe_structured_text(self, key, identity_text, fact_text, *, query_text=None,
                                minimum_write_strength=0.20, replace_existing=False):
        """Persist one explicitly segmented arbitrary-text memory record.

        Natural-language routing is deliberately outside this primitive; the
        caller must supply a verified entity key and two text lanes.
        """
        if type(key) is not int or not 0 <= key < self.organism.config.key_count:
            raise ValueError('Structured text key is outside the individual address space')
        if not isinstance(identity_text, str) or not identity_text.strip():
            raise ValueError('Identity text must be nonempty')
        if not isinstance(fact_text, str) or not fact_text.strip():
            raise ValueError('Fact text must be nonempty')
        if query_text is not None and (not isinstance(query_text, str) or not query_text.strip()):
            raise ValueError('Query text must be nonempty when supplied')
        if type(replace_existing) is not bool:
            raise ValueError('replace_existing must be a boolean')
        from dataclasses import replace
        from spora.integrated_checkpoint import save_integrated
        codec = self._structured_codec()
        identity_ids = self.tokenizer.encode(identity_text.strip(), add_special_tokens=False)
        fact_ids = self.tokenizer.encode(fact_text.strip(), add_special_tokens=False)
        packet, metadata = codec.encode(identity_ids, fact_ids)
        structured = self.state.structured_text
        if structured is None:
            structured = StructuredTextMemoryState(codec.config_dict())
        elif structured.codec_config != codec.config_dict():
            raise ValueError('Structured text codec configuration changed')
        previous_record = structured.records.get(key)
        previous = None if previous_record is None else previous_record.packet
        if previous_record is not None:
            decoded = codec.decode(previous_record.packet, previous_record.metadata)
            if decoded['identity']['token_ids'] != identity_ids:
                return {'action': 'structured_text_deferred', 'key': key,
                        'reason': 'identity_mismatch'}
            if decoded['fact']['token_ids'] != fact_ids and not replace_existing:
                return {'action': 'structured_text_deferred', 'key': key,
                        'reason': 'replacement_requires_explicit_intent'}
        observation_ids = fact_ids
        query_ids = (None if query_text is None else
                     self.tokenizer.encode(query_text.strip(), add_special_tokens=False))
        gate = StructuredTextWriteGate(
            self.base.model.embed_tokens.weight.detach().cpu(), minimum_write_strength)
        decision = gate.evaluate(packet, previous, observation_ids, query_ids)
        if not decision['write']:
            return {'action': 'structured_text_deferred', 'key': key, 'gate': decision}
        candidate = replace(self.state, structured_text=structured.put(
            key, packet.detach().cpu(), metadata, self.organism.config.memory_width,
            self.organism.config.key_count))
        with lock_working_checkpoint(WORKING_CHECKPOINT):
            current_sha = sha(WORKING_CHECKPOINT) if WORKING_CHECKPOINT.exists() else None
            if current_sha != self._working_sha256:
                raise RevisionConflict('Working checkpoint changed; reload before writing')
            save_integrated(WORKING_CHECKPOINT, self.organism, candidate, self.decoder.reader,
                            MODEL, self.decoder.layer_indices, replacements=self._replacements(),
                            output_adapter=self.decoder.output_adapter)
            self.state = candidate
            self._working_sha256 = sha(WORKING_CHECKPOINT)
        return {'action': 'structured_text_written', 'key': key, 'gate': decision,
                'identity_length': metadata['identity_length'],
                'fact_length': metadata['fact_length'],
                'revision': candidate.structured_text.revision}

    @torch.no_grad()
    def read_structured_text(self, key):
        """Decode one persisted structured record without exposing its tensor."""
        if type(key) is not int or key < 0:
            raise ValueError('Structured text key must be a nonnegative integer')
        structured = self.state.structured_text
        if structured is None or key not in structured.records:
            return None
        codec = self._structured_codec()
        record = structured.records[key]
        decoded = codec.decode(record.packet, record.metadata)
        return {
            'identity_text': self.tokenizer.decode(decoded['identity']['token_ids'], skip_special_tokens=True),
            'fact_text': self.tokenizer.decode(decoded['fact']['token_ids'], skip_special_tokens=True),
            'identity_token_ids': decoded['identity']['token_ids'],
            'fact_token_ids': decoded['fact']['token_ids'],
            'metadata': dict(record.metadata),
            'revision': record.revision,
        }

    @torch.no_grad()
    def generate_structured_text(self, key, decoder_path, *, max_new_tokens=32):
        """Generate a fact sentence from a persisted packet via an opt-in head."""
        if type(key) is not int or key < 0:
            raise ValueError('Structured text key must be a nonnegative integer')
        structured = self.state.structured_text
        if structured is None or key not in structured.records:
            return {'accepted': False, 'reason': 'unknown_structured_memory'}
        device = next(self.base.parameters()).device
        record = structured.records[key]
        codec = self._structured_codec()
        payload = torch.load(decoder_path, map_location='cpu', weights_only=True)
        decoder_format = payload.get('format') if isinstance(payload, dict) else None
        if decoder_format == 'spora-structured-fact-suffix-decoder-v1':
            decoder = StructuredFactSuffixDecoder.from_checkpoint(
                self.base.model.embed_tokens.weight.detach(), decoder_path,
                expected_source_sha256=sha(CHECKPOINT), expected_base_sha256=WEIGHT_SHA,
                expected_codec=codec.config_dict()).to(device).eval()
            packet = record.packet.to(device=device, dtype=torch.float32).unsqueeze(0)
            suffix_ids = decoder.generate(packet, eos_id=self.tokenizer.eos_token_id,
                                          max_new_tokens=max_new_tokens)[0].tolist()
            decoded = codec.decode(record.packet, record.metadata)
            identity_ids = decoded['identity']['token_ids']
            fact_ids = decoded['fact']['token_ids']
            # A token-subsequence check can catch observed substitutions; it is
            # not proof that the generated relation or identity is correct.
            if not contains_fact_tokens(suffix_ids, fact_ids):
                return {'accepted': False, 'reason': 'generated_fact_mismatch',
                        'revision': record.revision}
            token_ids = identity_ids + suffix_ids
        else:
            decoder = StructuredMemoryTextDecoder.from_checkpoint(
                self.base.model.embed_tokens.weight.detach(), decoder_path,
                expected_source_sha256=sha(CHECKPOINT), expected_base_sha256=WEIGHT_SHA,
                expected_format='spora-structured-memory-text-decoder-v1').to(device).eval()
            packet = record.packet.to(device=device, dtype=torch.float32).unsqueeze(0)
            token_ids = decoder.generate(packet, eos_id=self.tokenizer.eos_token_id,
                                         max_new_tokens=max_new_tokens)[0].tolist()
        return {'accepted': True, 'reason': 'structured_memory_decoder',
                'answer': self.tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
                'token_ids': token_ids, 'revision': structured.records[key].revision}

    @torch.no_grad()
    def observe_sourced(self, key, value, source='direct', quality=1.0, valid_at=None):
        """Write one guarded fact and atomically publish the same individual."""
        from spora.integrated_checkpoint import save_integrated
        with lock_working_checkpoint(WORKING_CHECKPOINT):
            current_sha = sha(WORKING_CHECKPOINT) if WORKING_CHECKPOINT.exists() else None
            if current_sha != self._working_sha256:
                raise RevisionConflict('Working checkpoint changed; reload before writing')
            candidate, decision = observe_sourced(self.organism, self.state, key, value,
                                                  source, quality, valid_at)
            save_integrated(WORKING_CHECKPOINT, self.organism, candidate, self.decoder.reader,
                            MODEL, self.decoder.layer_indices, replacements=self._replacements(),
                            output_adapter=self.decoder.output_adapter)
            # Publish the committed state before hashing, so an I/O error cannot
            # leave this process behind a file that was already replaced.
            self.state = candidate
            self._working_sha256 = sha(WORKING_CHECKPOINT)
        return decision

    @torch.no_grad()
    def resolve_historical(self, key, event, *, value, valid_at, decision):
        """Persist an explicit adjudication via a private copy and revision lock.

        This experimental API does not infer truth or verify caller identity.
        Only already-deferred direct/reported events can be adjudicated.
        """
        from spora.historical_resolutions import HistoricalResolutions
        from spora.integrated_checkpoint import save_integrated
        from spora.state_codec import flatten, restore

        with lock_working_checkpoint(WORKING_CHECKPOINT):
            current_sha = sha(WORKING_CHECKPOINT) if WORKING_CHECKPOINT.exists() else None
            if current_sha != self._working_sha256:
                raise RevisionConflict('Working checkpoint changed; reload before resolving')
            schema, tensors = flatten(self.organism, self.state)
            # restore() may otherwise reuse same-device tensors for state pages.
            private = {name: tensor.detach().clone() for name, tensor in tensors.items()}
            device = next(self.organism.parameters()).device
            with torch.random.fork_rng(devices=[]):
                candidate, candidate_state = restore(schema, private, device)
            if not hasattr(candidate, 'historical_resolutions'):
                candidate.historical_resolutions = HistoricalResolutions(candidate.config.key_count).to(device)
            candidate.historical_resolutions.record(candidate, candidate_state, key, event,
                                                    value=value, valid_at=valid_at, decision=decision)
            save_integrated(WORKING_CHECKPOINT, candidate, candidate_state, self.decoder.reader,
                            MODEL, self.decoder.layer_indices, replacements=self._replacements(),
                            output_adapter=self.decoder.output_adapter)
            # Publish immediately after the atomic save, before hashing.
            self.organism, self.state = candidate, candidate_state
            self._working_sha256 = sha(WORKING_CHECKPOINT)
        return {'action': 'historical_resolution_recorded', 'key': key, 'event': event, 'decision': decision}

    @torch.no_grad()
    def observe_candidate(self, candidate, *, calibrated_gate=None, policy_id=None):
        """Persist only a fully resolved candidate; rejected candidates are side-effect free."""
        if not isinstance(candidate, ObservationCandidate):
            raise TypeError('An ObservationCandidate is required')
        if (calibrated_gate is None) != (policy_id is None):
            raise ValueError('Calibrated gate and policy id must be supplied together')
        if calibrated_gate is None:
            decision = write_guard(candidate, key_count=self.organism.config.key_count,
                                   value_count=self.organism.config.value_count,
                                   current_cursor=self.state.cursor)
        else:
            decision = calibrated_write_guard(
                candidate, calibrated_gate, key_count=self.organism.config.key_count,
                value_count=self.organism.config.value_count, current_cursor=self.state.cursor,
                policy_id=policy_id)
        # This runtime's output adapter has exactly the beverage vocabulary in
        # WORDS. Never silently coerce a parsed surface value to another symbol.
        if candidate.value_index is not None and (
                candidate.value_index >= len(WORDS)
                or candidate.value_text != WORDS[candidate.value_index]):
            decision['reasons'].append('finite_value_text_mismatch')
            decision['write'] = False
        if not decision['write']:
            return {'action': 'defer', 'reason': 'candidate_guard', 'reasons': decision['reasons']}
        return self.observe_sourced(candidate.subject_key, candidate.value_index,
                                    candidate.source, candidate.quality, candidate.valid_at)

    @torch.no_grad()
    def observe_text(self, text, *, policy_id='spora-polarity-calibration-20261111-v1',
                     source='direct', quality=1.0):
        """Parse and transactionally observe one Chinese statement.

        This is opt-in because the parser and heads are still research
        artifacts. It shares the normal candidate guard and persistence path.
        """
        if self.natural_write_heads is None:
            raise ValueError('Natural write heads are not enabled for this runtime')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('A nonempty observation text is required')
        from dataclasses import replace
        from scripts.deepseek_spora_natural_observation_transaction_pilot import (
            build_components, parse_candidate)
        parsed = parse_candidate(self, text, build_components(self))
        candidate = replace(parsed.pop('candidate'), source=source, quality=quality)
        gate = parsed.pop('gate')
        gate['policy_id'] = policy_id
        result = self.observe_candidate(candidate, calibrated_gate=gate, policy_id=policy_id)
        return {**parsed, 'action': result.get('action', 'written') if isinstance(result, dict) else 'written',
                'result': result}

    @torch.no_grad()
    def answer(self, question, *, identity_guard=None):
        if not isinstance(question, str) or not question.strip():
            raise ValueError('A nonempty Chinese question is required')
        features, masks = extract(self.base, self.tokenizer, [question], batch_size=1)
        qvec = pool(features, masks).float()
        token_vec = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],))
                               for i in range(2)], dim=-1)
        locator = nn.Linear(token_vec.shape[-1], 1).eval()
        locator.load_state_dict(self.router['locator_state'], strict=True)
        entity = nn.Linear(qvec.shape[-1], 1).eval()
        entity.load_state_dict(self.entity_data['gate_state'], strict=True)
        intent = nn.Linear(qvec.shape[-1], 1).eval()
        intent.load_state_dict(self.intent_data['state'], strict=True)
        time_logits = self.temporal(qvec)
        probabilities = time_logits.softmax(-1)[0]
        winner = int(probabilities.argmax())
        temporal_intent = (winner if winner != 3 and
                           float(probabilities[:3].max()) >= self.temporal_data['threshold'] else 3)
        temporal_intent = constrain_temporal_intent(question, temporal_intent)
        selected, single = single_span_mask(
            locator(token_vec).squeeze(-1).sigmoid()[0], masks[0], self.router['locator_threshold'])
        intent_logit = float(intent(qvec).squeeze(-1)[0])
        entity_score = float(entity(qvec).squeeze(-1)[0].sigmoid())
        result = {'question': question, 'accepted': False, 'answer': None,
                  'temporal_intent': temporal_intent, 'intent_logit': intent_logit,
                  'entity_score': entity_score}
        if not any(anchor in question for anchor in MEMORY_DOMAIN_ANCHORS):
            result['reason'] = 'unsupported_memory_predicate'
            return result
        intent_confident = intent_logit >= self.intent_data['threshold']
        intent_recovered = (intent_logit >= getattr(self, 'read_intent_recovery_floor', READ_INTENT_RECOVERY_FLOOR)
                            and entity_score >= self.entity_data['gate_threshold'] and single)
        if not intent_confident and not intent_recovered:
            result['reason'] = 'not_confident_read_intent'
            result['intent_recovered'] = False
            return result
        result['intent_recovered'] = not intent_confident
        if entity_score < self.entity_data['gate_threshold']:
            result['reason'] = 'not_single_entity'
            return result
        if temporal_intent == 3:
            result['reason'] = 'ambiguous_temporal'
            return result
        if not single:
            result['reason'] = 'no_single_name_span'
            return result
        v = pooled(features, selected.unsqueeze(0))
        indices, similarities = predict(v, self.router['projection'], self.bases)
        result['indices'] = indices[0].tolist()
        result['similarities'] = similarities[0].tolist()
        if not bool(indices[0, 0] == indices[0, 1]
                    and (similarities[0] >= self.entity_data['address_threshold']).all()):
            result['reason'] = 'name_binding_rejected'
            return result
        key = int(indices[0, 0])
        # Experimental opt-in only: reject before inspecting memory state or
        # generating. The caller must supply trustworthy enrollment and span
        # verification; the default route has no such guard.
        if identity_guard is not None and identity_guard.authorize(question, selected, key) is not True:
            result['reason'] = 'identity_guard_rejected'
            return result
        organism, state = self.organism, self.state
        if current_memory_status(organism, state, key)['status'] != 'known':
            result['reason'] = 'unknown_memory'
            return result
        mode = 'current' if temporal_intent == 0 else 'past' if temporal_intent == 1 else 'later'
        selection = None if mode == 'current' else select_event(
            organism, state,
            key, 'effective_previous' if mode == 'past' else 'effective_latest',
            as_of=state.cursor-1)
        result['memory_event_selection'] = selection
        if selection is not None and selection['status'] != 'experimental_candidate':
            result['reason'] = 'memory_event_abstain'
            return result
        event_time = None if selection is None else selection['event_time']
        args, memory_logits = _prepare_memory_read(organism, state, key, mode, event_time)
        if not _supported_beverage_evidence(memory_logits, organism.config.value_count):
            result['reason'] = 'unsupported_memory_value'
            return result
        prefix = closed_think_inputs(self.tokenizer, question, 'cuda')
        ids = self.decoder.generate(**prefix, **args, memory_output_logits=memory_logits,
                                    max_new_tokens=32, do_sample=False,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    eos_token_id=self.tokenizer.eos_token_id)
        tail = ids[0, prefix['input_ids'].shape[1]:]
        result['accepted'], result['reason'], result['output_retry'] = True, 'routed', False
        result['answer'] = self.tokenizer.decode(tail, skip_special_tokens=True).strip()
        result['key'] = key
        return result


def load_chat(*, base_loader=None, natural_write_heads_path=None):
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA or not CHECKPOINT.exists():
        raise ValueError('Protected model or integrated checkpoint unavailable')
    checkpoint = WORKING_CHECKPOINT if WORKING_CHECKPOINT.exists() else CHECKPOINT
    working_sha = sha(checkpoint) if checkpoint == WORKING_CHECKPOINT else None
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    if base_loader is None:
        base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                    torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    else:
        # Research seam: the caller owns in-memory base-weight provenance.
        base = base_loader().eval()
    base.requires_grad_(False)
    decoder, organism, state = load_integrated(checkpoint, base, MODEL, individual_device='cuda')
    router_report = json.loads((ROUTER/'result.json').read_text(encoding='utf-8'))
    entity_report = json.loads((ENTITY/'result.json').read_text(encoding='utf-8'))
    intent_report = json.loads((INTENT/'result.json').read_text(encoding='utf-8'))
    for path, expected in ((ROUTER/'router.pt', router_report['router_sha256']),
                           (ENTITY/'gate.pt', entity_report['gate_sha256']),
                           (INTENT/'intent.pt', intent_report['intent_sha256'])):
        if sha(path) != expected:
            raise ValueError('Router artifact digest mismatch')
    router = torch.load(ROUTER/'router.pt', map_location='cpu', weights_only=True)
    entity_data = torch.load(ENTITY/'gate.pt', map_location='cpu', weights_only=True)
    intent_data = torch.load(INTENT/'intent.pt', map_location='cpu', weights_only=True)
    temporal_report = json.loads((TEMPORAL.parent/'result.json').read_text(encoding='utf-8'))
    if sha(TEMPORAL) != temporal_report['temporal_sha256']:
        raise ValueError('Temporal artifact digest mismatch')
    temporal_data = torch.load(TEMPORAL, map_location='cpu', weights_only=True)
    replacements = {i: layer for i, layer in enumerate(base.model.layers)
                    if isinstance(layer, SPORALayerReplacement)}
    variant = replacement_feature_variant_sha256(MODEL, replacements)
    router_sha, entity_sha = sha(ROUTER/'router.pt'), sha(ENTITY/'gate.pt')
    provenance_errors = []
    if router.get('format') != 'spora-recalibrated-name-router-v1': provenance_errors.append('router_format')
    if router.get('base_sha256') != WEIGHT_SHA: provenance_errors.append('router_base')
    if router.get('feature_config_sha256') != feature_files_sha256(MODEL): provenance_errors.append('router_config')
    if router.get('feature_variant_sha256') != variant: provenance_errors.append('router_variant')
    if router.get('basis_sha256') != address_basis_sha256(organism): provenance_errors.append('router_basis')
    if router.get('feature_layers') != [6, 18]: provenance_errors.append('router_layers')
    if entity_data.get('format') != 'spora-single-entity-gate-v1': provenance_errors.append('entity_format')
    if entity_data.get('router_sha256') != router_sha: provenance_errors.append('entity_router')
    if entity_data.get('feature_variant_sha256') != variant: provenance_errors.append('entity_variant')
    if intent_data.get('format') != 'spora-finite-read-intent-v1': provenance_errors.append('intent_format')
    if intent_data.get('base_sha256') != WEIGHT_SHA: provenance_errors.append('intent_base')
    if intent_data.get('router_sha256') != router_sha: provenance_errors.append('intent_router')
    if intent_data.get('entity_gate_sha256') != entity_sha: provenance_errors.append('intent_entity')
    if intent_data.get('feature_variant_sha256') != variant: provenance_errors.append('intent_variant')
    if temporal_data.get('format') != 'spora-bounded-temporal-gate-v1': provenance_errors.append('temporal_format')
    if temporal_data.get('base_sha256') != WEIGHT_SHA: provenance_errors.append('temporal_base')
    if temporal_data.get('feature_variant_sha256') != variant: provenance_errors.append('temporal_variant')
    if temporal_data.get('classes') != ['current', 'past', 'later', 'abstain']: provenance_errors.append('temporal_classes')
    if provenance_errors:
        raise ValueError('Runtime router and individual provenance mismatch: ' + ','.join(provenance_errors))
    temporal_width = temporal_data['state']['weight'].shape[1]
    temporal = nn.Linear(temporal_width, 4).eval()
    temporal.load_state_dict(temporal_data['state'], strict=True)
    if ((checkpoint == WORKING_CHECKPOINT and sha(WORKING_CHECKPOINT) != working_sha)
            or (checkpoint == CHECKPOINT and WORKING_CHECKPOINT.exists())):
        raise RevisionConflict('Working checkpoint changed during load; retry')
    natural_write_heads = None
    if natural_write_heads_path is not None:
        natural_write_heads = load_natural_write_heads(
            natural_write_heads_path, base, MODEL, CHECKPOINT,
            device=torch.device('cpu'))
    return SporaChat(base, tokenizer, decoder, organism, state,
                     (router, entity_data, intent_data, temporal, temporal_data),
                     working_sha256=working_sha, natural_write_heads=natural_write_heads)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--question')
    parser.add_argument('--observe', nargs=2, type=int, metavar=('KEY', 'VALUE'))
    parser.add_argument('--observe-text', help='Opt-in natural-language observation; requires --natural-write-heads')
    parser.add_argument('--natural-write-heads', type=Path,
                        help='Research-only natural write head bundle')
    parser.add_argument('--source', default='direct', choices=('direct', 'report', 'inference', 'simulation'))
    parser.add_argument('--quality', type=float, default=1.0)
    parser.add_argument('--read-intent-floor', type=float, default=None,
                        help='Research-only per-instance read recovery floor override')
    args = parser.parse_args()
    if args.observe_text and args.natural_write_heads is None:
        parser.error('--observe-text requires --natural-write-heads')
    chat = load_chat(natural_write_heads_path=args.natural_write_heads)
    if args.read_intent_floor is not None:
        if not math.isfinite(args.read_intent_floor):
            parser.error('--read-intent-floor must be finite')
        chat.read_intent_recovery_floor = args.read_intent_floor
    if args.observe is not None:
        key, value = args.observe
        print(json.dumps(chat.observe_sourced(key, value, args.source, args.quality), ensure_ascii=False))
        return
    if args.observe_text:
        print(json.dumps(chat.observe_text(args.observe_text, source=args.source,
                                           quality=args.quality), ensure_ascii=False))
        return
    if args.question:
        print(json.dumps(chat.answer(args.question), ensure_ascii=False))
        return
    for line in sys.stdin:
        question = line.strip()
        if question:
            print(json.dumps(chat.answer(question), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
