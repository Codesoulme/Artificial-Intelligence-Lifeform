"""Local DeepSeek/SPORA route-and-answer entry point.

The router is experimental: it uses learned frozen-feature gates and refuses
ambiguous or unsupported requests. It never fabricates a memory address from
an external answer table.
"""
import argparse
import json
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
from scripts.deepseek_spora_read_intent_gate import OUTPUT as INTENT
from scripts.deepseek_spora_single_entity_gate import OUTPUT as ENTITY
from scripts.deepseek_spora_gate_fresh_intent_audit import KNOWN
from scripts.deepseek_named_binding_migrate import OUTPUT as NAMED
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_temporal_intent_router import TRAIN
from scripts.spora_event_scan_selector import select_event
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.current_memory_status import current_memory_status
from spora.growing import load
from spora.integrated_checkpoint import load_integrated
from spora.internal_memory import observe_sourced
from spora.memory_output import MemoryOutputAdapter

CHECKPOINT = ROOT/'runs/deepseek-spora-memory-output-integrated-v5-20261111/individual-with-output-adapter.pt'
TEMPORAL = ROOT/'runs/deepseek-spora-temporal-gate-train-20261111/temporal.pt'
READ_INTENT_RECOVERY_FLOOR = 2.0


class SporaChat:
    def __init__(self, base, tokenizer, decoder, organism, state, gates):
        self.base, self.tokenizer, self.decoder = base, tokenizer, decoder
        self.organism, self.state = organism, state
        self.router, self.entity_data, self.intent_data, self.temporal, self.temporal_data = gates
        self.bases = torch.stack([F.normalize(site.current_basis.detach().cpu().float(), dim=-1)
                                  for site in organism.memories])

    @torch.no_grad()
    def answer(self, question):
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
        selected, single = single_span_mask(
            locator(token_vec).squeeze(-1).sigmoid()[0], masks[0], self.router['locator_threshold'])
        intent_logit = float(intent(qvec).squeeze(-1)[0])
        entity_score = float(entity(qvec).squeeze(-1)[0].sigmoid())
        result = {'question': question, 'accepted': False, 'answer': None,
                  'temporal_intent': temporal_intent, 'intent_logit': intent_logit,
                  'entity_score': entity_score}
        intent_confident = intent_logit >= self.intent_data['threshold']
        intent_recovered = (intent_logit >= READ_INTENT_RECOVERY_FLOOR
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
        if current_memory_status(self.organism, self.state, key)['status'] != 'known':
            result['reason'] = 'unknown_memory'
            return result
        mode = 'current' if temporal_intent == 0 else 'past' if temporal_intent == 1 else 'later'
        selection = None if mode == 'current' else select_event(
            self.organism, self.state,
            key, 'effective_previous' if mode == 'past' else 'effective_latest',
            as_of=self.state.cursor-1)
        result['memory_event_selection'] = selection
        if selection is not None and selection['status'] != 'experimental_candidate':
            result['reason'] = 'memory_event_abstain'
            return result
        event_time = None if selection is None else selection['event_time']
        args = _memory_args(self.organism, self.state, {'key': key, 'intent': mode,
                                                        'event_time': 0 if event_time is None else event_time})
        device = next(self.organism.parameters()).device
        key_tensor = torch.tensor([key], dtype=torch.long, device=device)
        mode_tensor = torch.zeros_like(key_tensor) if mode == 'current' else torch.ones_like(key_tensor)
        time_tensor = torch.zeros_like(key_tensor) if event_time is None else torch.tensor([event_time], device=device)
        memory_logits = __import__('spora.growing', fromlist=['query']).query(
            self.organism, self.state, key_tensor, mode_tensor, time_tensor)
        prefix = closed_think_inputs(self.tokenizer, question, 'cuda')
        ids = self.decoder.generate(**prefix, **args, memory_output_logits=memory_logits,
                                    max_new_tokens=32, do_sample=False,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    eos_token_id=self.tokenizer.eos_token_id)
        result['accepted'], result['reason'] = True, 'routed'
        result['answer'] = self.tokenizer.decode(
            ids[0, prefix['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        result['key'] = key
        return result


def load_chat():
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA or not CHECKPOINT.exists():
        raise ValueError('Protected model or integrated checkpoint unavailable')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    decoder, organism, state = load_integrated(CHECKPOINT, base, MODEL, individual_device='cuda')
    router_report = json.loads((ROUTER/'result.json').read_text(encoding='utf-8'))
    entity_report = json.loads((ENTITY/'result.json').read_text(encoding='utf-8'))
    intent_report = json.loads((INTENT/'result.json').read_text(encoding='utf-8'))
    router = torch.load(ROUTER/'router.pt', map_location='cpu', weights_only=True)
    entity_data = torch.load(ENTITY/'gate.pt', map_location='cpu', weights_only=True)
    intent_data = torch.load(INTENT/'intent.pt', map_location='cpu', weights_only=True)
    for path, expected in ((ROUTER/'router.pt', router_report['router_sha256']),
                           (ENTITY/'gate.pt', entity_report['gate_sha256']),
                           (INTENT/'intent.pt', intent_report['intent_sha256'])):
        if sha(path) != expected:
            raise ValueError('Router artifact digest mismatch')
    temporal_report = json.loads((TEMPORAL.parent/'result.json').read_text(encoding='utf-8'))
    if sha(TEMPORAL) != temporal_report['temporal_sha256']:
        raise ValueError('Temporal artifact digest mismatch')
    temporal_data = torch.load(TEMPORAL, map_location='cpu', weights_only=True)
    temporal_width = temporal_data['state']['weight'].shape[1]
    temporal = nn.Linear(temporal_width, 4).eval()
    temporal.load_state_dict(temporal_data['state'], strict=True)
    state = fresh_controlled_state(organism) if state.cursor == 0 else state
    return SporaChat(base, tokenizer, decoder, organism, state,
                     (router, entity_data, intent_data, temporal, temporal_data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--question')
    args = parser.parse_args()
    chat = load_chat()
    if args.question:
        print(json.dumps(chat.answer(args.question), ensure_ascii=False))
        return
    for line in sys.stdin:
        question = line.strip()
        if question:
            print(json.dumps(chat.answer(question), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
