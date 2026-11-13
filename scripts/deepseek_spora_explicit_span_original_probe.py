"""Compare neural routing with explicitly selected names on original layers."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch import nn
from torch.nn import functional as F
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_learned_name_span import single_span_mask
from scripts.deepseek_named_binding_probe import KEYS, NAMES, pooled, predict
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_original_layer_joint_runtime_audit import checked, spans_agree
from scripts.deepseek_spora_original_layer_name_router import OUT as ROUTER
from spora.explicit_span_address import explicit_span_address
from spora.identity_fingerprints import IdentityFingerprints
from spora.identity_read_guard import IdentityReadGuard
from spora.named_address import address_basis_sha256, feature_files_sha256

ROOT_RUN = ROOT / 'runs'
GATES = ROOT_RUN / 'deepseek-spora-original-layer-request-gates-v2-20261114'
MULTI = ROOT_RUN / 'deepseek-spora-original-multi-span-pilot-20261114'
CAL = ROOT_RUN / 'deepseek-spora-original-layer-joint-calibration-20261114.json'
OUTPUT = ROOT_RUN / 'deepseek-spora-explicit-span-original-20261114.json'
RECORDS = tuple(zip(KEYS, NAMES, (1, 2, 3, 4, 5, 6, 7, 0))) + ((7, '小林舟', 2), (8, '阿明晨', 3))
CASES = (
    ('known', '你记得小林喜欢喝哪种饮品吗？', '小林', 6),
    ('known', '从记录看，阿宁常喝什么饮料？', '阿宁', 22),
    ('known', '你记得小雨喜欢喝哪种饮品吗？', '小雨', 35),
    ('known', '从记录看，阿哲常喝什么饮料？', '阿哲', 51),
    ('known', '你记得小梅喜欢喝哪种饮品吗？', '小梅', 16),
    ('known', '从记录看，阿远常喝什么饮料？', '阿远', 27),
    ('long', '你记得小林舟喜欢喝哪种饮品吗？', '小林舟', 7),
    ('long', '你记得阿明晨喜欢喝哪种饮品吗？', '阿明晨', 8),
    ('wrong_selection', '你记得小林舟喜欢喝哪种饮品吗？', '小林', 7),
    ('missing', '你记得小林喜欢喝哪种饮品吗？', None, 6),
    ('unknown', '你记得小林华喜欢喝哪种饮品吗？', '小林华', None),
)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    ckpt_sha = p.sha(p.runtime.CHECKPOINT)
    router, _, router_path, router_sha = checked(ROUTER, 'router.pt', 'router_sha256')
    gates, _, gate_path, gate_sha = checked(GATES, 'gates.pt', 'gates_sha256')
    multi, multi_report, multi_path, multi_sha = checked(MULTI, 'locator.pt', 'locator_sha256')
    multi_report_sha = p.sha(MULTI / 'result.json')
    cal_sha = p.sha(CAL)
    cal = json.loads(CAL.read_text(encoding='utf-8'))
    config_sha = feature_files_sha256(p.MODEL)
    if (base_sha != p.WEIGHT_SHA or any(x['base_sha256'] != base_sha for x in (router, gates, multi))
            or router['checkpoint_sha256'] != ckpt_sha or gates['checkpoint_sha256'] != ckpt_sha
            or multi_report['checkpoint_sha256'] != ckpt_sha or gates['router_sha256'] != router_sha
            or any(x['feature_config_sha256'] != config_sha for x in (router, gates, multi))
            or any(x['hidden_indices'] != [6, 18] for x in (router, gates, multi))
            or cal['gates_sha256'] != gate_sha or cal['thresholds']['read'] != gates['thresholds']['read']
            or cal['thresholds']['single_current'] != gates['thresholds']['single']):
        raise ValueError('Research provenance mismatch')
    if len({key for key, _, _ in RECORDS}) != len(RECORDS):
        raise ValueError('Duplicate controlled key')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix='spora-explicit-span-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        base = p.load_full_streamed().eval()
        original_layers = {i: base.model.layers[i] for i in (6, 12)}
        chat = p.runtime.load_chat(base_loader=lambda: base)
        source_state = chat.state
        source_digest = p.frozen_digest(chat.organism, source_state)
        replacements = {i: base.model.layers[i] for i in original_layers}
        for i in original_layers:
            base.model.layers[i] = original_layers[i]
        if address_basis_sha256(chat.organism) != router['address_basis_sha256']:
            raise ValueError('Address basis mismatch')
        chat.tokenizer.padding_side = 'right'
        state = p.fresh_controlled_state(chat.organism)
        fingerprints = IdentityFingerprints(chat.organism.config.key_count).to(next(chat.organism.parameters()).device)
        for key, name, value in RECORDS:
            fingerprints.bind(key, name)
            state, decision = p.observe_sourced(chat.organism, state, key, value, 'direct', 1.0)
            if not decision['current_write']:
                raise ValueError('Temporary write rejected')
        state_digest = p.frozen_digest(chat.organism, state)
        fingerprint_state = {k: v.clone() for k, v in fingerprints.state_dict().items()}
        guard = IdentityReadGuard(fingerprints, chat.tokenizer)
        bases = torch.stack([F.normalize(site.current_basis.detach().cpu().float(), dim=-1)
                             for site in chat.organism.memories])

        def linear(state_dict):
            head = nn.Linear(state_dict['weight'].shape[1], 1).eval()
            head.load_state_dict(state_dict, strict=True)
            return head

        locator, secondary = linear(router['locator_state']), linear(multi['state'])
        gate_models = {label: linear(gates['states'][label]) for label in ('single', 'read')}
        for group, question, selected_name, target in CASES:
            span = None
            if selected_name is not None:
                if question.count(selected_name) != 1:
                    raise ValueError('Manual selection must occur once')
                start = question.index(selected_name)
                span = [start, start + len(selected_name)]
            explicit = explicit_span_address(fingerprints, question, span, selected_name)
            features, masks = extract(base, chat.tokenizer, [question], batch_size=1)
            qvec = pool(features, masks).float()
            vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],))
                                 for i in range(2)], dim=-1)
            with torch.inference_mode():
                scores = {k: float(model(qvec).squeeze()) for k, model in gate_models.items()}
                first, one = single_span_mask(locator(vectors).squeeze(-1).sigmoid()[0], masks[0], router['locator_threshold'])
                second, two = single_span_mask(secondary(vectors).squeeze(-1).sigmoid()[0], masks[0], multi['threshold'])
            gates_pass = (scores['single'] >= cal['thresholds']['single_joint']
                          and scores['read'] >= cal['thresholds']['read'])
            learned_key = None
            if gates_pass and spans_agree(first, second, one, two):
                indices, similarities = predict(pooled(features, first.unsqueeze(0)), router['projection'], bases)
                if bool(indices[0, 0] == indices[0, 1] and (similarities[0] >= router['address_threshold']).all()):
                    candidate = int(indices[0, 0])
                    if guard.authorize(question, first, candidate):
                        learned_key = candidate
            row = {'group': group, 'question': question, 'selected_name': selected_name, 'span': span,
                   'expected_key': target, 'scores': scores, 'gates_pass': gates_pass,
                   'learned_key_before_read': learned_key, 'explicit': explicit,
                   'memory_accessed': False, 'answer': None, 'result_key': None}
            if gates_pass and explicit['key'] is not None:
                key = explicit['key']
                row['memory_accessed'] = True
                status = p.runtime.current_memory_status(chat.organism, state, key)
                row['memory_status'] = status['status']
                if status['status'] == 'known':
                    args, evidence = p.runtime._prepare_memory_read(chat.organism, state, key, 'current', None)
                    if p.runtime._supported_beverage_evidence(evidence, chat.organism.config.value_count):
                        prefix = closed_think_inputs(chat.tokenizer, question, 'cuda')
                        with torch.inference_mode():
                            ids = chat.decoder.generate(**prefix, **args, max_new_tokens=24, do_sample=False,
                                                        pad_token_id=chat.tokenizer.pad_token_id,
                                                        eos_token_id=chat.tokenizer.eos_token_id)
                        row['answer'] = chat.tokenizer.decode(ids[0, prefix['input_ids'].shape[1]:],
                                                              skip_special_tokens=True).strip()
                        row['result_key'] = key
                        row['native_value_id'] = int(evidence.argmax(-1)[0])
            rows.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
        for i in original_layers:
            base.model.layers[i] = replacements[i]
        if (p.frozen_digest(chat.organism, source_state) != source_digest
                or p.frozen_digest(chat.organism, state) != state_digest
                or any(not torch.equal(v, fingerprints.state_dict()[k]) for k, v in fingerprint_state.items())):
            raise ValueError('Protected in-memory state changed')
    for path, digest in ((p.MODEL / 'model.safetensors', base_sha), (p.runtime.CHECKPOINT, ckpt_sha),
                         (router_path, router_sha), (gate_path, gate_sha), (multi_path, multi_sha),
                         (MULTI / 'result.json', multi_report_sha), (CAL, cal_sha)):
        if p.sha(path) != digest:
            raise ValueError('Protected artifact changed')
    report = {'rows': rows, 'base_sha256': base_sha, 'checkpoint_sha256': ckpt_sha, 'router_sha256': router_sha,
              'gates_sha256': gate_sha, 'multi_sha256': multi_sha, 'calibration_sha256': cal_sha,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Hand-selected spans and manually enrolled names; input is not model extraction.',
                              'Selection is neither authentication nor proof of intended person.',
                              'Existing intent and single-entity gates, no real user or independent benchmark.',
                              'Known values manually written by numeric key into temporary state.',
                              'Learned route measured before memory read; explicit route actually reads.',
                              'No historical queries, persistence, aliases, homonyms or normal chat evaluation.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
