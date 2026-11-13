"""Research-only end-to-end current-fact audit on original language layers."""
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
from scripts.deepseek_spora_original_layer_name_router import OUT as ROUTER
from scripts.deepseek_spora_original_layer_request_gates import OUT as GATES, CHALLENGE
from spora.identity_fingerprints import IdentityFingerprints
from spora.identity_read_guard import IdentityReadGuard
from spora.named_address import address_basis_sha256, feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-spora-original-layer-joint-runtime-20261114.json'
VALUES = (1, 2, 3, 4, 5, 6, 7, 0)
FRESH = (
    ('fresh_known', '请告诉我小林现在喜欢哪种饮品？', 0),
    ('fresh_known', '阿宁目前喜欢喝什么饮料？', 1),
    ('fresh_known', '请查小雨的饮料偏好。', 2),
    ('fresh_known', '对于阿哲，记录里现在的饮料是什么？', 3),
    ('fresh_known', '你知道小梅目前爱喝哪种饮料吗？', 4),
    ('fresh_known', '阿远现在偏爱什么饮品？', 5),
)


def checked(folder, name, digest_key):
    path = folder / name
    digest = p.sha(path)
    info = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    if info[digest_key] != digest:
        raise ValueError('Research artifact checksum mismatch')
    return torch.load(path, map_location='cpu', weights_only=True), info, path, digest


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    router, prior, router_path, router_sha = checked(ROUTER, 'router.pt', 'router_sha256')
    gates, _, gate_path, gate_sha = checked(GATES, 'gates.pt', 'gates_sha256')
    config_sha = feature_files_sha256(p.MODEL)
    if (base_sha != p.WEIGHT_SHA or router['base_sha256'] != base_sha
            or gates['base_sha256'] != base_sha or router['checkpoint_sha256'] != checkpoint_sha
            or gates['checkpoint_sha256'] != checkpoint_sha or gates['router_sha256'] != router_sha
            or router['feature_config_sha256'] != config_sha or gates['feature_config_sha256'] != config_sha
            or router['hidden_indices'] != [6, 18] or gates['hidden_indices'] != [6, 18]):
        raise ValueError('Research artifact provenance mismatch')
    cases = [(row['group'], row['question'], KEYS.index(row['expected_key']) if row['expected_key'] is not None else None)
             for row in prior['rows']]
    cases += [('challenge_' + group, question, None) for group, question, _, _ in CHALLENGE]
    cases += list(FRESH)
    if len({q for _, q, _ in cases}) != len(cases):
        raise ValueError('Duplicate audit case')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix='spora-original-joint-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        base = p.load_full_streamed().eval()
        original_layers = {index: base.model.layers[index] for index in (6, 12)}
        chat = p.runtime.load_chat(base_loader=lambda: base)
        original_state = chat.state
        source_digest = p.frozen_digest(chat.organism, original_state)
        replacements = {index: base.model.layers[index] for index in original_layers}
        for index in original_layers:
            base.model.layers[index] = original_layers[index]
        if address_basis_sha256(chat.organism) != router['address_basis_sha256']:
            raise ValueError('Router address basis mismatch')
        chat.tokenizer.padding_side = 'right'
        state = p.fresh_controlled_state(chat.organism)
        fingerprints = IdentityFingerprints(chat.organism.config.key_count).to(next(chat.organism.parameters()).device)
        for i, name in enumerate(NAMES):
            fingerprints.bind(KEYS[i], name)
            state, decision = p.observe_sourced(chat.organism, state, KEYS[i], VALUES[i], 'direct', 1.0)
            if not decision['current_write']:
                raise ValueError('Controlled write rejected')
        state_digest = p.frozen_digest(chat.organism, state)
        guard = IdentityReadGuard(fingerprints, chat.tokenizer)
        bases = torch.stack([F.normalize(site.current_basis.detach().cpu().float(), dim=-1)
                             for site in chat.organism.memories])
        gate_models = {}
        for label in ('single', 'read'):
            weight = gates['states'][label]['weight']
            gate_models[label] = nn.Linear(weight.shape[1], 1).eval()
            gate_models[label].load_state_dict(gates['states'][label], strict=True)
        locator_weight = router['locator_state']['weight']
        locator = nn.Linear(locator_weight.shape[1], 1).eval()
        locator.load_state_dict(router['locator_state'], strict=True)
        for group, question, expected_index in cases:
            features, masks = extract(base, chat.tokenizer, [question], batch_size=1)
            qvec = pool(features, masks).float()
            token_vec = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],))
                                   for i in range(2)], dim=-1)
            with torch.inference_mode():
                scores = {label: float(model(qvec).squeeze()) for label, model in gate_models.items()}
                selected, single = single_span_mask(locator(token_vec).squeeze(-1).sigmoid()[0],
                                                    masks[0], router['locator_threshold'])
            row = {'group': group, 'question': question, 'expected_key': None if expected_index is None else KEYS[expected_index],
                   'expected_value': None if expected_index is None else p.WORDS[VALUES[expected_index]],
                   'scores': scores, 'accepted': False, 'answer': None, 'memory_accessed': False}
            if scores['single'] < gates['thresholds']['single']:
                row['reason'] = 'single_gate_rejected'
            elif scores['read'] < gates['thresholds']['read']:
                row['reason'] = 'read_gate_rejected'
            elif not single:
                row['reason'] = 'name_span_rejected'
            else:
                name_vector = pooled(features, selected.unsqueeze(0))
                indices, similarities = predict(name_vector, router['projection'], bases)
                row['indices'], row['similarities'] = indices[0].tolist(), similarities[0].tolist()
                if not bool(indices[0, 0] == indices[0, 1] and (similarities[0] >= router['address_threshold']).all()):
                    row['reason'] = 'address_rejected'
                else:
                    key = int(indices[0, 0])
                    if not guard.authorize(question, selected, key):
                        row['reason'] = 'identity_guard_rejected'
                    else:
                        row['memory_accessed'] = True
                        status = p.runtime.current_memory_status(chat.organism, state, key)
                        if status['status'] != 'known':
                            row['reason'] = 'unknown_memory'
                        else:
                            args, evidence = p.runtime._prepare_memory_read(chat.organism, state, key, 'current', None)
                            if not p.runtime._supported_beverage_evidence(evidence, chat.organism.config.value_count):
                                row['reason'] = 'unsupported_memory_value'
                            else:
                                prefix = closed_think_inputs(chat.tokenizer, question, 'cuda')
                                with torch.inference_mode():
                                    ids = chat.decoder.generate(**prefix, **args, max_new_tokens=24, do_sample=False,
                                                                pad_token_id=chat.tokenizer.pad_token_id,
                                                                eos_token_id=chat.tokenizer.eos_token_id)
                                tail = ids[0, prefix['input_ids'].shape[1]:].tolist()
                                row.update({'reason': 'routed', 'accepted': True, 'key': key,
                                            'native_value_id': int(evidence.argmax(-1)[0]), 'token_ids': tail,
                                            'answer': chat.tokenizer.decode(tail, skip_special_tokens=True).strip(),
                                            'ended': chat.tokenizer.eos_token_id in tail})
            row['correct_key'] = bool(expected_index is not None and row.get('key') == KEYS[expected_index]
                                      and row['accepted'])
            row['exact_answer'] = bool(row['correct_key'] and row['answer'].rstrip('。.!！ ') == row['expected_value'])
            rows.append(row)
            print(json.dumps({'done': len(rows), 'group': group, 'reason': row['reason'],
                              'answer': row['answer']}, ensure_ascii=True), flush=True)
        for index in original_layers:
            base.model.layers[index] = replacements[index]
        if p.frozen_digest(chat.organism, original_state) != source_digest or p.frozen_digest(chat.organism, state) != state_digest:
            raise ValueError('Protected memory state changed')
    for path, digest in ((p.MODEL / 'model.safetensors', base_sha), (p.runtime.CHECKPOINT, checkpoint_sha),
                         (router_path, router_sha), (gate_path, gate_sha)):
        if p.sha(path) != digest:
            raise ValueError('Protected artifact changed')
    summary = {}
    for group in sorted({row['group'] for row in rows}):
        subset = [row for row in rows if row['group'] == group]
        summary[group] = {'cases': len(subset), 'accepted': sum(row['accepted'] for row in subset),
                          'memory_accessed': sum(row['memory_accessed'] for row in subset),
                          'correct_key': sum(row['correct_key'] for row in subset),
                          'exact_answer': sum(row['exact_answer'] for row in subset)}
    report = {'summary': summary, 'rows': rows, 'base_sha256': base_sha, 'checkpoint_sha256': checkpoint_sha,
              'router_sha256': router_sha, 'gates_sha256': gate_sha, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Research-only current-value route; no time classification or historical questions.',
                              'Manually enrolled names and numerical fact writes; not natural learning.',
                              'Small authored development questions reused from prior experiments, not independent.',
                              'Single identity guard can accept truncated names or fail on homonyms.',
                              'Exact short answer matching may miss valid paraphrases.',
                              'No default entry point, base weights, checkpoint or real individual changed.']}
    with OUTPUT.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
