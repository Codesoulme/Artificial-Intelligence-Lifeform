"""Audit a fingerprint veto using default runtime's predicted subject spans.

This is an offline diagnostic. The individual and enrollment are temporary;
the fingerprint module is not installed in the default read path.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch import nn
from torch.nn import functional as F
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_learned_name_span import single_span_mask
from scripts.deepseek_named_binding_probe import KEYS
from scripts.deepseek_spora_default_near_name_audit import CASES
from scripts.deepseek_spora_cluener_locator_retrain import OUTPUT as CANDIDATE_LOCATOR
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.identity_fingerprints import IdentityFingerprints
from spora.internal_memory import observe_sourced

OUTPUT = ROOT / 'runs/deepseek-spora-fingerprint-route-audit-v2-20261111.json'
ENROLLMENTS = [(KEYS[0], '小林', 6)]
EXPECTED_KEYS = {question: KEYS[0] for group, question in CASES if group == 'known'}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.set_num_threads(2)
    source_sha = runtime.sha(runtime.CHECKPOINT)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = runtime.load_chat()
        device = next(chat.organism.parameters()).device
        chat.organism.identity_fingerprints = IdentityFingerprints(chat.organism.config.key_count).to(device)
        for key, name, _ in ENROLLMENTS:
            chat.organism.identity_fingerprints.bind(key, name)
        chat.state = fresh_controlled_state(chat.organism)
        for key, _, value in ENROLLMENTS:
            chat.state, decision = observe_sourced(chat.organism, chat.state, key, value, 'direct', 1.0)
            if not decision['current_write']:
                raise ValueError('Controlled initialization refused')
        questions = [q for _, q in CASES]
        texts = [chat.tokenizer.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False,
                                                    add_generation_prompt=True) for q in questions]
        encoded = chat.tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False,
                                 return_offsets_mapping=True)
        features, masks = runtime.extract(chat.base, chat.tokenizer, questions, batch_size=2)
        vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
        locator = nn.Linear(vectors.shape[-1], 1).eval()
        locator.load_state_dict(chat.router['locator_state'], strict=True)
        candidate_path = CANDIDATE_LOCATOR / 'locator.pt'
        candidate_report = json.loads((CANDIDATE_LOCATOR / 'result.json').read_text(encoding='utf-8'))
        candidate_sha = runtime.sha(candidate_path)
        if candidate_sha != candidate_report['locator_sha256']:
            raise ValueError('Candidate locator digest mismatch')
        candidate_data = torch.load(candidate_path, map_location='cpu', weights_only=True)
        if candidate_data.get('feature_variant_sha256') != chat.router.get('feature_variant_sha256'):
            raise ValueError('Candidate locator feature variant mismatch')
        candidate_locator = nn.Linear(vectors.shape[-1], 1).eval()
        candidate_locator.load_state_dict(candidate_data['state'], strict=True)
        with torch.no_grad():
            scores = locator(vectors).squeeze(-1).sigmoid()
            candidate_scores = candidate_locator(vectors).squeeze(-1).sigmoid()
        rows = []
        for i, (group, question) in enumerate(CASES):
            text_start = texts[i].find(question)
            def fingerprint_for(score, threshold):
                selected, single = single_span_mask(score, masks[i], threshold)
                chars = set()
                for token in selected.nonzero(as_tuple=True)[0].tolist():
                    lo, hi = encoded['offset_mapping'][i, token].tolist()
                    chars.update(range(max(0, lo - text_start), min(len(question), hi - text_start)))
                text = ''.join(question[j] for j in sorted(chars))
                try:
                    key = chat.organism.identity_fingerprints.lookup(text) if single else None
                except ValueError:
                    key = None
                return text, key
            selected_text, identity_key = fingerprint_for(scores[i], chat.router['locator_threshold'])
            candidate_text, candidate_key = fingerprint_for(candidate_scores[i], candidate_data['threshold'])
            result = chat.answer(question)
            gated = bool(result['accepted'] and identity_key is not None and result.get('key') == identity_key)
            candidate_gated = bool(result['accepted'] and candidate_key is not None and result.get('key') == candidate_key)
            rows.append({'group': group, 'question': question, 'expected_key': EXPECTED_KEYS.get(question),
                         'selected_text': selected_text,
                         'fingerprint_key': identity_key, 'runtime': result,
                         'candidate_selected_text': candidate_text, 'candidate_fingerprint_key': candidate_key,
                         'baseline_accepted': bool(result['accepted']), 'gated_accepted': gated,
                         'candidate_gated_accepted': candidate_gated})
            print(json.dumps({'question': question, 'selected_text': selected_text,
                              'runtime_key': result.get('key'), 'fingerprint_key': identity_key,
                              'gated': gated, 'candidate_selected_text': candidate_text,
                              'candidate_gated': candidate_gated}, ensure_ascii=False), flush=True)
    if runtime.sha(runtime.CHECKPOINT) != source_sha:
        raise ValueError('Protected bootstrap changed')
    known = [r for r in rows if r['group'] == 'known']
    unknown = [r for r in rows if r['group'] == 'unknown']
    summary = {'known_total': len(known),
               'baseline_known_correct_key': sum(r['baseline_accepted'] and r['runtime'].get('key') == r['expected_key'] for r in known),
               'gated_known_correct_key': sum(r['gated_accepted'] and r['runtime'].get('key') == r['expected_key'] for r in known),
               'candidate_gated_known_correct_key': sum(r['candidate_gated_accepted'] and r['runtime'].get('key') == r['expected_key'] for r in known),
               'unknown_total': len(unknown),
               'baseline_unknown_accepted': sum(r['baseline_accepted'] for r in unknown),
               'gated_unknown_accepted': sum(r['gated_accepted'] for r in unknown),
               'candidate_gated_unknown_accepted': sum(r['candidate_gated_accepted'] for r in unknown)}
    report = {'status': 'development_only_no_user_state', 'summary': summary, 'rows': rows,
              'source_checkpoint_sha256': source_sha, 'candidate_locator_sha256': candidate_sha,
              'seconds': time.perf_counter() - started,
              'limitations': ['Authored development questions; a diagnostic, not an independent population test.',
                              'Names and keys are manually enrolled into an in-memory temporary individual.',
                              'Token offsets reconstruct predicted spans; truncation can match the wrong enrolled name.',
                              'The veto runs after generation for measurement and is not integrated into runtime.',
                              'No real user state was opened or modified; no automatic enrollment or write.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
