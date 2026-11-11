"""Read-only runtime comparison of two locator heads in a temporary individual."""
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
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_boundary_train_pilot import OUTPUT as CANDIDATE
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_full_stream_parity_audit import OUTPUT as PARITY
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.identity_fingerprints import IdentityFingerprints
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-boundary-runtime-pilot-20261112.json'
ENROLLMENTS = [(KEYS[0], '小林', 6), (KEYS[1], '阿宁', 1),
               (KEYS[2], '小雨', 2), (KEYS[3], '阿哲', 3)]
CASES = [('known', '小林现在想喝哪种饮料？', 0), ('known', '阿宁现在想喝哪种饮料？', 1),
         ('known', '小雨现在想喝哪种饮料？', 2), ('known', '阿哲现在想喝哪种饮料？', 3),
         ('unknown', '小林华现在想喝哪种饮料？', None), ('unknown', '阿宁悦现在想喝哪种饮料？', None),
         ('unknown', '小雨晴现在想喝哪种饮料？', None), ('unknown', '阿哲轩现在想喝哪种饮料？', None)]


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checkpoint_sha = runtime.sha(runtime.CHECKPOINT)
    parity = json.loads(PARITY.read_text(encoding='utf-8'))
    if (parity['source_checkpoint_sha256'] != checkpoint_sha or len(parity['rows']) != 2
            or not all(r['same_fields'] and r['entity_absolute_difference'] < 0.001
                       and r['intent_absolute_difference'] < 0.001 for r in parity['rows'])):
        raise ValueError('Full stream parity mismatch')
    candidate_path = CANDIDATE / 'locator.pt'
    candidate_sha = runtime.sha(candidate_path)
    candidate_report = json.loads((CANDIDATE / 'result.json').read_text(encoding='utf-8'))
    if candidate_report['locator_sha256'] != candidate_sha:
        raise ValueError('Candidate digest mismatch')
    candidate = torch.load(candidate_path, map_location='cpu', weights_only=True)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        runtime.AutoModelForCausalLM.from_pretrained = lambda *args, **kwargs: load_full_streamed()
        chat = runtime.load_chat()
        if candidate['feature_variant_sha256'] != chat.router['feature_variant_sha256']:
            raise ValueError('Candidate feature variant mismatch')
        fingerprints = IdentityFingerprints(chat.organism.config.key_count)
        chat.state = fresh_controlled_state(chat.organism)
        for key, name, value in ENROLLMENTS:
            fingerprints.bind(key, name)
            chat.state, decision = observe_sourced(chat.organism, chat.state, key, value, 'direct', 1.0)
            if not decision['current_write']:
                raise ValueError('Temporary memory write rejected')
        before = frozen_digest(chat.organism, chat.state)
        questions = [q for _, q, _ in CASES]
        texts = [chat.tokenizer.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False,
                                                    add_generation_prompt=True) for q in questions]
        offsets = chat.tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False,
                                 return_offsets_mapping=True)['offset_mapping']
        features, masks = runtime.extract(chat.base, chat.tokenizer, questions, batch_size=1)
        vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
        default_router = chat.router.copy()
        candidate_router = chat.router.copy()
        candidate_router['locator_state'] = candidate['state']
        candidate_router['locator_threshold'] = candidate['threshold']
        heads = []
        for router in (default_router, candidate_router):
            head = nn.Linear(vectors.shape[-1], 1).eval()
            head.load_state_dict(router['locator_state'], strict=True)
            heads.append(head)
        with torch.no_grad():
            scores = [head(vectors).squeeze(-1).sigmoid() for head in heads]
        rows = []
        for i, (group, question, expected_index) in enumerate(CASES):
            row = {'group': group, 'question': question, 'expected_key': None if expected_index is None else ENROLLMENTS[expected_index][0]}
            start = texts[i].find(question)
            for label, router, score in zip(('default', 'candidate'), (default_router, candidate_router), scores):
                selected, single = single_span_mask(score[i], masks[i], router['locator_threshold'])
                chars = set()
                for token in selected.nonzero(as_tuple=True)[0].tolist():
                    left, right = offsets[i, token].tolist()
                    chars.update(range(max(0, left - start), min(len(question), right - start)))
                picked = ''.join(question[j] for j in sorted(chars))
                try:
                    identity_key = fingerprints.lookup(picked) if single else None
                except ValueError:
                    identity_key = None
                chat.router = router
                actual = chat.answer(question)
                accepted = bool(actual['accepted'])
                gated = bool(accepted and identity_key is not None and actual.get('key') == identity_key)
                expected_key = row['expected_key']
                expected_value = None if expected_index is None else WORDS[ENROLLMENTS[expected_index][2]]
                row[label] = {'actual': actual, 'picked': picked, 'fingerprint_key': identity_key,
                              'offline_gated': gated,
                              'correct_key': bool(accepted and expected_key is not None and actual.get('key') == expected_key),
                              'correct_answer': bool(accepted and expected_key is not None and actual.get('key') == expected_key
                                                     and actual.get('answer') == expected_value)}
            rows.append(row)
            print(json.dumps({'group': group, 'default': row['default']['actual']['reason'],
                              'candidate': row['candidate']['actual']['reason']}), flush=True)
        if frozen_digest(chat.organism, chat.state) != before:
            raise ValueError('Read changed temporary individual')
    if runtime.sha(runtime.CHECKPOINT) != checkpoint_sha or runtime.sha(candidate_path) != candidate_sha:
        raise ValueError('Protected input changed')
    summary = {}
    for group in ('known', 'unknown'):
        subset = [row for row in rows if row['group'] == group]
        summary[group] = {'total': len(subset)}
        for label in ('default', 'candidate'):
            summary[group][label] = {'accepted': sum(r[label]['actual']['accepted'] for r in subset),
                                     'correct_key': sum(r[label]['correct_key'] for r in subset),
                                     'correct_answer': sum(r[label]['correct_answer'] for r in subset),
                                     'offline_gated': sum(r[label]['offline_gated'] for r in subset)}
    report = {'status': 'temporary_individual_runtime_development_pilot', 'summary': summary, 'rows': rows,
              'candidate_sha256': candidate_sha, 'source_checkpoint_sha256': checkpoint_sha,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Eight authored questions and four hand-enrolled names; not an independent benchmark.',
                              'Fingerprint veto is applied after the runtime reads and generates; not a real protection.',
                              'Default and candidate locator heads share all other runtime components.',
                              'Temporary controlled fact writes use numerical keys, not language understanding.',
                              'No real user individual opened or modified; experimental loader parity checked on two cases.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
