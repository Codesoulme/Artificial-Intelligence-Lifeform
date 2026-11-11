"""Evaluate an opt-in identity check before a temporary individual's reads."""
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_boundary_runtime_pilot import ENROLLMENTS, OUTPUT as PREVIOUS
from scripts.deepseek_spora_boundary_train_pilot import OUTPUT as CANDIDATE
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.identity_fingerprints import IdentityFingerprints
from spora.identity_read_guard import IdentityReadGuard
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-preread-guard-pilot-20261112.json'
CASES = [
    ('known', '小林现在想喝哪种饮料？', 0), ('known', '阿宁现在想喝哪种饮料？', 1),
    ('known', '小雨现在想喝哪种饮料？', 2), ('known', '阿哲现在想喝哪种饮料？', 3),
    ('unknown', '小林华现在想喝哪种饮料？', None), ('unknown', '阿宁悦现在想喝哪种饮料？', None),
    ('unknown', '小雨晴现在想喝哪种饮料？', None), ('unknown', '阿哲轩现在想喝哪种饮料？', None),
    ('unknown', '小林子现在想喝哪种饮料？', None), ('unknown', '林小林现在想喝哪种饮料？', None),
    ('unknown', '小小雨现在想喝哪种饮料？', None), ('unknown', '哲阿哲现在想喝哪种饮料？', None),
    ('multiple', '小林和阿宁现在喜欢喝什么饮料？', None),
    ('multiple', '阿哲与小雨现在各喜欢什么饮料？', None),
    ('none', '现在有什么饮料可以推荐？', None), ('none', '你能推荐什么饮料？', None),
]


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_sha = runtime.sha(runtime.CHECKPOINT)
    previous = json.loads(PREVIOUS.read_text(encoding='utf-8'))
    candidate_path = CANDIDATE / 'locator.pt'
    candidate_sha = runtime.sha(candidate_path)
    candidate_report = json.loads((CANDIDATE / 'result.json').read_text(encoding='utf-8'))
    if (previous['source_checkpoint_sha256'] != source_sha or previous['candidate_sha256'] != candidate_sha
            or candidate_report['locator_sha256'] != candidate_sha):
        raise ValueError('Reference or candidate digest mismatch')
    candidate = torch.load(candidate_path, map_location='cpu', weights_only=True)
    torch.set_num_threads(2)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        runtime.AutoModelForCausalLM.from_pretrained = lambda *args, **kwargs: load_full_streamed()
        chat = runtime.load_chat()
        if candidate['feature_variant_sha256'] != chat.router['feature_variant_sha256']:
            raise ValueError('Locator feature mismatch')
        chat.router = chat.router.copy()
        chat.router['locator_state'] = candidate['state']
        chat.router['locator_threshold'] = candidate['threshold']
        chat.organism.identity_fingerprints = IdentityFingerprints(chat.organism.config.key_count).to(
            next(chat.organism.parameters()).device)
        chat.state = fresh_controlled_state(chat.organism)
        for key, name, value in ENROLLMENTS:
            chat.organism.identity_fingerprints.bind(key, name)
            chat.state, decision = observe_sourced(chat.organism, chat.state, key, value, 'direct', 1.0)
            if not decision['current_write']:
                raise ValueError('Temporary write rejected')
        guard = IdentityReadGuard(chat.organism.identity_fingerprints, chat.tokenizer)
        before = frozen_digest(chat.organism, chat.state)
        rows = []
        for group, question, expected_index in CASES:
            with patch.object(runtime, 'current_memory_status', wraps=runtime.current_memory_status) as status, \
                 patch.object(chat.decoder, 'generate', wraps=chat.decoder.generate) as generate:
                result = chat.answer(question, identity_guard=guard)
            expected_key = None if expected_index is None else ENROLLMENTS[expected_index][0]
            expected_value = None if expected_index is None else WORDS[ENROLLMENTS[expected_index][2]]
            row = {'group': group, 'question': question, 'result': result,
                   'memory_status_calls': status.call_count, 'generation_calls': generate.call_count,
                   'correct_key': bool(result['accepted'] and expected_key is not None and result.get('key') == expected_key),
                   'correct_answer': bool(result['accepted'] and expected_key is not None and result.get('key') == expected_key
                                          and result.get('answer') == expected_value)}
            if result['reason'] == 'identity_guard_rejected' and (status.call_count or generate.call_count):
                raise ValueError('Guard rejection occurred after memory access or generation')
            rows.append(row)
            print(json.dumps({'group': group, 'reason': result['reason'],
                              'status_calls': status.call_count, 'generation_calls': generate.call_count}), flush=True)
        if frozen_digest(chat.organism, chat.state) != before:
            raise ValueError('Temporary individual changed during reads')
    if runtime.sha(runtime.CHECKPOINT) != source_sha or runtime.sha(candidate_path) != candidate_sha:
        raise ValueError('Protected input changed')
    summary = {}
    for group in ('known', 'unknown', 'multiple', 'none'):
        subset = [r for r in rows if r['group'] == group]
        summary[group] = {'total': len(subset), 'accepted': sum(r['result']['accepted'] for r in subset),
                          'guard_rejected': sum(r['result']['reason'] == 'identity_guard_rejected' for r in subset),
                          'correct_key': sum(r['correct_key'] for r in subset),
                          'correct_answer': sum(r['correct_answer'] for r in subset),
                          'memory_status_calls': sum(r['memory_status_calls'] for r in subset),
                          'generation_calls': sum(r['generation_calls'] for r in subset)}
    result = {'status': 'experimental_opt_in_pre_read_guard', 'summary': summary, 'rows': rows,
              'source_checkpoint_sha256': source_sha, 'candidate_sha256': candidate_sha,
              'seconds': time.perf_counter() - started,
              'limitations': ['Sixteen authored questions; eight repeat the earlier development pilot.',
                              'Names, numerical keys and memory values manually enrolled in a temporary individual.',
                              'Exact fingerprint comparison cannot identify truncation, aliases or same-name people.',
                              'Multiple/no-person tests are small and do not establish broad refusal.',
                              'The guard is opt-in and not connected to the default local entry point.',
                              'Status-call and generation instrumentation does not cover all possible future code paths.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
