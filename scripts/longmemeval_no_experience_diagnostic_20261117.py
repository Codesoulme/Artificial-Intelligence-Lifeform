"""Seen-case no-experience diagnostic; deliberately NOT a memory benchmark.

Neither the original model nor the SPORA candidate receives the sessions. The
candidate's general conversation writer is not available. This script never
reads gold.private.jsonl and must not be described as end-to-end memory QA.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from transformers import AutoTokenizer

from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.language_training import frozen_digest
from scripts.spora_unknown_state_probe import fresh_controlled_state

INPUT = ROOT / 'runs/research/longmemeval-oracle-prepared-20261117/input.jsonl'
OUT = ROOT / 'runs/longmemeval-no-experience-diagnostic-20261117.json'
# These are the five cases already inspected while auditing the data source.
CASE_IDS = ('case-0000', 'case-0001', 'case-0054', 'case-0122', 'case-0123')


def select_cases(path):
    """Load public input only. Reject duplicates or missing predetermined cases."""
    cases = {}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line)
            if item['case_id'] in CASE_IDS:
                if item['case_id'] in cases:
                    raise ValueError('Duplicate case')
                cases[item['case_id']] = item
    if set(cases) != set(CASE_IDS):
        raise ValueError('Missing case')
    return [cases[case_id] for case_id in CASE_IDS]


def base_answer(base, tokenizer, question):
    inputs = closed_think_inputs(tokenizer, question, 'cuda')
    with torch.inference_mode():
        ids = base.generate(**inputs, max_new_tokens=48, do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id)
    tail = ids[0, inputs['input_ids'].shape[1]:]
    return {'text': tokenizer.decode(tail, skip_special_tokens=True).strip(),
            'generated_tokens': int(tail.numel()),
            'ended_with_eos': bool(tail.numel() and int(tail[-1]) == tokenizer.eos_token_id)}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    cases = select_cases(INPUT)
    sources = {'input': INPUT, 'base': runtime.MODEL / 'model.safetensors',
               'integrated': runtime.CHECKPOINT}
    hashes = {key: runtime.sha(path) for key, path in sources.items()}
    if hashes['base'] != runtime.WEIGHT_SHA:
        raise ValueError('Base hash mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(runtime.MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed()
    rows = []
    for case in cases:
        baseline = base_answer(base, tokenizer, case['question'])
        rows.append({'case_id': case['case_id'], 'question': case['question'],
                     'session_count_not_ingested': len(case['sessions']),
                     'ingestion': 'not_attempted_unsupported_general_conversation_writer',
                     'baseline_question_only': baseline})
        print(json.dumps({'stage': 'base', 'case_id': case['case_id']}, ensure_ascii=False), flush=True)
    with tempfile.TemporaryDirectory(prefix='spora-lme-no-exp-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: base)
        original = chat.state
        original_digest = frozen_digest(chat.organism, original)
        chat.state = fresh_controlled_state(chat.organism)
        fresh_digest = frozen_digest(chat.organism, chat.state)
        for row in rows:
            row['integrated_router_question_only'] = chat.answer(row['question'])
            print(json.dumps({'stage': 'route', 'case_id': row['case_id'],
                              'reason': row['integrated_router_question_only']['reason']}), flush=True)
        if (frozen_digest(chat.organism, original) != original_digest or
                frozen_digest(chat.organism, chat.state) != fresh_digest):
            raise ValueError('Unexpected individual state change')
    if any(runtime.sha(path) != hashes[key] for key, path in sources.items()):
        raise ValueError('Protected source changed')
    report = {'status': 'seen_case_no_experience_diagnostic_not_memory_evaluation',
              'rows': rows, 'source_sha256': hashes, 'seconds': time.perf_counter() - start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Both paths receive questions only; no sessions are ingested.',
                              'Five previously inspected examples; no gold scoring or generalization claim.',
                              'Baseline uses original language generation; candidate uses a constrained memory router.',
                              'Paths have different output semantics; results are not a matched capability comparison.',
                              'Output budget 48 tokens; truncated generation is recorded explicitly.',
                              'A fresh isolated individual is used; no production state is modified.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'completed': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
