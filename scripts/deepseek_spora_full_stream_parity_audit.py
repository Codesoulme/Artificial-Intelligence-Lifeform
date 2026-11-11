"""Compare streamed full-model loading with saved runtime reference outputs."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_named_binding_probe import KEYS
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest

REFERENCE = ROOT / 'runs/deepseek-spora-default-near-name-audit-20261111.json'
OUTPUT = ROOT / 'runs/deepseek-spora-full-stream-parity-20261112.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    reference = json.loads(REFERENCE.read_text(encoding='utf-8'))
    checkpoint_sha = runtime.sha(runtime.CHECKPOINT)
    if reference['source_checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Reference checkpoint mismatch')
    torch.manual_seed(9814)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        # Override only in this process; regular runtime and original checkpoint remain unchanged.
        runtime.AutoModelForCausalLM.from_pretrained = lambda *args, **kwargs: load_full_streamed()
        chat = runtime.load_chat()
        chat.state = fresh_controlled_state(chat.organism)
        chat.state, decision = observe_sourced(chat.organism, chat.state, KEYS[0], 6, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('Controlled write rejected')
        before = frozen_digest(chat.organism, chat.state)
        rows = []
        # One known and one near-name negative; earlier full model reference had accepted both.
        for index in (0, 2):
            expected = reference['rows'][index]
            result = chat.answer(expected['question'])
            fields = ('accepted', 'reason', 'key', 'answer', 'temporal_intent')
            row = {'reference_index': index, 'same_fields': all(result.get(k) == expected.get(k) for k in fields),
                   'entity_absolute_difference': abs(result['entity_score'] - expected['entity_score']),
                   'intent_absolute_difference': abs(result['intent_logit'] - expected['intent_logit']),
                   'reference_accepted': expected['accepted'], 'actual_accepted': result['accepted'],
                   'reference_reason': expected['reason'], 'actual_reason': result['reason']}
            rows.append(row)
            print(json.dumps(row), flush=True)
        if frozen_digest(chat.organism, chat.state) != before:
            raise ValueError('Read changed the temporary individual')
    if runtime.sha(runtime.CHECKPOINT) != checkpoint_sha:
        raise ValueError('Source checkpoint changed')
    report = {'status': 'full_streamed_load_development_parity', 'rows': rows,
              'source_checkpoint_sha256': checkpoint_sha, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Two synthetic examples only, not general generation or hidden-state equivalence.',
                              'Near-name acceptance is a known incorrect routing behavior, not a passing safety result.',
                              'Temporary memory state and process-only loader override; no production change.',
                              'Saved reference is from an earlier run, not a parallel full-weight comparison.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'same_fields': sum(r['same_fields'] for r in rows), 'total': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
