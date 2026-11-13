"""Exercise explicit-name gating on temporary controlled runtime state."""
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.confirmed_name_read import answer_with_confirmed_name
from spora.identity_fingerprints import IdentityFingerprints
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-confirmed-name-runtime-20261114.json'
SHORT = '小林现在想喝哪种饮料？'
LONG = '小林华现在想喝哪种饮料？'
CASES = (('missing_confirmation', SHORT, None),
         ('unknown_confirmation', LONG, '小林舟'),
         ('confirmed_known', SHORT, '小林'),
         ('confirmed_long', LONG, '小林华'),
         ('mistaken_short_confirmation', LONG, '小林'))


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base_sha = runtime.sha(runtime.MODEL / 'model.safetensors')
    checkpoint_sha = runtime.sha(runtime.CHECKPOINT)
    if base_sha != runtime.WEIGHT_SHA:
        raise ValueError('Base checksum mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix='spora-confirmed-name-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        source_state = chat.state
        source_digest = frozen_digest(chat.organism, source_state)
        chat.state = fresh_controlled_state(chat.organism)
        fingerprints = IdentityFingerprints(chat.organism.config.key_count)
        for key, name, value in ((6, '小林', 6), (8, '小林华', 1)):
            fingerprints.bind(key, name)
            chat.state, decision = observe_sourced(chat.organism, chat.state, key, value, 'direct', 1.0)
            if not decision['current_write']:
                raise ValueError('Controlled write rejected')
        # Local temporary object only; the source checkpoint is never saved.
        chat.organism.identity_fingerprints = fingerprints
        temporary_digest = frozen_digest(chat.organism, chat.state)
        for group, question, name in CASES:
            with patch.object(runtime, 'current_memory_status', wraps=runtime.current_memory_status) as status, \
                 patch.object(runtime, '_prepare_memory_read', wraps=runtime._prepare_memory_read) as read:
                actual = answer_with_confirmed_name(chat, question, name)
            row = {'group': group, 'question': question, 'confirmed_name': name, 'result': actual,
                   'memory_status_calls': status.call_count, 'memory_read_calls': read.call_count}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
        if frozen_digest(chat.organism, chat.state) != temporary_digest:
            raise ValueError('Temporary individual changed during reads')
        del chat.organism.identity_fingerprints
        # The source state is separate and was never used for controlled writes.
        if frozen_digest(chat.organism, source_state) != source_digest:
            raise ValueError('Source individual changed')
    if (runtime.sha(runtime.MODEL / 'model.safetensors') != base_sha
            or runtime.sha(runtime.CHECKPOINT) != checkpoint_sha):
        raise ValueError('Protected file changed')
    result = {'rows': rows, 'base_sha256': base_sha, 'checkpoint_sha256': checkpoint_sha,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Experimental older replacement-layer runtime, not original-layer route.',
                              'Two hand-enrolled names and synthetic numeric writes, five development cases.',
                              'Name confirmation is not authentication and can be wrong or dishonest.',
                              'No homonym, alias, concurrency, persistence or natural-learning evaluation.',
                              'No source checkpoint or real individual was modified.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
