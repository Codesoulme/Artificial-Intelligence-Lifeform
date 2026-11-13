"""Measure opt-in identity check in the real read path on temporary state."""
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import torch

from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_named_binding_probe import KEYS
from spora.identity_fingerprints import IdentityFingerprints
from spora.identity_read_guard import IdentityReadGuard

OUTPUT = p.ROOT / 'runs/deepseek-spora-pre-read-guard-span-audit-20261112.json'
NAMES = ('小林', '阿宁', '小雨', '阿哲')
CASES = [(f'{name}现在想喝哪种饮料？', i) for i, name in enumerate(NAMES)] + [
    (f'{name}{suffix}现在想喝哪种饮料？', None)
    for name, suffix in zip(NAMES, ('华', '悦', '晴', '轩'))]


class RecordingGuard(IdentityReadGuard):
    def authorize(self, question, selected, key):
        text = self.tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
            tokenize=False, add_generation_prompt=True)
        offsets = self.tokenizer([text], return_tensors='pt', padding=True, add_special_tokens=False,
                                 return_offsets_mapping=True)['offset_mapping'][0]
        start = text.find(question)
        indices = selected.nonzero(as_tuple=True)[0].tolist()
        pieces = [(int(offsets[i, 0]) - start, int(offsets[i, 1]) - start) for i in indices]
        chars = sorted({j for lo, hi in pieces for j in range(max(0, lo), min(len(question), hi))})
        self.last = {'selected_text': ''.join(question[i] for i in chars), 'character_positions': chars,
                     'key': key}
        self.last['authorized'] = super().authorize(question, selected, key)
        return self.last['authorized']


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_sha = p.sha(p.runtime.CHECKPOINT)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        p.runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        if any(x.is_meta for x in list(chat.base.parameters()) + list(chat.base.buffers())):
            raise ValueError('Unmaterialized layer')
        original_state = chat.state
        original = p.frozen_digest(chat.organism, original_state)
        device = next(chat.organism.parameters()).device
        fingerprints = IdentityFingerprints(chat.organism.config.key_count).to(device)
        state = p.fresh_controlled_state(chat.organism)
        for index, name in enumerate(NAMES):
            fingerprints.bind(KEYS[index], name)
            state, decision = p.observe_sourced(chat.organism, state, KEYS[index], (6, 1, 2, 3)[index], 'direct', 1.0)
            if not decision['current_write']:
                raise ValueError('Controlled observation rejected')
        chat.state = state
        controlled = p.frozen_digest(chat.organism, state)
        guard = RecordingGuard(fingerprints, chat.tokenizer)
        for question, expected_index in CASES:
            results = {}
            for label, identity_guard in (('baseline', None), ('guarded', guard)):
                guard.last = None
                with patch.object(p.runtime, 'current_memory_status', wraps=p.runtime.current_memory_status) as status, \
                     patch.object(chat.decoder, 'generate', wraps=chat.decoder.generate) as generate:
                    result = chat.answer(question, identity_guard=identity_guard)
                results[label] = {'result': result, 'guard_span': guard.last if identity_guard is not None else None,
                                  'status_calls': status.call_count,
                                  'generate_calls': generate.call_count}
                if result['reason'] == 'identity_guard_rejected' and (status.call_count or generate.call_count):
                    raise ValueError('Guard rejected after memory access')
            row = {'question': question, 'expected_key': None if expected_index is None else KEYS[expected_index],
                   'results': results}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
        if p.frozen_digest(chat.organism, state) != controlled:
            raise ValueError('Controlled state changed')
        if p.frozen_digest(chat.organism, original_state) != original:
            raise ValueError('Original state changed')
    if p.sha(p.runtime.CHECKPOINT) != source_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected source changed')
    summary = {}
    for label in ('baseline', 'guarded'):
        known = [r for r in rows if r['expected_key'] is not None]
        unknown = [r for r in rows if r['expected_key'] is None]
        summary[label] = {'known_correct_key': sum(r['results'][label]['result']['accepted'] and
                             r['results'][label]['result'].get('key') == r['expected_key'] for r in known),
                          'known_total': len(known),
                          'unknown_accepted': sum(r['results'][label]['result']['accepted'] for r in unknown),
                          'unknown_total': len(unknown),
                          'guard_rejections_before_memory': sum(r['results'][label]['result']['reason'] == 'identity_guard_rejected'
                              and not r['results'][label]['status_calls'] and not r['results'][label]['generate_calls'] for r in rows)}
    result = {'summary': summary, 'rows': rows, 'checkpoint_sha256': source_sha, 'base_sha256': p.WEIGHT_SHA,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Authored development questions and four manually enrolled names; no population estimate.',
                              'Exact fingerprints can be defeated by truncated name spans and do not resolve homonyms.',
                              'Pre-read guard is explicit opt-in and not a general access control mechanism.',
                              'Numeric fact writes, no natural-language write or persistent enrollment.',
                              'No real individual or default guard policy changed.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
