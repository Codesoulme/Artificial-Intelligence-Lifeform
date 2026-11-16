"""Audit cached generation with native memory, without a vocabulary output adapter."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.deepseek_spora_sentence_lora_pilot import make_state

OUT = ROOT / 'runs/deepseek-spora-native-memory-cache-audit-20261116.json'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch.manual_seed(68214)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    base_sha, source_sha = p.sha(p.MODEL / 'model.safetensors'), p.sha(p.runtime.CHECKPOINT)
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Base checksum mismatch')
    rows = []
    with tempfile.TemporaryDirectory(prefix='spora-native-cache-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        original = p.frozen_digest(chat.organism, chat.state)
        states = {value: make_state(chat, KEYS[0], value) for value in (1, 6)}
        digests = {value: p.frozen_digest(chat.organism, state) for value, state in states.items()}
        cases = [(f'{NAMES[0]}现在最喜欢喝什么饮料？', 1),
                 (f'{NAMES[0]}现在最喜欢喝什么饮料？', 6),
                 ('小明有三本书，又买了两本，送给朋友一本。他现在有几本书？', 1)]
        for question, value in cases:
            inputs = closed_think_inputs(chat.tokenizer, question, 'cuda')
            args, native = p.runtime._prepare_memory_read(chat.organism, states[value], KEYS[0], 'current', None)
            if int(native.argmax(-1)[0]) != value:
                raise ValueError('Controlled native read failed')
            variants = {}
            for cache in (False, True):
                options = dict(args)
                options['use_cache'] = cache
                begin = time.perf_counter()
                torch.cuda.synchronize()
                with torch.inference_mode():
                    output = chat.decoder.generate(**inputs, **options, do_sample=False,
                                                   max_new_tokens=24,
                                                   pad_token_id=chat.tokenizer.pad_token_id,
                                                   eos_token_id=chat.tokenizer.eos_token_id)
                torch.cuda.synchronize()
                tail = output[0, inputs['input_ids'].shape[1]:].tolist()
                variants['cached' if cache else 'uncached'] = {
                    'token_ids': tail, 'text': chat.tokenizer.decode(tail, skip_special_tokens=True).strip(),
                    'ended': bool(tail and tail[-1] == chat.tokenizer.eos_token_id),
                    'seconds': time.perf_counter() - begin}
            row = {'question': question, 'controlled_value': p.WORDS[value], 'variants': variants,
                   'tokens_match': variants['cached']['token_ids'] == variants['uncached']['token_ids']}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        if (p.frozen_digest(chat.organism, chat.state) != original or
                any(p.frozen_digest(chat.organism, states[v]) != digests[v] for v in states)):
            raise ValueError('Memory state changed during inference')
    if p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(p.runtime.CHECKPOINT) != source_sha:
        raise ValueError('Protected input changed')
    report = {'rows': rows, 'base_sha256': base_sha, 'checkpoint_sha256': source_sha,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Two controlled numeric values for one known entity; addresses supplied by caller.',
                              'Ordinary question forces an unrelated read; not a natural routing test.',
                              'No vocabulary output adapter; closed reasoning prefix and greedy 24-token budget.',
                              'Each generation takes a fresh immutable memory snapshot and fresh cache.',
                              'No natural writing, long conversation or open-world generalization.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
