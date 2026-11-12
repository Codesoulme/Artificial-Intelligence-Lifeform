"""Integrate the reusable text codec with isolated SPORA fast matrices."""
import json
import tempfile
import time
from pathlib import Path

import torch

from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_spora_continuous_text_capacity import sampled_sequences
from spora.continuous_text_memory import ContinuousTextCodec

OUTPUT = p.ROOT / 'runs/deepseek-spora-continuous-text-codec-integration-retention-20261112.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    p.torch.set_num_threads(2)
    p.torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    examples, pool_size = sampled_sequences(tokenizer)
    correction = examples[16]
    examples = examples[:16]
    with tempfile.TemporaryDirectory() as directory:
        p.runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        if any(x.is_meta for x in list(chat.base.parameters()) + list(chat.base.buffers())):
            raise ValueError('Unmaterialized layer')
        organism = chat.organism.eval()
        original = p.frozen_digest(organism, chat.state)
        device = next(organism.parameters()).device
        codec = ContinuousTextCodec(chat.base.model.embed_tokens.weight.detach(),
                                    memory_width=organism.config.memory_width, slot_count=4,
                                    seed=29071, eos_token_id=tokenizer.eos_token_id)
        code_width = codec.slot_count * codec.slot_width
        fast, velocity = organism.initial_state(1)
        for index in range(len(fast)):
            fast[index][0] = torch.zeros_like(fast[index][0])
        mode = torch.zeros(1, dtype=torch.long, device=device)
        gate = torch.ones(1, device=device)

        def write(key, item):
            code = codec.encode(torch.tensor(item['token_ids'], dtype=torch.long, device=device))
            target = torch.zeros((1, organism.config.memory_width), device=device)
            target[0, :code_width] = code
            for index, site in enumerate(organism.memories):
                fast[index], velocity[index] = site.write_temporal(
                    fast[index], velocity[index], site.current_basis[key:key + 1].to(device),
                    target, gate, mode, False)

        def read(key):
            results = []
            for index, site in enumerate(organism.memories):
                address = site.current_basis[key:key + 1].to(device)
                with torch.no_grad():
                    vector = site.retrieve([fast[index][0]], address)[0, :code_width]
                results.append(codec.decode(vector))
            return results

        for key, item in enumerate(examples):
            write(key, item)
        rows = []
        for key, item in enumerate(examples):
            reads = read(key)
            expected = item['token_ids'] + [tokenizer.eos_token_id]
            rows.append({'key': key, 'expected': item['text'],
                         'sites': [{'token_ids': r['token_ids'], 'margins': r['margins'],
                                    'exact': r['token_ids'] == expected} for r in reads]})
        write(0, correction)
        corrected = read(0)
        rows.append({'key': 0, 'expected': correction['text'], 'group': 'correction',
                     'sites': [{'token_ids': r['token_ids'], 'margins': r['margins'],
                                'exact': r['token_ids'] == correction['token_ids'] + [tokenizer.eos_token_id]}
                               for r in corrected]})
        retained = []
        for key, item in enumerate(examples[1:], start=1):
            reads = read(key)
            expected = item['token_ids'] + [tokenizer.eos_token_id]
            retained.append(all(r['token_ids'] == expected for r in reads))
        if p.frozen_digest(organism, chat.state) != original:
            raise ValueError('Original state changed')
    if p.sha(p.runtime.CHECKPOINT) != checkpoint_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected input changed')
    result = {'rows': rows, 'summary': {'initial_exact_both': sum(all(s['exact'] for s in r['sites']) for r in rows[:16]),
                                        'correction_exact_both': all(s['exact'] for s in rows[-1]['sites']),
                                        'other_retained_both': sum(retained), 'other_total': len(retained)},
              'codec': codec.config_dict(), 'token_pool': pool_size,
              'checkpoint_sha256': checkpoint_sha, 'base_sha256': p.WEIGHT_SHA,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Sixteen random CJK sequences and one correction in a temporary matrix.',
                              'Numeric orthogonal addresses and direct writes bypass routing and persistence.',
                              'Codec nearest-neighbor output is not language-model generation or unknown detection.',
                              'No default memory, checkpoint, or real individual changed.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
