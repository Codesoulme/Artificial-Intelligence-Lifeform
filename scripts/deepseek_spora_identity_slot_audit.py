"""Audit whether continuous memory preserves identity token sequences.

This is a research-only diagnostic. It bypasses natural-language routing and
uses explicit numeric addresses, but it separates tokenizer loss from memory
loss before any learned text decoder is evaluated.
"""
import json
import random
import tempfile
import time
from pathlib import Path

from scripts import deepseek_spora_long_memory_generate_parity as p
from spora.continuous_text_memory import ContinuousTextCodec


OUT = p.ROOT / 'runs/continuous-reader-identity-slot-audit-20261113'
SEED = 99118
NAMES = ('小林', '小周', '小王', '小赵')
WORDS = ('苹果', '香蕉', '西瓜', '葡萄', '草莓', '橘子', '桃子', '芒果',
         '梨子', '柠檬', '菠萝', '樱桃', '柚子', '荔枝', '榴莲', '蓝莓',
         '火龙果', '木瓜')
SLOT_COUNT = 5


def token_info(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return {'text': text, 'ids': ids, 'decoded': tokenizer.decode(ids, skip_special_tokens=True)}


def write_and_read(model, codec, tokenizer, device, address_key, ids):
    fast, velocity = model.initial_state(1)
    for site in fast:
        site[0] = p.torch.zeros_like(site[0])
    encoded = codec.encode(p.torch.tensor(ids, dtype=p.torch.long, device=device))
    target = p.torch.zeros((1, model.config.memory_width), device=device)
    target[0, :encoded.numel()] = encoded
    gate = p.torch.ones(1, device=device)
    mode = p.torch.zeros(1, dtype=p.torch.long, device=device)
    with p.torch.no_grad():
        for index, site in enumerate(model.memories):
            basis = site.current_basis[address_key:address_key + 1].to(device)
            fast[index], velocity[index] = site.write_temporal(
                fast[index], velocity[index], basis, target, gate, mode, False)
    reads = []
    for index, site in enumerate(model.memories):
        with p.torch.no_grad():
            basis = site.current_basis[address_key:address_key + 1].to(device)
            vector = site.retrieve([fast[index][0]], basis)[0]
            reads.append(codec.decode(vector))
    stacked = p.torch.tensor([r['slot_norms'] for r in reads])
    average_vector = p.torch.stack([
        site.retrieve([fast[index][0]], site.current_basis[address_key:address_key + 1].to(device))[0]
        for index, site in enumerate(model.memories)
    ]).mean(0)
    average = codec.decode(average_vector)
    expected = ids + [tokenizer.eos_token_id]
    return {
        'expected_ids': expected,
        'sites': [{
            'token_ids': row['token_ids'],
            'decoded': tokenizer.decode(row['token_ids'], skip_special_tokens=True),
            'exact': row['token_ids'] == expected,
            'margins': row['margins'],
            'slot_norms': row['slot_norms'],
        } for row in reads],
        'average': {
            'token_ids': average['token_ids'],
            'decoded': tokenizer.decode(average['token_ids'], skip_special_tokens=True),
            'exact': average['token_ids'] == expected,
            'margins': average['margins'],
            'slot_norms': average['slot_norms'],
        },
        'all_sites_exact': all(row['token_ids'] == expected for row in reads),
        'average_exact': average['token_ids'] == expected,
        'slot_norm_summary': stacked.cpu().tolist(),
    }


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch = p.torch
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    source_sha = p.sha(p.runtime.CHECKPOINT)
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    with tempfile.TemporaryDirectory() as temp:
        p.runtime.WORKING_CHECKPOINT = Path(temp) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        model = chat.organism.eval()
        device = next(model.parameters()).device
        original = p.frozen_digest(model, chat.state)
        embedding = chat.base.model.embed_tokens.weight.detach()
        codec = ContinuousTextCodec(embedding.cpu(), memory_width=model.config.memory_width,
                                    slot_count=SLOT_COUNT, seed=29071,
                                    eos_token_id=tokenizer.eos_token_id).to(device)

        names = [token_info(tokenizer, name) for name in NAMES]
        words = [token_info(tokenizer, word) for word in WORDS]
        cases = []
        for key, name in enumerate(names):
            variants = [('identity_only', name['ids'])]
            for word in words[:4]:
                variants.append(('identity_plus_fact', name['ids'] + word['ids']))
            for variant, ids in variants:
                if len(ids) + 1 > SLOT_COUNT:
                    cases.append({'key': key, 'name': name['text'], 'variant': variant,
                                  'input_ids': ids, 'status': 'capacity_exceeded'})
                    continue
                read = write_and_read(model, codec, tokenizer, device, key, ids)
                cases.append({'key': key, 'name': name['text'], 'variant': variant,
                              'input_ids': ids, 'decoded_input': tokenizer.decode(ids),
                              **read})

        # Reproduce the previous script's lossy identity representation.
        lossy = []
        for key, name in enumerate(names):
            ids = [name['ids'][-1]]
            read = write_and_read(model, codec, tokenizer, device, key, ids)
            lossy.append({'key': key, 'name': name['text'], 'input_ids': ids, **read})

        if p.frozen_digest(model, chat.state) != original:
            raise ValueError('Original individual changed')
    if p.sha(p.runtime.CHECKPOINT) != source_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected source changed')

    usable = [row for row in cases if row.get('status') != 'capacity_exceeded']
    report = {
        'status': 'identity_slot_audit',
        'tokenizer': {'names': names, 'words_sample': words[:4], 'eos_id': tokenizer.eos_token_id},
        'config': codec.config_dict(),
        'cases': cases,
        'lossy_last_token_cases': lossy,
        'summary': {
            'usable_cases': len(usable),
            'all_site_exact': sum(row['all_sites_exact'] for row in usable),
            'average_exact': sum(row['average_exact'] for row in usable),
            'identity_only_cases': sum(row['variant'] == 'identity_only' for row in usable),
            'identity_only_all_site_exact': sum(row['variant'] == 'identity_only' and row['all_sites_exact'] for row in usable),
            'identity_plus_fact_cases': sum(row['variant'] == 'identity_plus_fact' for row in usable),
            'identity_plus_fact_all_site_exact': sum(row['variant'] == 'identity_plus_fact' and row['all_sites_exact'] for row in usable),
            'lossy_last_token_all_site_exact': sum(row['all_sites_exact'] for row in lossy),
            'lossy_last_token_average_exact': sum(row['average_exact'] for row in lossy),
        },
        'source_checkpoint_sha256': source_sha,
        'base_sha256': p.WEIGHT_SHA,
        'seconds': time.perf_counter() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'limitations': [
            'Explicit numeric addresses and manually constructed token sequences.',
            'No natural-language routing, learned text generation, or production persistence.',
            'The audit tests the fixed codec and memory read path, not open-domain identity understanding.',
        ],
    }
    OUT.mkdir(parents=False, exist_ok=False)
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
