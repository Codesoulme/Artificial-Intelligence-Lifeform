"""Validate separate identity/fact lanes in the real SPORA memory sites."""
import json
import random
import tempfile
import time
from pathlib import Path

from scripts import deepseek_spora_long_memory_generate_parity as p
from spora.structured_text_memory import StructuredTextMemoryCodec


OUT = p.ROOT / 'runs/continuous-reader-structured-text-memory-pilot-20261113'
SEED = 99121
NAMES = ('小林', '小周', '小王', '小赵')
WORDS = ('苹果', '香蕉', '西瓜', '葡萄', '草莓', '橘子', '桃子', '芒果',
         '梨子', '柠檬', '菠萝', '樱桃', '柚子', '荔枝', '榴莲', '蓝莓',
         '火龙果', '木瓜')
IDENTITY_SLOTS = 3
FACT_SLOTS = 5
IDENTITY_SLOT_WIDTH = 15
FACT_SLOT_WIDTH = 9


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
        codec = StructuredTextMemoryCodec(
            embedding.cpu(), memory_width=model.config.memory_width,
            identity_slots=IDENTITY_SLOTS, fact_slots=FACT_SLOTS,
            identity_slot_width=IDENTITY_SLOT_WIDTH, fact_slot_width=FACT_SLOT_WIDTH,
            seed=29071, eos_token_id=tokenizer.eos_token_id).to(device)
        name_ids = [tokenizer.encode(name, add_special_tokens=False) for name in NAMES]
        word_ids = [tokenizer.encode(word, add_special_tokens=False) for word in WORDS]
        eligible = [index for index, ids in enumerate(word_ids) if len(ids) + 1 <= FACT_SLOTS]

        def empty_state():
            fast, velocity = model.initial_state(1)
            for site in fast:
                site[0] = torch.zeros_like(site[0])
            return fast, velocity

        def write(fast, velocity, key, value_index):
            packet, metadata = codec.encode(name_ids[key], word_ids[value_index])
            target = packet.reshape(1, -1).to(device)
            gate = torch.ones(1, device=device)
            mode = torch.zeros(1, dtype=torch.long, device=device)
            with torch.no_grad():
                for index, site in enumerate(model.memories):
                    basis = site.current_basis[key:key + 1].to(device)
                    fast[index], velocity[index] = site.write_temporal(
                        fast[index], velocity[index], basis, target, gate, mode, False)
            return metadata

        def read(fast, key, metadata):
            vectors = []
            with torch.no_grad():
                for index, site in enumerate(model.memories):
                    basis = site.current_basis[key:key + 1].to(device)
                    vectors.append(site.retrieve([fast[index][0]], basis)[0])
            decoded = codec.decode(torch.stack(vectors).mean(0), metadata)
            return {
                'identity_ids': decoded['identity']['token_ids'],
                'fact_ids': decoded['fact']['token_ids'],
                'identity_exact': decoded['identity']['token_ids'] == name_ids[key],
                'fact_exact': decoded['fact']['token_ids'] == word_ids[value_index],
                'margins': {'identity': decoded['identity']['margins'],
                            'fact': decoded['fact']['margins']},
            }

        single = []
        for key in range(len(NAMES)):
            for value_index in eligible:
                fast, velocity = empty_state()
                metadata = write(fast, velocity, key, value_index)
                # Keep the expected value local to avoid any hidden global
                # answer table in the codec or reader.
                packet, _ = codec.encode(name_ids[key], word_ids[value_index])
                vectors = []
                with torch.no_grad():
                    for index, site in enumerate(model.memories):
                        basis = site.current_basis[key:key + 1].to(device)
                        vectors.append(site.retrieve([fast[index][0]], basis)[0])
                decoded = codec.decode(torch.stack(vectors).mean(0), metadata)
                single.append({
                    'name': NAMES[key], 'word': WORDS[value_index],
                    'identity_ids': decoded['identity']['token_ids'],
                    'fact_ids': decoded['fact']['token_ids'],
                    'identity_exact': decoded['identity']['token_ids'] == name_ids[key],
                    'fact_exact': decoded['fact']['token_ids'] == word_ids[value_index],
                    'packet_norm': float(packet.norm()),
                })

        multi_fast, multi_velocity = empty_state()
        multi_metadata = []
        for key in range(len(NAMES)):
            multi_metadata.append(write(multi_fast, multi_velocity, key, eligible[key % len(eligible)]))
        multi = []
        for key in range(len(NAMES)):
            value_index = eligible[key % len(eligible)]
            metadata = multi_metadata[key]
            vectors = []
            with torch.no_grad():
                for index, site in enumerate(model.memories):
                    basis = site.current_basis[key:key + 1].to(device)
                    vectors.append(site.retrieve([multi_fast[index][0]], basis)[0])
            decoded = codec.decode(torch.stack(vectors).mean(0), metadata)
            multi.append({'name': NAMES[key], 'word': WORDS[value_index],
                          'identity_exact': decoded['identity']['token_ids'] == name_ids[key],
                          'fact_exact': decoded['fact']['token_ids'] == word_ids[value_index]})

        correction_key = 0
        correction_old, correction_new = eligible[0], eligible[-1]
        correction_fast, correction_velocity = empty_state()
        write(correction_fast, correction_velocity, correction_key, correction_old)
        correction_metadata = write(correction_fast, correction_velocity, correction_key, correction_new)
        vectors = []
        with torch.no_grad():
            for index, site in enumerate(model.memories):
                basis = site.current_basis[correction_key:correction_key + 1].to(device)
                vectors.append(site.retrieve([correction_fast[index][0]], basis)[0])
        corrected = codec.decode(torch.stack(vectors).mean(0), correction_metadata)
        correction = {
            'name': NAMES[correction_key], 'old': WORDS[correction_old], 'new': WORDS[correction_new],
            'identity_exact': corrected['identity']['token_ids'] == name_ids[correction_key],
            'new_fact_exact': corrected['fact']['token_ids'] == word_ids[correction_new],
            'old_fact_absent': corrected['fact']['token_ids'] != word_ids[correction_old],
        }
        if p.frozen_digest(model, chat.state) != original:
            raise ValueError('Original individual changed')
    if p.sha(p.runtime.CHECKPOINT) != source_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected source changed')

    report = {
        'status': 'structured_text_memory_pilot',
        'codec': codec.config_dict(),
        'eligible_words': [WORDS[index] for index in eligible],
        'single': single,
        'multi_entity': multi,
        'correction': correction,
        'summary': {
            'single_cases': len(single),
            'single_identity_exact': sum(row['identity_exact'] for row in single),
            'single_fact_exact': sum(row['fact_exact'] for row in single),
            'multi_entity_cases': len(multi),
            'multi_entity_exact': sum(row['identity_exact'] and row['fact_exact'] for row in multi),
            'correction_exact': correction['identity_exact'] and correction['new_fact_exact'] and correction['old_fact_absent'],
        },
        'source_checkpoint_sha256': source_sha,
        'base_sha256': p.WEIGHT_SHA,
        'seconds': time.perf_counter() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'limitations': [
            'Explicit numeric addresses and tokenizer-provided identity/fact sequences.',
            'Boundary metadata is supplied by the write path and is not yet persisted in the formal runtime state.',
            'The codec is a serialization primitive; natural-language routing and truth validation remain separate.',
            'No open-domain conversation or production adoption.',
        ],
    }
    OUT.mkdir(parents=False, exist_ok=False)
    torch.save({'format': 'spora-structured-text-memory-v1',
                'codec': {key: value.detach().cpu() for key, value in codec.state_dict().items()},
                'config': codec.config_dict(), 'source_sha256': source_sha,
                'base_sha256': p.WEIGHT_SHA}, OUT / 'candidate.pt')
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
