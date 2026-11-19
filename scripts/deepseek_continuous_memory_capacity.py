"""Fill the native address basis and verify corrections across a fresh reload."""

import argparse
import time
from itertools import product
from pathlib import Path

from deepseek_continuous_memory_chat import ADDRESSES, EARLY_TIMING_THRESHOLD, FIELDS, load_chat
from deepseek_continuous_memory_stress import verify_matrix
from deepseek_memory_field_transfer import FACTS, OUT as TRANSFER_RUN
from deepseek_output_aligned_memory import ROOT, SLOTS, encode, read, sha, torch, write

OUT = ROOT / 'runs/deepseek-continuous-memory-capacity'
SURNAMES = '周宋秦唐许韩施魏梁顾陶姜孟白苏陆杜程吕曹熊金'
GIVEN_NAMES = '岚文禾远仪清景桐宁越'
FLAVORS = ('青梅', '蜜桃', '雪梨', '白桃', '金桔', '陈皮', '桂花', '紫苏', '菠萝', '杏仁', '柠檬', '枇杷')
DRINKS = ('花茶', '冰茶', '奶茶', '果茶', '气泡饮', '冷泡茶', '乌龙茶', '苏打水')


def probe_dialogue(chat, expected, names):
    rows = []
    for name in names:
        entity = chat.addresses[name]
        value = expected[entity, 'drink', 1]
        query = f'{name}平常最爱喝什么？'
        result = chat.answer(query, timing_threshold=EARLY_TIMING_THRESHOLD)
        exact = (result['entity'] == entity and result['field'] == 'drink' and result['action'] == 'read'
                 and result['response'].rstrip('。.!！ ').endswith(value))
        rows.append(dict(name=name, expected=value, exact=exact, **result))
    return rows


def measure_reads(chat, keys, repeats=4):
    durations = []
    for _ in range(repeats):
        for entity in keys:
            started = time.perf_counter()
            chat._read_tokens(chat.banks['drink', 1], entity)
            durations.append(time.perf_counter() - started)
    return durations


def main(out, reload_only=False):
    started = time.perf_counter()
    path = out / 'banks.pt'
    seed = TRANSFER_RUN / 'banks.pt'
    if reload_only:
        reference = read(out / 'result.json')
        if (out / 'reload.json').exists():
            raise FileExistsError(out / 'reload.json')
        if sha(path) != reference['bank_sha256'] or sha(seed) != reference['seed_sha256']:
            raise ValueError('Source or working bank changed')
        chat = load_chat(path)
        expected = {(entity, field, version): value for entity, field, version, value in reference['expected']}
        if chat.addresses != reference['addresses'] or set(chat.filled) != set(expected):
            raise ValueError('Registry or occupancy changed on reload')
        verify_matrix(chat, expected)
        dialogue = probe_dialogue(chat, expected, reference['probe_names'])
        if dialogue != reference['dialogue']:
            raise ValueError('Dialogue changed on reload')
        write(out / 'reload.json', dict(matrix_reads=len(expected), dialogue=dialogue, bank_sha256=sha(path),
                                       elapsed_seconds=time.perf_counter() - started))
        print(f'Reload verified all {len(expected)} matrix records', flush=True)
        return

    out.mkdir()
    seed_sha = sha(seed)
    chat = load_chat(path)
    capacity = chat.site.current_basis.shape[0]
    names = [surname + given for surname, given in product(SURNAMES, GIVEN_NAMES)]
    existing_values = {value for group in FACTS.values() for pair in group.values() for value in pair}
    values = [flavor + drink for flavor, drink in product(FLAVORS, DRINKS)
              if flavor + drink not in existing_values and 1 <= len(encode(chat.tokenizer, flavor + drink)) < SLOTS]
    added = capacity - len(chat.addresses)
    if len(names) <= added or len(values) < added + 12:
        raise ValueError(f'Insufficient unique fixtures for the {capacity}-address basis')
    expected = {(entity, field, version): value for entity in ADDRESSES.values() for field in FIELDS
                for version, value in enumerate(FACTS[entity][field])}
    verify_matrix(chat, expected)
    diagnostic_keys = tuple(ADDRESSES.values())
    read_empty = measure_reads(chat, diagnostic_keys)
    basis = chat.site.current_basis.float()
    gram = basis @ basis.T
    max_cross = float((gram - torch.diag(gram.diagonal())).abs().max())
    print(f'Basis capacity {capacity}, max absolute cross-address dot product {max_cross:.3g}', flush=True)

    writes = []
    fill_names = names[:added]
    for index, name in enumerate(fill_names):
        result = chat.register(name)
        if result['action'] != 'registered':
            raise ValueError(f'Registration failed: {name}: {result}')
        entity = result['entity']
        before = time.perf_counter()
        result = chat.remember(entity, 'drink', values[index])
        if result['action'] != 'stored':
            raise ValueError(f'Initial write failed: {name}: {result}')
        writes.append(dict(operation='fill', entity=entity, seconds=time.perf_counter() - before,
                           bank_bytes=path.stat().st_size, subjects=len(chat.addresses)))
        expected[entity, 'drink', 1] = values[index]
        if (index + 1) % 8 == 0 or index + 1 == added:
            print(f'Filled {len(chat.addresses)}/{capacity} addresses', flush=True)
    before = sha(path)
    if chat.register(names[added]).get('reason') != 'no_free_address' or sha(path) != before:
        raise ValueError('Capacity rejection changed the checkpoint')

    verify_matrix(chat, expected)
    correction_names = [fill_names[index * (added - 1) // 11] for index in range(12)]
    for index, name in enumerate(correction_names):
        entity = chat.addresses[name]
        before = time.perf_counter()
        result = chat.remember(entity, 'drink', values[added + index])
        if result['action'] != 'stored':
            raise ValueError(f'Correction failed: {name}: {result}')
        writes.append(dict(operation='correction', entity=entity, seconds=time.perf_counter() - before,
                           bank_bytes=path.stat().st_size, subjects=len(chat.addresses)))
        expected[entity, 'drink', 1] = values[added + index]
    verify_matrix(chat, expected)
    print(f'Verified {len(expected)} records after 12 corrections at capacity', flush=True)
    read_full = measure_reads(chat, diagnostic_keys)
    probe_names = [*ADDRESSES, *correction_names[::2]]
    dialogue = probe_dialogue(chat, expected, probe_names)
    if sha(seed) != seed_sha:
        raise ValueError('Seed bank changed')
    write(out / 'result.json', dict(capacity=capacity, addresses=chat.addresses, max_cross=max_cross, writes=writes,
          expected=[[*key, value] for key, value in expected.items()], probe_names=probe_names, dialogue=dialogue,
          read_seconds={'before': read_empty, 'full': read_full}, seed_sha256=seed_sha, bank_sha256=sha(path),
          elapsed_seconds=time.perf_counter() - started))
    print(f'Full-capacity dialogue: {sum(row["exact"] for row in dialogue)}/{len(dialogue)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--reload', action='store_true')
    args = parser.parse_args()
    main(args.out, args.reload)
