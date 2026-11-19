"""Exercise repeated continuous-memory updates and independently reload the result."""

import argparse
import time
from pathlib import Path

from deepseek_continuous_memory_chat import ENTITIES, FIELDS, load_chat
from deepseek_memory_field_transfer import FACTS, OUT as TRANSFER_RUN, QUESTIONS
from deepseek_output_aligned_memory import ROOT, SLOTS, encode, read, sha, write

OUT = ROOT / 'runs/deepseek-continuous-memory-stress'
ROUNDS = 12
UPDATES = {
    (26, 'drink'): ('茶', '黑糖冰豆浆', '菠萝柠檬冷泡茶', '清水'),
    (47, 'drink'): ('豆奶', '白桃乌龙气泡饮', '红茶', '青梅花茶'),
    (26, 'work'): ('画家', '城市桥梁巡检员', '厨师', '皮具修复师'),
    (47, 'work'): ('导游', '机场跑道维护员', '裁缝', '海岸测量员'),
    (26, 'place'): ('码头', '白鹭湿地观测站', '图书馆', '松桥书院'),
    (47, 'place'): ('山顶', '紫杉峡谷露营地', '书店', '新塘水厂'),
}


def verify_matrix(chat, expected):
    for (entity, field, version), value in expected.items():
        tokens = [*encode(chat.tokenizer, value), chat.tokenizer.eos_token_id]
        tokens.extend([-1] * (SLOTS - len(tokens)))
        if chat._read_tokens(chat.bank_for(entity, field, version), entity) != tokens:
            raise ValueError(f'Matrix read differs: {entity}/{field}/{version}')


def verify_chat(chat, expected):
    rows = []
    for entity, field, _, history, question in QUESTIONS:
        if field not in FIELDS:
            continue
        for version in (0, 1):
            value = expected[entity, field, version]
            result = chat.answer(question, entity, version, history)
            rows.append(dict(question=question, requested_field=field, expected=value,
                             exact_fact=result['action'] == 'read' and result['field'] == field
                             and value in result['response'], **result))
    return rows


def main(out, reload_only=False):
    working = out / 'banks.pt'
    source = TRANSFER_RUN / 'banks.pt'
    started = time.perf_counter()
    if reload_only:
        reference = read(out / 'result.json')
        if (out / 'reload.json').exists():
            raise FileExistsError(out / 'reload.json')
        if sha(source) != reference['source_sha256'] or sha(working) != reference['bank_sha256']:
            raise ValueError('Source or saved matrix changed after the write run')
        expected = {(entity, field, version): value for entity, field, version, value in reference['expected']}
        chat = load_chat(working)
        verify_matrix(chat, expected)
        live = verify_chat(chat, expected)
        if live != reference['dialogue']:
            raise ValueError('Dialogue results differ after restart')
        write(out / 'reload.json', dict(matrix_checks=len(expected), dialogue=live, bank_sha256=sha(working),
                                       elapsed_seconds=time.perf_counter() - started))
        print(f'Reload verified: {len(expected)} matrix reads, {sum(row["exact_fact"] for row in live)}/{len(live)} dialogue facts', flush=True)
        return

    out.mkdir()
    source_sha = sha(source)
    chat = load_chat(working)
    expected = {(entity, field, version): value for entity in ENTITIES for field in FIELDS
                for version, value in enumerate(FACTS[entity][field])}
    verify_matrix(chat, expected)
    checks = len(expected)
    durations, sizes = [], []
    for round_index in range(ROUNDS):
        order = ENTITIES if round_index % 2 == 0 else tuple(reversed(ENTITIES))
        for field in FIELDS:
            for entity in order:
                value = UPDATES[entity, field][round_index % len(UPDATES[entity, field])]
                before = time.perf_counter()
                result = chat.remember(entity, field, value)
                durations.append(time.perf_counter() - before)
                if result['action'] != 'stored':
                    raise ValueError(f'Expected a distinct committed correction: {round_index}/{entity}/{field}')
                expected[entity, field, 1] = value
                verify_matrix(chat, expected)
                checks += len(expected)
                sizes.append(working.stat().st_size)
        print(f'Round {round_index + 1}/{ROUNDS}, {len(durations)} corrections, {checks} matrix checks', flush=True)
    live = verify_chat(chat, expected)
    if sha(source) != source_sha:
        raise ValueError('Source matrix changed')
    write(out / 'result.json', dict(expected=[[*key, value] for key, value in expected.items()],
          writes=len(durations), matrix_checks=checks, bank_bytes=sizes, write_seconds=durations, dialogue=live,
          source_sha256=source_sha, bank_sha256=sha(working), elapsed_seconds=time.perf_counter() - started))
    print(f'Write run verified: {sum(row["exact_fact"] for row in live)}/{len(live)} dialogue facts', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--reload', action='store_true')
    args = parser.parse_args()
    main(args.out, args.reload)
