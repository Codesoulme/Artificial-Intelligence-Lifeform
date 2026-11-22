"""Replay saved temporal memories through the Chinese date-query entry point."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoTokenizer

from scripts.audit_qwen_compatible_natural_memory_20261121 import KEYS, NAMES
from scripts.audit_spora_bitemporal_memory import CASES, stamp
from scripts.audit_spora_native_query_key import write
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_calendar_history_chat import load_calendar_state, read_calendar
from spora.growing import load, save
from spora.identity_fingerprints import IdentityFingerprints
from spora.integrated_checkpoint import load_integrated_runtime
from spora.language_training import frozen_digest


def paraphrase(name, effective, known_by):
    year, month, day = map(int, stamp(effective).split('-'))
    question = f'帮我查{name}在{year}年{month}月{day}日当天登记的饮料是哪种'
    if known_by is not None:
        question += f'，依据截止到{stamp(known_by)}已经掌握的资料'
    return question + '？'


def judge(result, route, expected_event, expected_value):
    if result['routing'] != route:
        return False
    selected, generated = result['selection'], result['generated']
    if expected_event is None:
        return selected.get('reason') == 'unresolved_factual_observation' and generated is None
    return (selected.get('event_time') == expected_event and generated is not None
            and generated.get('matrix_value') == expected_value
            and (generated.get('answer') or '').rstrip('。！ ') == expected_value)


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    started = time.perf_counter()
    prior_path = args.prior / 'result.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    source_sha = sha(args.checkpoint)
    if prior['source_sha256'] != source_sha or not all(row['correct_answer'] for row in prior['rows']):
        raise ValueError('The calendar audit needs the verified temporal memory source')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(args.checkpoint, load_full_streamed(), MODEL)
    decoder = runtime.decoder.eval().requires_grad_(False)
    pretrained_individual = frozen_digest(runtime.organism, runtime.state)
    cases = {label: (effective, known_by) for label, effective, known_by, _ in CASES}
    positives, negatives, inputs = [], [], {str(prior_path): sha(prior_path), str(args.checkpoint): source_sha}
    args.output.mkdir(parents=True)
    for index in (0, 3):
        name, key = NAMES[index], KEYS[index]
        for world in (0, 2):
            tag = f'person-{index}-world-{world}'
            seed = args.prior / f'{tag}-present.pt'
            inputs[str(seed)] = sha(seed)
            organism, state, _ = load(seed, 'cpu')
            if frozen_digest(organism, state) != prior['state_sha256'][tag]:
                raise ValueError('Unexpected temporal memory fixture')
            organism.identity_fingerprints = IdentityFingerprints(organism.config.key_count)
            organism.identity_fingerprints.bind(key, name)
            calendar = {'format': 'spora-observation-calendar-v1', 'days': [stamp(i) for i in range(state.cursor)]}
            saved = args.output / f'{tag}.pt'
            save(saved, organism, state, {'observation_calendar': calendar, 'decoder_checkpoint_sha256': source_sha})
            organism, state, calendar = load_calendar_state(saved, source_sha)
            before = frozen_digest(organism, state)
            for row in prior['rows']:
                if (row['person'], row['world']) != (name, world):
                    continue
                effective, known_by = cases[row['case']]
                route = {'reason': 'routed', 'key': key, 'as_of': effective, 'known_by': known_by}
                for text in (row['question'], paraphrase(name, effective, known_by)):
                    result = read_calendar(decoder, tokenizer, organism, state, calendar, text)
                    passed = judge(result, route, row['expected_event'], row['expected_value'])
                    positives.append({'question': text, 'world': world, 'case': row['case'], 'passed': passed,
                                      'expected_value': row['expected_value'], **result})
            for text, reason in (
                (f'{name}在2026年2月30日的饮品是什么？', 'invalid_calendar_date'),
                (f'{name}在2026-01-06的饮品是什么？', 'date_outside_saved_calendar'),
                (f'{name}遥在2026-01-02的饮品是什么？', 'date_subject_not_enrolled'),
                (f'请把「{name}在2026-01-02的饮品是什么」翻成英文。', 'unsupported_date_question'),
                (f'{name}在2026-01-02和2026-01-03的饮品分别是什么？', 'unsupported_date_question'),
            ):
                result = read_calendar(decoder, tokenizer, organism, state, calendar, text)
                passed = (result['routing']['reason'] == reason and result['selection'] is None
                          and result['generated'] is None)
                negatives.append({'question': text, 'world': world, 'passed': passed, **result})
            if frozen_digest(organism, state) != before:
                raise ValueError('Calendar read changed its individual')
    if frozen_digest(runtime.organism, runtime.state) != pretrained_individual or any(
        sha(Path(path)) != digest for path, digest in inputs.items()
    ):
        raise ValueError('Calendar audit changed protected sources')
    summary = {'dated_questions': len(positives), 'dated_pass': sum(row['passed'] for row in positives),
               'unsupported_questions': len(negatives), 'unsupported_pass': sum(row['passed'] for row in negatives)}
    write(args.output / 'result.json', {'summary': summary, 'dated': positives, 'unsupported': negatives,
                                        'sources': inputs, 'seconds': time.perf_counter() - started})
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'prior', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
