"""Read a saved SPORA individual's dated memories from Chinese questions."""

import argparse
from datetime import date
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoTokenizer

from scripts.audit_qwen_compatible_natural_memory_20261121 import WORDS, answer
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_event_scan_selector import select_event
from spora.calendar_history import CalendarFocus, ObservationCalendar, resolve_calendar_followup, resolve_calendar_question
from spora.growing import load
from spora.integrated_checkpoint import load_integrated_runtime
from spora.integrated_durable import create_integrated_durable, load_integrated_durable
from spora.named_address import address_basis_sha256


_MESSAGES = {
    'unsupported_date_question': '请用完整日期和一个已登记姓名询问饮品记录。',
    'invalid_calendar_date': '这个日期无效。',
    'date_outside_saved_calendar': '这个日期超出了已保存记录的范围。',
    'date_subject_not_enrolled': '这个姓名尚未登记。',
    'date_followup_requires_subject': '请先告诉我你想查谁、哪一天的记录。',
    'date_followup_requires_backfill': '请指定要以哪一天收到的资料为准。',
}


def load_calendar_state(path, source_sha):
    organism, state, metadata = load(path, 'cpu')
    if metadata.get('decoder_checkpoint_sha256') != source_sha:
        raise ValueError('Calendar individual was saved for a different decoder checkpoint')
    calendar = ObservationCalendar.from_payload(metadata.get('observation_calendar'), state.cursor)
    if not hasattr(organism, 'identity_fingerprints'):
        raise ValueError('Calendar history requires enrolled names in the saved individual')
    organism.identity_fingerprints.validate(organism)
    return organism, state, calendar


def read_calendar(decoder, tokenizer, organism, state, calendar, question, *, focus=None):
    route = resolve_calendar_question(question, organism.identity_fingerprints, calendar)
    followup = None
    if route['reason'] == 'unsupported_date_question':
        followup = resolve_calendar_followup(question, calendar, focus)
        if followup is not None:
            route = followup
    result = {'routing': route, 'selection': None, 'generated': None}
    if route['reason'] != 'routed':
        return result
    selected = select_event(organism, state, route['key'], 'effective_latest',
                            as_of=route['as_of'], known_by=route['known_by'])
    result['selection'] = selected
    if selected['status'] == 'experimental_candidate':
        generated = answer(decoder, tokenizer, organism, question, state, route['key'], 2,
                           event=selected['event_time'], use_cache=True, output_gain=1.5,
                           allowed_replies=WORDS if followup is not None else None)
        if (followup is not None and generated['reason'] == 'generated'
                and generated['answer'] != generated['matrix_value']):
            generated = dict(generated, reason='inconsistent_memory_reply', answer=None)
        result['generated'] = generated
    return result


def calendar_turn(decoder, tokenizer, organism, state, calendar, question, focus):
    if question == '/reset':
        return {'routing': {'reason': 'date_focus_reset'}, 'selection': None, 'generated': None}, None
    result = read_calendar(decoder, tokenizer, organism, state, calendar, question, focus=focus)
    return result, CalendarFocus.from_read(result['routing'], result['selection'], focus)


def console(decoder, tokenizer, organism, state, calendar, questions):
    if calendar is None or not hasattr(organism, 'identity_fingerprints'):
        raise ValueError('Calendar history requires dates and enrolled names in the saved individual')
    focus = None
    if questions:
        for question in questions:
            result, focus = calendar_turn(decoder, tokenizer, organism, state, calendar, question, focus)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        return
    print('输入带日期的饮品记忆问题。用 /reset 清除追问对象，用 /quit 退出。')
    while True:
        try:
            question = input('你：').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question == '/quit':
            break
        result, focus = calendar_turn(decoder, tokenizer, organism, state, calendar, question, focus)
        generated = result['generated'] or {}
        reason = result['routing']['reason']
        response = (generated.get('answer') or ('已清除追问对象。' if reason == 'date_focus_reset'
                                               else _MESSAGES.get(reason, '这条记录目前无法确认。')))
        print(f'孢子：{response}', flush=True)


def main(args):
    for stream in (sys.stdin, sys.stdout):
        if not stream.isatty():
            stream.reconfigure(encoding='utf-8')
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    if args.database or args.create_database:
        base = load_full_streamed().eval().requires_grad_(False)
        if args.create_database:
            runtime = create_integrated_durable(args.create_database, args.checkpoint, base, MODEL,
                                                calendar_individual=args.individual)
        else:
            runtime = load_integrated_durable(args.database, args.checkpoint, base, MODEL)
        with runtime:
            if args.observe:
                name, value = args.observe
                if runtime.observation_calendar is None or not hasattr(runtime.organism, 'identity_fingerprints'):
                    raise ValueError('Explicit observations require saved dates and enrolled names')
                key = runtime.organism.identity_fingerprints.lookup(name)
                if key is None or value not in WORDS:
                    raise ValueError('Use an enrolled name and a supported beverage value')
                decision, stats = runtime.observe_sourced(key, WORDS.index(value), 'direct', 1.,
                                                         effective_on=args.effective_on)
                print(json.dumps({'observation': decision, 'revision': stats['revision'],
                                  'observed_on': runtime.observation_calendar.days[-1].isoformat()},
                                 ensure_ascii=False))
                if not args.question:
                    return
            console(runtime.decoder, tokenizer, runtime.organism, runtime.state, runtime.observation_calendar,
                    args.question)
        return
    organism, state, calendar = load_calendar_state(args.individual, sha(args.checkpoint))
    runtime = load_integrated_runtime(args.checkpoint, load_full_streamed(), MODEL)
    if (organism.config != runtime.organism.config
            or address_basis_sha256(organism) != address_basis_sha256(runtime.organism)):
        raise ValueError('The individual is incompatible with the decoder checkpoint')
    console(runtime.decoder.eval().requires_grad_(False), tokenizer, organism, state, calendar, args.question)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--individual', type=Path, help='Read a portable calendar individual')
    source.add_argument('--database', type=Path, help='Reopen a transactional calendar individual')
    parser.add_argument('--create-database', type=Path, help='Start a new transactional copy of --individual')
    parser.add_argument('--observe', nargs=2, metavar=('NAME', 'BEVERAGE'), help='Record an explicit direct observation')
    parser.add_argument('--effective-on', type=date.fromisoformat, help='Earlier recorded effective date, YYYY-MM-DD')
    parser.add_argument('--question', action='append', help='Read a Chinese question and print the result')
    args = parser.parse_args()
    if args.create_database and args.individual is None:
        parser.error('--create-database requires --individual')
    if args.observe and not (args.database or args.create_database):
        parser.error('--observe requires --database or --create-database')
    if args.effective_on and not args.observe:
        parser.error('--effective-on requires --observe')
    main(args)
