"""Read a saved SPORA individual's dated memories from Chinese questions."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoTokenizer

from scripts.audit_qwen_compatible_natural_memory_20261121 import answer
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_event_scan_selector import select_event
from spora.calendar_history import ObservationCalendar, resolve_calendar_question
from spora.growing import load
from spora.integrated_checkpoint import load_integrated_runtime
from spora.named_address import address_basis_sha256


_MESSAGES = {
    'unsupported_date_question': '请用完整日期和一个已登记姓名询问饮品记录。',
    'invalid_calendar_date': '这个日期无效。',
    'date_outside_saved_calendar': '这个日期超出了已保存记录的范围。',
    'date_subject_not_enrolled': '这个姓名尚未登记。',
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


def read_calendar(decoder, tokenizer, organism, state, calendar, question):
    route = resolve_calendar_question(question, organism.identity_fingerprints, calendar)
    result = {'routing': route, 'selection': None, 'generated': None}
    if route['reason'] != 'routed':
        return result
    selected = select_event(organism, state, route['key'], 'effective_latest',
                            as_of=route['as_of'], known_by=route['known_by'])
    result['selection'] = selected
    if selected['status'] == 'experimental_candidate':
        result['generated'] = answer(decoder, tokenizer, organism, question, state, route['key'], 2,
                                     event=selected['event_time'], use_cache=True, output_gain=1.5)
    return result


def main(args):
    for stream in (sys.stdin, sys.stdout):
        if not stream.isatty():
            stream.reconfigure(encoding='utf-8')
    torch.set_num_threads(2)
    organism, state, calendar = load_calendar_state(args.individual, sha(args.checkpoint))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(args.checkpoint, load_full_streamed(), MODEL)
    decoder = runtime.decoder.eval().requires_grad_(False)
    if (organism.config != runtime.organism.config
            or address_basis_sha256(organism) != address_basis_sha256(runtime.organism)):
        raise ValueError('The individual is incompatible with the decoder checkpoint')
    if args.question:
        for question in args.question:
            print(json.dumps(read_calendar(decoder, tokenizer, organism, state, calendar, question),
                             ensure_ascii=False), flush=True)
        return
    print('输入带日期的饮品记忆问题，或输入 /quit 退出。')
    while True:
        try:
            question = input('你：').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question == '/quit':
            break
        result = read_calendar(decoder, tokenizer, organism, state, calendar, question)
        generated = result['generated'] or {}
        response = generated.get('answer') or _MESSAGES.get(result['routing']['reason'], '这条记录目前无法确认。')
        print(f'孢子：{response}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--individual', type=Path, required=True)
    parser.add_argument('--question', action='append', help='Read a Chinese question and print the result')
    main(parser.parse_args())
