"""Run mixed ordinary dialogue and temporal memory on one SPORA individual."""

import argparse
from collections import deque
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoTokenizer

from scripts.audit_spora_native_query_key import write
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_calendar_history_chat import _CHAT_TURNS, ordinary_reply, session_turn
from spora.integrated_durable import load_integrated_durable
from spora.language_training import frozen_digest


SCENES = {
    'outing': (
        ('周五我想去花园，人多就去图书馆。请用一句话复述。', 'ordinary_chat', None),
        ('小林在2026-01-02的饮品是什么？', 'routed', '果汁'),
        ('那按补录前的资料呢？', 'routed', '咖啡'),
        ('回到刚才的周五计划。如果花园人多，我准备去哪？', 'ordinary_chat', None),
        ('如果图书馆也关门了，给一个不费钱的备选。', 'ordinary_chat', None),
    ),
    'explanation': (
        ('你好，能帮我理一理一个生活小问题吗？', 'ordinary_chat', None),
        ('为什么水杯外壁会有小水珠？用简单中文解释。', 'ordinary_chat', None),
        ('小林在2027-01-20的饮品是什么？', 'routed', '豆浆'),
        ('那2027-03-25呢？', 'routed', '茶'),
        ('刚才那个水杯问题，试试用一句话解释给小朋友。', 'ordinary_chat', None),
    ),
    'revision': (
        ('用苹果和火车写一句话。', 'ordinary_chat', None),
        ('把火车改成公交车，苹果保留。', 'ordinary_chat', None),
        ('查小林前后两天的饮品记录，日期我还没说。', 'unsupported_date_question', None),
        ('帮我查小林在2026年1月2日当天登记的饮料是哪种？', 'routed', '果汁'),
        ('接着改前面那句话，让它读起来更幽默。', 'ordinary_chat', None),
    ),
    'boundaries': (
        ('那按补录前的资料呢？', 'date_followup_requires_subject', None),
        ('室友说昨晚回来得晚。能确定他为什么回来晚吗？', 'ordinary_chat', None),
        ('小林遥在2026-01-02的饮品是什么？', 'date_subject_not_enrolled', None),
        ('小林在2026-01-02和2026-01-03的饮品分别是什么？', 'unsupported_date_question', None),
        ('那2026-01-04呢？', 'date_followup_requires_subject', None),
        ('给出一个礼貌问室友到家时间的句子。', 'ordinary_chat', None),
    ),
}


def compare_original(base, tokenizer, position_limit, reference):
    """Hold the exact prior ordinary turns fixed for the raw-model comparison."""
    decoder = SimpleNamespace(base=base, position_table_max_position=position_limit, generate=base.generate)
    results = {}
    for label, rows in reference['scenes'].items():
        history = deque(maxlen=_CHAT_TURNS)
        ordinary = []
        for row in rows:
            if row['routing']['reason'] != 'ordinary_chat':
                continue
            question, modified = row['question'], row['generated']
            raw = ordinary_reply(decoder, tokenizer, history, question)
            if modified.get('prompt_tokens') != raw.get('prompt_tokens'):
                raise ValueError('Original and SPORA models received different context budgets')
            ordinary.append({'question': question, 'original': raw, 'spora': modified})
            if modified['answer']:
                history.append((question, modified['answer']))
        results[label] = ordinary
    return results


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    protected = {str(path): sha(path) for path in (args.database, args.checkpoint)}
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed().eval().requires_grad_(False)
    raw = None
    if args.compare_original is not None:
        protected[str(args.compare_original)] = sha(args.compare_original)
        reference = json.loads(args.compare_original.read_text(encoding='utf-8'))
        if any(reference['sources'].get(str(path)) != protected[str(path)] for path in (args.checkpoint, args.database)):
            raise ValueError('Original comparison requires the same audited checkpoint and individual')
        limit = torch.load(args.checkpoint, map_location='cpu', weights_only=True)['metadata']['position_table_max_position']
        raw = compare_original(base, tokenizer, limit, reference)
    results = {}
    with load_integrated_durable(args.database, args.checkpoint, base, MODEL) as runtime:
        original = frozen_digest(runtime.organism, runtime.state)
        for label, turns in SCENES.items():
            focus, history = None, deque(maxlen=_CHAT_TURNS)
            rows = []
            for question, expected_route, expected_value in turns:
                result, focus = session_turn(runtime.decoder, tokenizer, runtime.organism, runtime.state,
                                             runtime.observation_calendar, question, focus, history)
                generated = result['generated'] or {}
                passes = (result['routing']['reason'] == expected_route and
                          (expected_value is None or (generated.get('answer') == expected_value
                                                     and generated.get('matrix_value') == expected_value)))
                rows.append({'question': question, 'expected_route': expected_route, 'expected_value': expected_value,
                             'routing_and_memory_passed': passes, **result})
            results[label] = rows
            print(label, sum(row['routing_and_memory_passed'] for row in rows), len(rows), flush=True)
        if original != frozen_digest(runtime.organism, runtime.state):
            raise ValueError('Mixed dialogue changed the saved memory')
    if any(sha(Path(path)) != digest for path, digest in protected.items()):
        raise ValueError('Mixed dialogue changed a saved source')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, {'scenes': results, 'paired_original': raw, 'sources': protected,
                        'seconds': time.perf_counter() - started,
                        'peak_gpu_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'database', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--compare-original', type=Path, help='Prior mixed-chat report for the fixed-history comparison')
    main(parser.parse_args())
