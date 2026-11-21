"""Evaluate multi-turn Chinese generation at the anchored-window boundary."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoTokenizer

from scripts.audit_spora_dialogue_migration import dialogue
from scripts.audit_spora_native_query_key import write
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_checkpoint import load_integrated_runtime
from spora.language_training import frozen_digest


SCENARIOS = (
    ('寄存更正', (
        ('{text}\n\n我的行李寄存在西门。只回答寄存地点。', ('西门',)),
        ('刚才说错了，行李在东门。现在去哪取？只说地点。', ('东门',)),
        ('请重复最新的地点，去掉其他文字。', ('东门',)),
    )),
    ('人物交换', (
        ('小邱拿着纸扇，小许拿着雨伞。小许拿的是什么？请只答物品。\n\n{text}', ('雨伞',)),
        ('两个人交换了东西。现在小邱拿什么？只说物品。', ('雨伞',)),
        ('小许又把手里的东西送回去了。小许现在还拿着它吗？只答是或否。', ('否', '不', '没有')),
    )),
    ('日期前移', (
        ('{text}\n\n预约是周六。只回答星期几。', ('周六', '星期六')),
        ('预约提前两天，改到周几？只说星期。', ('周四', '星期四')),
        ('再推迟一天，现在定在周几？只说星期。', ('周五', '星期五')),
    )),
    ('格式继承', (
        ('{text}\n\n只写两个字，向我问好。', ('你好', '您好')),
        ('改成告别，还是两个字。', ('再见', '拜拜')),
        ('用两个字表达道谢。', ('谢谢', '感谢')),
    )),
)


def main(checkpoint, output):
    if output.exists():
        raise FileExistsError(output)
    source = ROOT / 'data/chinese-pilot/source.json'
    inputs = {str(path): sha(path) for path in (checkpoint, source)}
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL)
    runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    data = json.loads(source.read_text(encoding='utf-8'))
    pieces = (data['development'][::2], data['test'][::2], data['development'][1::2], data['test'][1::2])
    rows = {}
    for index, ((label, scenario), samples) in enumerate(zip(SCENARIOS, pieces, strict=True)):
        text = '\n'.join(samples + data['test'] + data['development'])
        tokens = tokenizer.encode(text, add_special_tokens=False)[:365]
        turns = tuple(question.format(text=tokenizer.decode(tokens)) for question, _ in scenario)
        generation = dialogue(runtime.decoder, tokenizer, turns, open_reasoning=False, limit=24, seed=25043 + index)
        for row, (_, expected) in zip(generation, scenario, strict=True):
            row['expected'] = list(expected)
            row['strict'] = row['answer'].strip('。！!，,；;？? \n') in expected
        rows[label] = generation
        print(label, [row['strict'] for row in generation], flush=True)
    if individual != frozen_digest(runtime.organism, runtime.state) or any(
        sha(Path(path)) != digest for path, digest in inputs.items()
    ):
        raise ValueError('The dialogue audit changed its source or individual')
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, {'dialogue': rows, 'layout': runtime.decoder.anchored_attention_layout, 'sources': inputs,
                   'seconds': time.perf_counter() - started, 'peak_gpu_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.checkpoint, args.output)
