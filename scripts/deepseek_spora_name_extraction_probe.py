"""Isolated base-model name extraction probe; no memory or enrollment access."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoTokenizer

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed

OUTPUT = ROOT / 'runs/deepseek-spora-name-extraction-probe-20261114.json'
CASES = (
    ('near', '你记得小林舟喜欢喝哪种饮品吗？', ['小林舟']),
    ('near', '阿远晨爱喝什么？', ['阿远晨']),
    ('near', '小雨舟的饮料是什么？', ['小雨舟']),
    ('near', '告诉我阿明晨的偏好。', ['阿明晨']),
    ('chinese', '请查一下欧阳晓月最近喜欢什么饮料。', ['欧阳晓月']),
    ('chinese', '王小明昨天喝了什么？', ['王小明']),
    ('chinese', '张伟的朋友今天来了。', ['张伟']),
    ('chinese', '你记得李娜的喜好吗？', ['李娜']),
    ('english', 'Alice喜欢什么饮品？', ['Alice']),
    ('english', '请问Mary Jane平时喝什么？', ['Mary Jane']),
    ('english', 'John和Anna分别喝什么？', ['John', 'Anna']),
    ('english', '小林和Alex是朋友。', ['小林', 'Alex']),
    ('multiple', '小林和小林舟各喜欢什么？', ['小林', '小林舟']),
    ('multiple', '阿远告诉阿宁，他喜欢茶。', ['阿远', '阿宁']),
    ('multiple', '王小明与李娜今天见面。', ['王小明', '李娜']),
    ('multiple', '陈晨、张伟和小梅都到了。', ['陈晨', '张伟', '小梅']),
    ('none', '现在几点了？', []),
    ('none', '我喜欢喝茶。', []),
    ('none', '公园里的树叶落下了。', []),
    ('none', '你还记得她喜欢什么吗？', []),
    ('alias', '小林也叫林晓，请查小林的喜好。', ['小林', '林晓']),
    ('alias', '阿宁，也就是宁悦，今天来过。', ['阿宁', '宁悦']),
    ('same_name', '两个都叫张伟的人分别喝什么？', ['张伟']),
    ('same_name', '班里两位李娜都喜欢茶吗？', ['李娜']),
)
PROMPT = ('只提取下列句子中明确出现的人名，保持原文完整拼写，不要推断代词指代。'
          '每个不同名字出现一次，按首次出现顺序输出JSON字符串数组；没有人名输出[]。'
          '不要输出解释或其他文字。\n句子：')


def parse_names(answer, question):
    try:
        value = json.loads(answer)
    except (ValueError, TypeError):
        return None
    if (not isinstance(value, list) or any(not isinstance(x, str) or not x or x not in question for x in value)
            or len(set(value)) != len(value)):
        return None
    return value


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Base checksum mismatch')
    if len({question for _, question, _ in CASES}) != len(CASES):
        raise ValueError('Duplicate questions')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    rows = []
    for group, question, expected in CASES:
        inputs = closed_think_inputs(tokenizer, PROMPT + question, 'cuda')
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=48, do_sample=False, use_cache=True,
                                 pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(ids[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        parsed = parse_names(answer, question)
        row = {'group': group, 'question': question, 'expected': expected, 'answer': answer,
               'parsed': parsed, 'exact': parsed == expected}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Base weights changed')
    summary = {}
    for group in sorted({row['group'] for row in rows}):
        subset = [row for row in rows if row['group'] == group]
        summary[group] = {'cases': len(subset), 'valid_json': sum(row['parsed'] is not None for row in subset),
                          'exact': sum(row['exact'] for row in subset)}
    result = {'summary': summary, 'rows': rows, 'prompt': PROMPT, 'base_sha256': WEIGHT_SHA,
              'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Small hand-authored development sample, not representative.',
                              'Output is name text, not entity identity, authorization or memory access.',
                              'Same-name and alias cases evaluate text only; no disambiguation.',
                              'Closed think prefix and deterministic decoding; no training or adaptation.',
                              'No deployment and no independent guarantee against shared errors.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
