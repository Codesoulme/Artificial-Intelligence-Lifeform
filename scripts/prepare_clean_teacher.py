"""Prepare a stricter training-only Chinese answer curriculum from v3."""
from __future__ import annotations
import collections
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/teaching-curriculum-v3'
OUTPUT = ROOT / 'runs/teaching-curriculum-v4'
REPORT = ROOT / 'reports/P30-clean-teacher-preparation.json'

IDENTITY = re.compile(r'(人工智能助手|语言模型|我是一个|我的知识|无法直接|不能直接|作为AI|OpenAI|GPT)', re.I)
CONTROL = re.compile(r'(<\||```|https?://|<script|\{\{|\[\[)')
HIGH_STAKES = re.compile(r'(诊断|治疗|药物|用药|剂量|处方|诉讼|股票|股价|投资|汇率|房价|疾病|病人|医疗|法律意见|最新规定)')
CODE = re.compile(r'(代码|编程|Python|JavaScript|SQL|API|HTTP|HTML|CSS|函数|变量|算法|程序)', re.I)
LIST_HEAVY = re.compile(r'(^|\n)\s*(?:\d+[.、)]|[-*•])')
REPEAT = re.compile(r'(.{2,20})\1{2,}')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_reason(messages):
    if not messages or messages[0].get('role') != 'user' or messages[-1].get('role') != 'assistant':
        return 'invalid_roles'
    prompt = messages[0].get('content', '').strip()
    answer = messages[-1].get('content', '').strip()
    joined = prompt + '\n' + answer
    if not prompt or not answer:
        return 'empty'
    if IDENTITY.search(joined):
        return 'identity_or_capability_script'
    if CONTROL.search(joined):
        return 'control_or_url'
    if HIGH_STAKES.search(prompt):
        return 'high_stakes_or_current_fact'
    if CODE.search(prompt + '\n' + answer):
        return 'code_or_technical_template'
    if len(answer) < 18 or len(answer) > 260:
        return 'answer_length'
    if LIST_HEAVY.search(answer) or answer.count('\n') > 3:
        return 'list_heavy'
    if REPEAT.search(answer.replace(' ', '')):
        return 'repeated_phrase'
    han = sum('\u3400' <= c <= '\u9fff' for c in answer)
    latin = sum('a' <= c.lower() <= 'z' for c in answer)
    if han < 8 or han / max(1, han + latin) < .78:
        return 'low_chinese_ratio'
    if not re.search(r'[。！？!?；;：:，,]$', answer):
        return 'unfinished_or_fragment'
    return None


def main():
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError('Preserve previous result; choose a fresh version')
    source_manifest = json.loads((SOURCE / 'manifest.json').read_text('utf-8'))
    for name, digest in source_manifest['files'].items():
        if sha(SOURCE / name) != digest:
            raise ValueError('Source changed: ' + name)
    OUTPUT.mkdir(parents=True)
    for name in ('foundations.jsonl', 'foundations-tokens.jsonl'):
        shutil.copyfile(SOURCE / name, OUTPUT / name)
    stats = collections.Counter()
    kept = 0
    targets = 0
    with (SOURCE / 'teacher.jsonl').open(encoding='utf-8') as raw, \
         (SOURCE / 'teacher-tokens.jsonl').open(encoding='utf-8') as encoded, \
         (OUTPUT / 'teacher.jsonl').open('x', encoding='utf-8') as out_raw, \
         (OUTPUT / 'teacher-tokens.jsonl').open('x', encoding='utf-8') as out_encoded:
        for raw_line, encoded_line in zip(raw, encoded):
            row = json.loads(raw_line)
            reason = clean_reason(row['messages'])
            if reason:
                stats['removed_' + reason] += 1
                continue
            out_raw.write(raw_line)
            out_encoded.write(encoded_line)
            kept += 1
            value = json.loads(encoded_line)
            targets += sum(value['mask'])
    result = {
        'status': 'prepared_training_only_clean_subset',
        'source': 'runs/teaching-curriculum-v3',
        'source_manifest_sha256': sha(SOURCE / 'manifest.json'),
        'tokenizer_sha256': source_manifest['tokenizer_sha256'],
        'teacher_rows_kept': kept,
        'teacher_targets_kept': targets,
        'source_teacher_rows': source_manifest['stats']['teacher_dialogues'],
        'stats': dict(stats),
        'files': {p.name: sha(p) for p in OUTPUT.iterdir() if p.is_file()},
        'scope': 'Stricter training-only subset for a bounded chat curriculum. It removes common identity scripts, high-stakes/current prompts, code-heavy answers, list-heavy answers, repeated phrases and fragments. It is not a factual correctness audit and does not use development/test text.',
        'limitations': ['Single mechanical filter; no blind human review.', 'No new facts or answers were generated.', 'Parent v3 already passed its finite isolation checks; this subset inherits that boundary.'],
    }
    (OUTPUT / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('teacher_rows_kept', 'teacher_targets_kept', 'stats')}))


if __name__ == '__main__':
    main()
