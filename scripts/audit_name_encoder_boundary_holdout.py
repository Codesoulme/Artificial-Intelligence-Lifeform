"""Compare exact name boundaries without involving memory intent or stored facts."""

import argparse
import json
from pathlib import Path

import torch

from scripts.audit_registered_name_spans import registry
from scripts.audit_registered_name_teacher import annotate, sha
from transformers import AutoModelForTokenClassification, AutoTokenizer


NAMES = ('小玫', '阿骞', '曹熙', '郁霏', '余亦珂', '孔思禾', '顾芷', '单榆')
SUFFIXES = ('然', '川', '越', '宁', '和', '信', '明', '白')
PREFIXES = ('陈', '江', '史', '周', '何', '黄', '田', '唐')


def fixture():
    templates = (
        '上面那张便笺是{name}提交的，给他拿什么饮料？',
        '按材料右边所写，麻烦给{name}找那杯喝的。',
        '{name}那份单子列的是哪款饮品？',
        '麻烦核查以{name}署名的那张纸上的点单内容。',
    )
    for key, (name, suffix, prefix) in enumerate(zip(NAMES, SUFFIXES, PREFIXES, strict=True)):
        for template in templates:
            for kind, value in (('exact', name), ('extended_right', name + suffix), ('extended_left', prefix + name)):
                question = template.format(name=value)
                start = question.index(value)
                yield {'question': question, 'kind': kind, 'expected': key if kind == 'exact' else None,
                       'mentions': [[start, start + len(value)]]}
        other = ('申翎', '林唯', '凌弦', '潘幼西', '冯悠', '翟羽桐', '殷菀', '伍濯')[key]
        question = f'可以分别说一下{name}还有{other}都选了什么吗？'
        yield {'question': question, 'kind': 'multiple', 'expected': None,
               'mentions': [[question.index(person), question.index(person) + len(person)] for person in (name, other)]}


def main(encoders, output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    fingerprints = registry(NAMES)
    rows = list(fixture())
    for row in rows:
        row['candidates'] = fingerprints.matching_spans(row['question'], max_name_chars=12)
    results = {}
    for directory in encoders:
        digest = sha(directory / 'model.safetensors')
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(
            directory, local_files_only=True, use_safetensors=True, trust_remote_code=False).eval()
        outcomes = []
        outputs = annotate(model, tokenizer, [row['question'] for row in rows])
        for row, tags in zip(rows, outputs, strict=True):
            detected, candidates = tags['bio'], row['candidates']
            routed = None
            if len(detected) == len(candidates) == 1 and detected[0] == list(candidates[0][:2]):
                routed = candidates[0][2]
            outcomes.append({**row, 'detected': detected, 'route': routed})
        summary = {}
        for kind in sorted({row['kind'] for row in outcomes}):
            subset = [row for row in outcomes if row['kind'] == kind]
            summary[kind] = {'total': len(subset),
                             'span_exact': sum(row['detected'] == row['mentions'] for row in subset),
                             'route_correct': sum(row['route'] == row['expected'] for row in subset),
                             'false_links': sum(row['route'] is not None and row['route'] != row['expected']
                                                for row in subset)}
        print(directory.name, summary, flush=True)
        if sha(directory / 'model.safetensors') != digest:
            raise ValueError('Encoder changed during holdout')
        results[str(directory)] = {'weights_sha256': digest, 'summary': summary, 'rows': outcomes}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoders', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(**vars(parser.parse_args()))
