"""Audit model-generated name extraction plus exact tensor fingerprint lookup.

The instruction is a diagnostic prompt, not SPORA's autonomous memory or a
security boundary. Only a temporary synthetic fingerprint table is used.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_local_baseline as base
from scripts import deepseek_spora_adapter_restart_audit as binding_source
from scripts import deepseek_spora_resume_equivalence_audit as io
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.identity_fingerprints import IdentityFingerprints

OUT = ROOT / 'runs/deepseek-spora-name-extraction-audit-20261116'
ENROLLED = ('小林', '阿宁', '小雨', '阿岚', '小乔')
CASES = (
    ('known', '小林喜欢喝什么？', '小林'),
    ('known', '请告诉我阿宁住在哪里。', '阿宁'),
    ('known', '记录里小雨的个人爱好是什么？', '小雨'),
    ('known', '你还记得阿岚的城市吗？', '阿岚'),
    ('known', '查阅小乔的饮品偏好。', '小乔'),
    ('known', '我想知道小林目前喝什么。', '小林'),
    ('known', '关于阿宁，你了解她的爱好吗？', '阿宁'),
    ('known', '那小雨呢？', '小雨'),
    ('near', '小林月喜欢喝什么？', '小林月'),
    ('near', '请告诉我小林林住在哪里。', '小林林'),
    ('near', '记录里林小林的个人爱好是什么？', '林小林'),
    ('near', '我想知道小林子目前喝什么。', '小林子'),
    ('near', '阿宁宁喜欢什么饮品？', '阿宁宁'),
    ('near', '关于小阿宁，你了解她的爱好吗？', '小阿宁'),
    ('near', '小雨晴住在哪座城市？', '小雨晴'),
    ('near', '记录里阿岚青喜欢什么？', '阿岚青'),
    ('near', '小乔乔喜欢喝什么？', '小乔乔'),
    ('near', '乔小乔现在住在哪里？', '乔小乔'),
    ('unknown', '阿青喜欢什么饮品？', '阿青'),
    ('unknown', '小阮住在哪里？', '小阮'),
    ('unknown', '阿沛的兴趣爱好是什么？', '阿沛'),
    ('unknown', '小柏业余做什么？', '小柏'),
    ('multi', '小林和阿宁都喜欢喝什么？', None),
    ('multi', '比较小雨和小乔的爱好。', None),
    ('multi', '小林月和阿岚各住在哪里？', None),
    ('multi', '阿宁与小林是什么关系？', None),
    ('none', '你好，今天怎么样？', None),
    ('none', '请推荐一杯不加糖的饮料。', None),
    ('none', '我以前告诉过你我的爱好吗？', None),
    ('none', '北京今天的天气如何？', None),
)
INSTRUCTION = ('从下面的句子中提取人物姓名。只有一个明确人物时，原样完整输出其姓名；'
               '没有人物或有多个不同人物时输出 null。只输出 JSON 对象，格式为 '
               '{"name":"姓名"} 或 {"name":null}，不要解释。\n句子：')


def parse_response(text, question):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result

    try:
        parsed = json.loads(text, object_pairs_hook=unique_object)
    except (TypeError, ValueError):
        return None, 'invalid_json'
    if not isinstance(parsed, dict) or set(parsed) != {'name'}:
        return None, 'invalid_schema'
    name = parsed['name']
    if name is None:
        return None, 'null'
    if not isinstance(name, str) or not name or len(name) > 32 or question.count(name) != 1:
        return None, 'invalid_name'
    return name, 'name'


def parse_embedded_response(text, question):
    """Accept exactly one decodable object in surrounding text for comparison."""
    name, status = parse_response(text, question)
    if status != 'invalid_json':
        return name, status
    if not isinstance(text, str):
        return None, 'invalid_json'
    decoder = json.JSONDecoder()
    objects = []
    for index, character in enumerate(text):
        if character != '{':
            continue
        try:
            value, end = decoder.raw_decode(text, index)
        except ValueError:
            continue
        if isinstance(value, dict):
            objects.append(text[index:end])
    if len(objects) != 1:
        return None, 'ambiguous_json' if objects else 'invalid_json'
    return parse_response(objects[0], question)


def rescore():
    report = json.loads((OUT / 'result.json').read_text(encoding='utf-8'))
    fingerprints = IdentityFingerprints(len(ENROLLED))
    for key, name in enumerate(ENROLLED):
        fingerprints.bind(key, name)
    rows = []
    for source in report['rows']:
        name, status = parse_embedded_response(source['raw_output'], source['question'])
        key = fingerprints.lookup(name) if name is not None else None
        rows.append({'group': source['group'], 'question': source['question'], 'name': name,
                     'parse_status': status, 'matched_key': key, 'expected_key': source['expected_key'],
                     'exact_extraction': name == source['expected_name'] and status in ('name', 'null'),
                     'correct_route': key == source['expected_key'],
                     'false_accept': source['expected_key'] is None and key is not None})
    io.write_json(OUT / 'embedded-rescore.json', {'rows': rows,
                  'limitations': ['A single embedded JSON object may still come from misleading surrounding text.',
                                  'Model-generated partial names can match a different enrolled person.',
                                  'This is an offline parser comparison, not a security boundary.']})
    print(json.dumps({'rows': len(rows), 'false_accepts': sum(row['false_accept'] for row in rows)}), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    base_sha = base.sha(base.MODEL / 'model.safetensors')
    feature_sha, files = binding_source.binding()
    if base_sha != base.WEIGHT_SHA:
        raise ValueError('Unexpected base weight')
    torch.manual_seed(17381)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    model.requires_grad_(False)
    fingerprints = IdentityFingerprints(len(ENROLLED))
    for key, name in enumerate(ENROLLED):
        fingerprints.bind(key, name)
    rows = []
    for group, question, expected in CASES:
        inputs = pilot.closed_think_inputs(tokenizer, INSTRUCTION + question, 'cuda')
        with torch.inference_mode():
            output = model.generate(**inputs, use_cache=True, max_new_tokens=48, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        token_ids = output[0, inputs['input_ids'].shape[1]:].tolist()
        text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        name, status = parse_response(text, question)
        matched = fingerprints.lookup(name) if name is not None else None
        expected_key = ENROLLED.index(expected) if expected in ENROLLED else None
        row = {'group': group, 'question': question, 'expected_name': expected,
               'raw_output': text, 'token_ids': token_ids, 'name': name, 'parse_status': status,
               'matched_key': matched, 'expected_key': expected_key,
               'exact_extraction': name == expected and status in ('name', 'null'),
               'correct_route': matched == expected_key, 'false_accept': expected_key is None and matched is not None}
        rows.append(row)
        print(json.dumps({'group': group, 'status': status, 'false_accept': row['false_accept']}), flush=True)
    if base.sha(base.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Base weights changed')
    torch.save({'fingerprints': fingerprints.state_dict(), 'key_count': len(ENROLLED),
                'base_sha256': base_sha, 'feature_sha256': feature_sha}, OUT / 'synthetic-identity.pt')
    report = {'base_sha256': base_sha, 'feature_sha256': feature_sha, 'feature_files': files,
              'rows': rows, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Model prompt is a diagnostic extraction task, not autonomous identity recognition.',
                              'Fingerprint table is explicitly enrolled, not learned memory or identity proof.',
                              'A substring check cannot prove that a generated name is complete.',
                              'Small synthetic sample with no independent benchmark.',
                              'Greedy generation with a closed thinking prefix and 48-token cap.',
                              'No real individual, direct memory reads, alias or same-name resolution.']}
    io.write_json(OUT / 'result.json', report)
    print(json.dumps({'rows': len(rows), 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--rescore']:
        rescore()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit('Usage: script [--rescore]')
