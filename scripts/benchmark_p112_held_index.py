"""Bounded, training-only comparison; emits aggregate metrics and hashes only."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TRAIN = ROOT / 'runs/p28-stories-data/candidate-v3/pretrain-train.jsonl'
SAMPLES = ROOT / 'runs/p112-story-quality-sample/samples.jsonl'
SAMPLE_MANIFEST = ROOT / 'runs/p112-story-quality-sample/manifest.json'
P28_MANIFEST = ROOT / 'runs/p28-stories-data/candidate-v3/manifest.json'
REPORT = ROOT / 'reports/P112-index-benchmark.md'
PIN = 'fd57c0ffa3878c1aef3e762d4c4153672c7c899088f6369ceec33663243af5bd'
LIMIT, QUERIES, TIMEOUT = 10000, 80, 570


def sha(data):
    return hashlib.sha256(data).hexdigest()


def restrict():
    if os.name == 'nt':
        import ctypes
        k = ctypes.WinDLL('kernel32', use_last_error=True)
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        process = k.GetCurrentProcess()
        mask, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not k.SetPriorityClass(process, 0x00004000):
            raise OSError('priority')
        if not k.GetProcessAffinityMask(process, ctypes.byref(mask), ctypes.byref(system)):
            raise OSError('affinity read')
        if not k.SetProcessAffinityMask(process, 1 << (mask.value.bit_length() - 1)):
            raise OSError('affinity write')


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    def percentile(p):
        return values[max(0, math.ceil(p * len(values)) - 1)]
    return {'n': len(values), 'total_ms': round(sum(values), 3),
            'min_ms': round(values[0], 3), 'median_ms': round(statistics.median(values), 3),
            'p95_ms': round(percentile(.95), 3), 'max_ms': round(values[-1], 3)}


def worker(kind):
    # Importing the legacy module adjusts its own affinity; restrict afterwards.
    from scripts.prepare_p28_prose import HeldIndex, normalized, digest
    from spora.fast_held_index import FastHeldIndex
    restrict()
    p28_bytes = P28_MANIFEST.read_bytes()
    sample_meta = json.loads(SAMPLE_MANIFEST.read_bytes())
    samples_bytes = SAMPLES.read_bytes()
    if sha(p28_bytes) != PIN or sample_meta['manifest_sha256'] != PIN or sha(samples_bytes) != sample_meta['sample_file_sha256']:
        raise ValueError('manifest mismatch')
    p28 = json.loads(p28_bytes)
    if p28['files']['pretrain-train.jsonl'] != sample_meta['original_train_sha256']:
        raise ValueError('train binding mismatch')
    queries = []
    for line in samples_bytes.splitlines():
        row = json.loads(line)
        if row['condition'] == 'B':
            if digest(normalized(row['text'])) != row['normalized_sha256'] or digest(row['text']) != row['source_original_sha256']:
                raise ValueError('query identity')
            queries.append(row['text'])
    if len(queries) != QUERIES:
        raise ValueError('query count')
    index = (HeldIndex if kind == 'held' else FastHeldIndex)()
    h = hashlib.sha256()
    build = []
    start_build = time.perf_counter()
    with TRAIN.open('rb') as stream:
        for _ in range(LIMIT):
            line = stream.readline()
            if not line:
                raise ValueError('training count')
            h.update(line)
            row = json.loads(line)
            text = row['text']
            if digest(normalized(text)) != row['sha256'] or digest(text) != row['source_original_sha256']:
                raise ValueError('training identity')
            start = time.perf_counter_ns()
            index.add(text)
            build.append((time.perf_counter_ns() - start) / 1e6)
    elapsed_build = (time.perf_counter() - start_build) * 1000
    reasons, timings = [], []
    for query in queries:
        start = time.perf_counter_ns()
        reasons.append(index.reason(query))
        timings.append((time.perf_counter_ns() - start) / 1e6)
    allowed = {None, 'held_exact', 'held_leading96', 'held_sampled_lexical'}
    if any(reason not in allowed for reason in reasons):
        raise ValueError('unknown reason')
    result = {'kind': kind, 'train_count': len(build), 'query_count': len(queries),
              'train_prefix_sha256': h.hexdigest(), 'sample_file_sha256': sha(samples_bytes),
              'sample_manifest_sha256': sha(SAMPLE_MANIFEST.read_bytes()),
              'p28_manifest_sha256': sha(p28_bytes),
              'reasons': reasons, 'build_wall_ms': round(elapsed_build, 3),
              'build_add': distribution(build), 'query_all': distribution(timings),
              'query_matched': distribution([v for v, r in zip(timings, reasons) if r]),
              'query_unmatched': distribution([v for v, r in zip(timings, reasons) if not r])}
    print(json.dumps(result, separators=(',', ':')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=('held', 'fast'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    restrict()
    started = time.monotonic()
    results = []
    for kind in ('held', 'fast'):
        remaining = TIMEOUT - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('benchmark timeout')
        process = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--worker', kind],
                                 cwd=ROOT, capture_output=True, text=True, timeout=remaining)
        if process.returncode:
            raise RuntimeError('worker failure')
        results.append(json.loads(process.stdout))
    a, b = results
    for key in ('train_count', 'query_count', 'train_prefix_sha256', 'sample_file_sha256',
                'sample_manifest_sha256', 'p28_manifest_sha256'):
        if a[key] != b[key]:
            raise ValueError('worker inputs differ')
    mismatches = sum(x != y for x, y in zip(a['reasons'], b['reasons']))
    counts = Counter(r or 'unmatched' for r in a['reasons'])
    reason_digest = sha(json.dumps(a['reasons'], separators=(',', ':')).encode())
    lines = ['# P112 索引有界基准', '',
        f'- 状态：{"一致" if not mismatches else "存在差异"}；逐条判定差异 {mismatches} / {a["query_count"]}。',
        f'- 输入：固定 P28 训练文件前 {LIMIT} 行；B 条目按文件顺序全部 {QUERIES} 条。两个实现分别在独立进程中顺序运行。',
        '- 限制：单核、低优先级；未读取测试或开发正文；不训练。总时限 570 秒。',
        '- 计时：单次冷建索引及每条查询各一次，毫秒；P95 为最近秩。建索引总墙钟含读取、解析和校验；add 分布仅含索引插入。',
        '- 此次为共享机器上的一次实测，调度与先后顺序会影响结果；样本无法证明全域等价。', '',
        '## 判定统计', '',
        '| 类型 | 数量 |', '|---|---:|']
    for reason in ('held_exact', 'held_leading96', 'held_sampled_lexical', 'unmatched'):
        lines.append(f'| {reason} | {counts[reason]} |')
    lines += ['', '## 耗时分布（毫秒）', '', '| 实现 / 范围 | 数量 | 总计 | 最小 | 中位数 | P95 | 最大 |', '|---|---:|---:|---:|---:|---:|---:|']
    for result in results:
        for metric in ('build_add', 'query_all', 'query_matched', 'query_unmatched'):
            d = result[metric]
            values = ([d[k] for k in ('n', 'total_ms', 'min_ms', 'median_ms', 'p95_ms', 'max_ms')] if d else [0, '-', '-', '-', '-', '-'])
            lines.append('| ' + result['kind'] + ' / ' + metric + ' | ' + ' | '.join(map(str, values)) + ' |')
    for result in results:
        lines.append(f'\n- {result["kind"]} 建索引总墙钟：{result["build_wall_ms"]} ms。')
    lines += ['', '## SHA-256', '']
    for name, value in [('P28 manifest', a['p28_manifest_sha256']), ('训练文件前缀原始字节', a['train_prefix_sha256']),
                        ('抽样文件', a['sample_file_sha256']), ('抽样清单', a['sample_manifest_sha256']),
                        ('顺序判定数组（紧凑 JSON）', reason_digest), ('脚本', sha(Path(__file__).read_bytes())),
                        ('HeldIndex 源文件', sha((ROOT/'scripts/prepare_p28_prose.py').read_bytes())),
                        ('FastHeldIndex 源文件', sha((ROOT/'spora/fast_held_index.py').read_bytes()))]:
        lines.append(f'- {name}: `{value}`')
    lines += ['', f'- 完成耗时：{round(time.monotonic() - started, 3)} 秒。',
              '- 训练前缀摘要绑定实际读取字节；未重新散列整个训练文件，故清单与全文的当前字节一致性未单独复核。', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'mismatches': mismatches, 'queries': a['query_count'], 'matched': sum(v for k, v in counts.items() if k != 'unmatched'), 'unmatched': counts['unmatched'], 'report_sha256': sha(REPORT.read_bytes())}))
    if mismatches:
        sys.exit(2)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__}), file=sys.stderr)
        sys.exit(1)
