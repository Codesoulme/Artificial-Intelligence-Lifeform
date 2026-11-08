"""Run only after the named P112/P113 jobs have ended. Uses synthetic data only."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('p114', ROOT/'scripts/p114_disk_story_dedupe.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


def other_jobs():
    command = "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Where-Object { $_.CommandLine -match 'p112_controlled_screen.py|p113-real-no-df-v1|p113_story_dedupe.py' } | Select-Object -ExpandProperty ProcessId"
    result = subprocess.run(['powershell', '-NoProfile', '-Command', command], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def constructed_pair_counts(path, count):
    expected = matched = quarantined = observed = 0
    prior = None
    with path.open(encoding='utf-8') as groups:
        for line in groups:
            item = json.loads(line)
            observed += 1
            if item['source_line'] != observed:
                raise ValueError('nonsequential group output')
            if (observed - 1) % 10 == 1:
                expected += 1
                matched += item['representative_line'] == prior['representative_line']
                quarantined += item['status'] == 'quarantine' or prior['status'] == 'quarantine'
            prior = item
    if observed != count:
        raise ValueError('incomplete group output')
    return {'expected': expected, 'same_group': matched, 'quarantined': quarantined}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=10000)
    parser.add_argument('--min-available-mib', type=int, default=1536)
    parser.add_argument('--max-memory-mib', type=int, default=192)
    parser.add_argument('--max-disk-mib', type=int, default=512)
    a = parser.parse_args()
    out = a.output.resolve()
    if not out.is_relative_to(p.RUNS) or out == p.RUNS or a.count < 2:
        raise ValueError('output must be a new directory under runs; count >= 2')
    jobs = other_jobs()
    if jobs:
        raise RuntimeError('protected P112/P113 jobs still running: '+jobs)
    if p.memory_info()['available_physical'] < a.min_available_mib*1048576:
        raise RuntimeError('available physical memory below guard')
    out.mkdir(parents=False, exist_ok=False)
    source = out/'synthetic.jsonl'
    rng = random.Random(114)
    started = time.monotonic()
    with source.open('x', encoding='utf-8', newline='\n') as f:
        for i in range(a.count):
            if i % 100 == 0 and p.memory_info()['available_physical'] < a.min_available_mib*1048576:
                raise RuntimeError('available physical memory below guard during generation')
            if i % 10 == 1:
                text = '甲'+previous[1:]
            else:
                text = ''.join(chr(0x4e00+rng.randrange(6000)) for _ in range(96))
                previous = text
            sha = hashlib.sha256(text.encode()).hexdigest()
            f.write(json.dumps(dict(text=text, group='synthetic', sha256=sha), ensure_ascii=False)+'\n')
    if other_jobs():
        raise RuntimeError('protected job started during generation')
    # Windows venv python.exe is a launcher with a separate interpreter child;
    # use the base interpreter so monitoring and termination target the worker.
    cmd = [getattr(sys, '_base_executable', sys.executable), '-B', str(ROOT/'scripts/p114_disk_story_dedupe.py'), '--input', str(source),
           '--output', str(out/'result'), '--min-available-mib', str(a.min_available_mib),
           '--max-memory-mib', str(a.max_memory_mib), '--max-disk-mib', str(a.max_disk_mib)]
    with (out/'stdout.txt').open('x', encoding='utf-8') as stdout, (out/'stderr.txt').open('x', encoding='utf-8') as stderr:
        child = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [(x, ctypes.c_size_t) for x in
                ('peak_working', 'working', 'peak_paged', 'paged', 'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]
        ctypes.windll.kernel32.OpenProcess.restype = wintypes.HANDLE
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        ctypes.windll.kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = ctypes.windll.kernel32.OpenProcess(0x1000 | 0x10, False, child.pid)
        if not handle:
            child.terminate(); child.wait()
            raise OSError('cannot monitor benchmark child')
        minimum = None; peak_working = peak_commit = 0; stop_reason = None
        try:
            while child.poll() is None:
                available = p.memory_info()['available_physical']
                minimum = min(minimum or available, available)
                counters = Counters(); counters.cb = ctypes.sizeof(counters)
                # memory_info uses a different local ctypes structure and resets
                # the shared function's argtypes on every call.
                ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
                if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                    stop_reason = 'cannot monitor child'; child.terminate(); break
                peak_working = max(peak_working, counters.peak_working)
                peak_commit = max(peak_commit, counters.peak_pagefile)
                if available < a.min_available_mib*1048576 or counters.pagefile > a.max_memory_mib*1048576:
                    stop_reason = 'memory guard'; child.terminate(); break
                time.sleep(.25)
            child.wait()
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait()
            ctypes.windll.kernel32.CloseHandle(handle)
    measurement = dict(count=a.count, exit_code=child.returncode, stop_reason=stop_reason,
        wall_seconds=time.monotonic()-started, peak_working_set_sampled=peak_working,
        peak_commit_sampled=peak_commit, minimum_available_physical_sampled=minimum)
    if not child.returncode and not stop_reason:
        measurement['constructed_pairs'] = constructed_pair_counts(out/'result'/'groups.jsonl', a.count)
    p.write_json(out/'measurement.json', measurement)
    print(json.dumps(measurement, indent=2))
    if child.returncode or stop_reason:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
