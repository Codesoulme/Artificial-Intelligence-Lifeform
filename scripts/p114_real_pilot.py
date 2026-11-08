"""Bounded, sequential P112-candidate-only pilot. Outputs stay under runs."""
import argparse
from collections import Counter
import ctypes
from ctypes import wintypes
import hashlib
import heapq
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'runs/p112-controlled-screen/candidates.jsonl'
MANIFEST = SOURCE.parent/'manifest.json'
BENCH = ROOT/'runs/p114-synthetic-10k-20261109-v2'
SEED = 'p114-real-pilot-20261109-v1'
spec = importlib.util.spec_from_file_location('p114', ROOT/'scripts/p114_disk_story_dedupe.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path, obj):
    p.write_json(path, obj)


def identity(path):
    s = path.stat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, inode=s.st_ino, device=s.st_dev)


def check_benchmark():
    m, s = load(BENCH/'measurement.json'), load(BENCH/'result/summary.json')
    assert m['exit_code'] == 0 and m['stop_reason'] is None and m['count'] == 10000
    assert m['constructed_pairs'] == dict(expected=1000, same_group=1000, quarantined=0)
    assert s['status'] == 'candidate_not_trained' and s['input']['records'] == 10000
    assert sha(BENCH/'synthetic.jsonl') == s['input']['sha256']
    for name, expected in s['output_sha256'].items():
        assert sha(BENCH/'result'/name) == expected
    count = 0
    with (BENCH/'result/groups.jsonl').open(encoding='utf-8') as f:
        for count, line in enumerate(f, 1):
            assert json.loads(line)['source_line'] == count
    assert count == 10000
    return dict(measurement_sha256=sha(BENCH/'measurement.json'), summary_sha256=sha(BENCH/'result/summary.json'),
                input_sha256=s['input']['sha256'], groups=count)


def sample(out):
    manifest = load(MANIFEST)
    assert manifest['status'] == 'candidate_only_not_trained' and not manifest['trained']
    expected = manifest['files']['candidates.jsonl']
    before = identity(SOURCE)
    h = hashlib.sha256(); heap = []; total = 0
    with SOURCE.open('rb') as f:
        while True:
            offset = f.tell(); line = f.readline(131073)
            if not line: break
            if len(line) > 131072: raise ValueError('source line exceeds limit')
            h.update(line); total += 1
            row = json.loads(line)
            key = hashlib.sha256(f"{SEED}:{total}:{row['source_row_1based']}:{row['sha256']}".encode()).digest()
            score = int.from_bytes(key, 'big')
            item = (-score, -total, offset, len(line))
            if len(heap) < 2000: heapq.heappush(heap, item)
            elif item > heap[0]: heapq.heapreplace(heap, item)
            if total % 1000 == 0: guard_self()
    assert h.hexdigest() == expected and total == manifest['exported'] and total >= 2000
    assert identity(SOURCE) == before
    top = sorted(heap, reverse=True)
    for size in (500, 2000):
        selected = sorted(top[:size], key=lambda x: -x[1])
        directory = out/str(size); directory.mkdir()
        with SOURCE.open('rb') as source, (directory/'sample.jsonl').open('xb') as output, (directory/'mapping.jsonl').open('x', encoding='utf-8') as mapping:
            for index, (_, minus_line, offset, length) in enumerate(selected, 1):
                source.seek(offset); raw = source.read(length); row = json.loads(raw)
                output.write(raw)
                mapping.write(json.dumps(dict(sample_line=index, candidate_line=-minus_line,
                    source_row_1based=row['source_row_1based'], score=hashlib.sha256(
                    f"{SEED}:{-minus_line}:{row['source_row_1based']}:{row['sha256']}".encode()).hexdigest()))+'\n')
    after_hash = sha(SOURCE)
    assert after_hash == expected and identity(SOURCE) == before
    dump(out/'sampling.json', dict(source_sha256=expected, source_identity=before, source_rows=total,
         manifest_sha256=sha(MANIFEST), script_sha256=sha(Path(__file__)), p114_sha256=sha(Path(p.__file__)),
         seed=SEED, samples={str(size):dict(sha256=sha(out/str(size)/'sample.jsonl'),
         mapping_sha256=sha(out/str(size)/'mapping.jsonl')) for size in (500,2000)}))


def guard_self():
    mem = p.memory_info()
    if mem['available_physical'] < 1536*1048576 or mem['commit'] > 192*1048576:
        raise RuntimeError('memory guard')


def verify(out, size):
    d = out/str(size); result = d/'result'; summary = load(result/'summary.json'); sampling = load(out/'sampling.json')
    assert summary['status'] == 'candidate_not_trained' and summary['input']['records'] == size
    assert sha(d/'sample.jsonl') == summary['input']['sha256'] == sampling['samples'][str(size)]['sha256']
    for name, expected in summary['output_sha256'].items(): assert sha(result/name) == expected
    db = sqlite3.connect(f'file:{(result/"index.sqlite3").as_posix()}?mode=ro', uri=True)
    db.execute('PRAGMA cache_size=-4096')
    assert db.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
    assert db.execute('SELECT COUNT(*) FROM records').fetchone()[0] == size
    assert db.execute('SELECT COUNT(*) FROM membership').fetchone()[0] == size
    assert db.execute('SELECT COUNT(*) FROM (SELECT root,MIN(n) m FROM membership GROUP BY root HAVING root != m)').fetchone()[0] == 0
    expected_risks = set(db.execute('SELECT DISTINCT m.root,r.reason FROM membership m JOIN risks r ON r.n=m.n'))
    actual_risks = set(db.execute('SELECT root,reason FROM group_risks'))
    assert expected_risks == actual_risks
    for item in map(json.loads, (result/'unchecked_buckets.jsonl').read_text(encoding='utf-8').splitlines()):
        missing = db.execute('SELECT COUNT(*) FROM buckets b LEFT JOIN risks r ON r.n=b.n AND r.reason=? WHERE b.band=? AND b.sig=? AND r.n IS NULL', (item['reason'],item['band'],bytes.fromhex(item['signature']))).fetchone()[0]
        assert missing == 0
    reasons = Counter(); sizes = Counter(); candidate_count = quarantined = count = 0
    with (result/'groups.jsonl').open(encoding='utf-8') as groups, (result/'candidates.jsonl').open('rb') as cand, (result/'quarantine.jsonl').open('rb') as quar:
        for count, line in enumerate(groups,1):
            g=json.loads(line); assert g['source_line']==count
            r = db.execute('SELECT m.root,rec.raw FROM membership m JOIN records rec ON rec.n=m.n WHERE m.n=?',(count,)).fetchone()
            assert r and r[0] == g['representative_line']
            why = [x[0] for x in db.execute('SELECT reason FROM group_risks WHERE root=? ORDER BY reason',(r[0],))]
            assert g['review_reasons'] == why and g['status'] == ('quarantine' if why else 'candidate')
            sizes[r[0]] += 1; reasons.update(why)
            if why: assert quar.readline() == r[1].rstrip(b'\r\n')+b'\n'; quarantined += 1
            elif count == r[0]: assert cand.readline() == r[1].rstrip(b'\r\n')+b'\n'; candidate_count += 1
        assert not cand.read(1) and not quar.read(1)
    assert count == size and candidate_count == summary['counts']['candidates'] and quarantined == summary['counts']['quarantined_records']
    bucket_sizes = [r[0] for r in db.execute('SELECT COUNT(*) FROM buckets GROUP BY band,sig')]
    def quant(values):
        v=sorted(values)
        return {str(q): v[int((len(v)-1)*q)] if v else None for q in (0.5,0.9,0.99,1.0)}
    distinct_reasons = dict(db.execute('SELECT reason,COUNT(*) FROM group_risks GROUP BY reason'))
    dump(d/'verification.json', dict(status='pass', records=count, groups=len(sizes), candidate_records=candidate_count,
        quarantine_records=quarantined, risk_records=dict(reasons), risk_groups=distinct_reasons,
        group_size_quantiles=quant(sizes.values()), bucket_size_quantiles=quant(bucket_sizes),
        bucket_count=len(bucket_sizes), verified_pairs=db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0],
        unchecked_buckets=summary['counts']['unchecked_buckets']))
    db.close()


def reference(out):
    d=out/'500'; db=sqlite3.connect(f'file:{(d/"result/index.sqlite3").as_posix()}?mode=ro', uri=True)
    db.execute('PRAGMA cache_size=-4096')
    positive = verified = missed = isolated = 0; checked = 0
    with (d/'reference-positive-pairs.jsonl').open('x', encoding='utf-8') as details:
        for start in range(1,501,20):
            left = [(n,p.grams(t)) for n,t in db.execute('SELECT n,text FROM records WHERE n>=? AND n<? ORDER BY n',(start,start+20))]
            for other in range(start,501,20):
                right = left if other == start else [(n,p.grams(t)) for n,t in db.execute('SELECT n,text FROM records WHERE n>=? AND n<? ORDER BY n',(other,other+20))]
                for a,ga in left:
                    for b,gb in right:
                        if a>=b: continue
                        checked += 1
                        if ga and gb:
                            inter=len(ga & gb); union=len(ga)+len(gb)-inter
                            if inter/union >= .8:
                                positive += 1
                                found = db.execute('SELECT 1 FROM pairs WHERE a=? AND b=?',(a,b)).fetchone() is not None
                                risks = db.execute('SELECT COUNT(*) FROM membership m JOIN group_risks g ON m.root=g.root WHERE m.n IN (?,?)',(a,b)).fetchone()[0]
                                verified += found; missed += not found; isolated += bool(risks)
                                details.write(json.dumps(dict(a=a,b=b,intersection=inter,union=union,verified=found,isolated=bool(risks)))+'\n')
                guard_self()
    assert checked == 500*499//2
    dump(d/'reference.json', dict(status='pass', pairs_checked=checked, jaccard_positive=positive,
        verified_positive=verified, missed_positive=missed, positive_with_quarantine=isolated,
        note='pairs table records LSH-verified pairs; exact duplicate suppression may omit pairs'))
    db.close()


def sysmem():
    class S(ctypes.Structure):
        _fields_=[('length',wintypes.DWORD),('load',wintypes.DWORD)]+[(x,ctypes.c_ulonglong) for x in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
    s=S();s.length=ctypes.sizeof(s)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)): raise OSError('memory status')
    return s.avail_phys,s.avail_page


def supervise(stage, out, args, timeout=900):
    phys, commit=sysmem()
    if phys < 1536*1048576 or commit < 1536*1048576 or min(shutil.disk_usage('C:/').free, shutil.disk_usage(out).free) < 2*1024**3:
        raise RuntimeError('preflight resource guard')
    cmd=[getattr(sys,'_base_executable',sys.executable),'-B',str(Path(__file__).resolve()),'--worker',stage,'--out',str(out)]+args
    started=time.monotonic(); minimum_phys=phys; minimum_commit=commit; peak_working=peak_commit=0; reason=None; max_disk=0; min_disk_c=shutil.disk_usage('C:/').free; min_disk_d=shutil.disk_usage(out).free
    with (out/(stage+'.stdout')).open('x',encoding='utf-8') as stdout, (out/(stage+'.stderr')).open('x',encoding='utf-8') as stderr:
        child=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(x,ctypes.c_size_t) for x in ('peak_working','working','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
        ctypes.windll.kernel32.OpenProcess.restype=wintypes.HANDLE
        ctypes.windll.kernel32.CloseHandle.argtypes=[wintypes.HANDLE]
        h=ctypes.windll.kernel32.OpenProcess(0x1010,False,child.pid)
        try:
            if not h: raise OSError('cannot monitor child')
            while child.poll() is None:
                phys,commit=sysmem(); minimum_phys=min(minimum_phys,phys);minimum_commit=min(minimum_commit,commit)
                c=Counters();c.cb=ctypes.sizeof(c)
                ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
                if not ctypes.windll.psapi.GetProcessMemoryInfo(h,ctypes.byref(c),c.cb): reason='process monitoring failed';break
                peak_working=max(peak_working,c.peak_working);peak_commit=max(peak_commit,c.peak_pagefile)
                min_disk_c=min(min_disk_c,shutil.disk_usage('C:/').free); min_disk_d=min(min_disk_d,shutil.disk_usage(out).free)
                disk=sum(f.stat().st_size for f in out.rglob('*') if f.is_file());max_disk=max(max_disk,disk)
                if phys < 1536*1048576 or commit < 1536*1048576 or c.pagefile > 192*1048576: reason='memory guard';break
                if min_disk_c < 1024**3 or min_disk_d < 1024**3 or disk > 512*1048576: reason='disk guard';break
                if time.monotonic()-started>timeout: reason='time guard';break
                time.sleep(.5)
            if reason and child.poll() is None: child.terminate()
            child.wait()
        finally:
            if child.poll() is None: child.terminate();child.wait()
            if h: ctypes.windll.kernel32.CloseHandle(h)
    m=dict(stage=stage,exit_code=child.returncode,stop_reason=reason,wall_seconds=time.monotonic()-started,
        peak_working_set=peak_working,peak_commit=peak_commit,min_available_physical=minimum_phys,
        min_available_commit=minimum_commit,peak_pilot_directory_bytes=max_disk,min_free_c=min_disk_c,min_free_d=min_disk_d)
    dump(out/(stage+'-measurement.json'),m)
    print(json.dumps(m),flush=True)
    if child.returncode or reason: raise RuntimeError(f'{stage} failed; partial output unusable')


def main():
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);a.add_argument('--worker');a.add_argument('--size',type=int)
    args=a.parse_args();out=args.out.resolve()
    if not out.is_relative_to(ROOT/'runs') or out == ROOT/'runs': raise ValueError('output must be under runs')
    if args.worker:
        if args.worker=='sample': sample(out)
        elif args.worker.startswith('run'):
            size=args.size
            p.run(out/str(size)/'sample.jsonl',out/str(size)/'result',max_memory_mib=192,min_available_mib=1536,max_disk_mib=512)
        elif args.worker.startswith('verify'): verify(out,args.size)
        elif args.worker=='reference': reference(out)
        elif args.worker=='finalhash':
            sampling=load(out/'sampling.json'); assert identity(SOURCE)==sampling['source_identity'] and sha(SOURCE)==sampling['source_sha256'] and identity(SOURCE)==sampling['source_identity']
            dump(out/'finalhash.json',dict(status='pass',sha256=sampling['source_sha256'],identity=sampling['source_identity']))
        else: raise ValueError('unknown stage')
        return
    out.mkdir(parents=False,exist_ok=False)
    dump(out/'benchmark-check.json',check_benchmark())
    supervise('sample',out,[])
    supervise('run500',out,['--size','500'])
    supervise('verify500',out,['--size','500'])
    supervise('reference',out,[],timeout=900)
    supervise('run2000',out,['--size','2000'])
    supervise('verify2000',out,['--size','2000'])
    supervise('finalhash',out,[])
    dump(out/'completed.json',dict(status='completed_candidate_only',trained=False))


if __name__=='__main__': main()
