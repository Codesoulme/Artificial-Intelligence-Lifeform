from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import unicodedata

RUNS = (Path(__file__).resolve().parents[1] / 'runs').resolve()
PRIME = (1 << 61) - 1
COEFF = [(int.from_bytes(hashlib.sha256(f'p114-a-{i}'.encode()).digest()[:8], 'big') % (PRIME-1)+1,
          int.from_bytes(hashlib.sha256(f'p114-b-{i}'.encode()).digest()[:8], 'big') % PRIME) for i in range(32)]


def digest(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def norm(s):
    return ''.join(unicodedata.normalize('NFKC', s).casefold().split())


def grams(s):
    s = unicodedata.normalize('NFKC', s)
    s = ''.join(c for c in s if '\u3400' <= c <= '\u4dbf' or '\u4e00' <= c <= '\u9fff'
                or '\U00020000' <= c <= '\U0002fa1f' or '\U00030000' <= c <= '\U000323af')
    return {s[i:i+5] for i in range(len(s)-4)}


def signature(gs):
    mins = [PRIME] * 32
    for g in gs:
        h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8, person=b'P114gram').digest(), 'big') % PRIME
        for i, (a, b) in enumerate(COEFF):
            mins[i] = min(mins[i], (a*h+b) % PRIME)
    return tuple(mins)


def memory_info():
    if os.name != 'nt':
        raise RuntimeError('resource guard currently requires Windows')
    import ctypes
    from ctypes import wintypes
    class Status(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [(x, ctypes.c_ulonglong) for x in
            ('total_phys', 'avail_phys', 'total_page', 'avail_page', 'total_virtual', 'avail_virtual', 'avail_extended')]
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [(x, ctypes.c_size_t) for x in
            ('peak_working', 'working', 'peak_paged', 'paged', 'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]
    status, counters = Status(), Counters()
    status.length, counters.cb = ctypes.sizeof(status), ctypes.sizeof(counters)
    ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) or not ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError('Windows memory measurement failed')
    return {'available_physical': status.avail_phys, 'working_set': counters.working,
            'peak_working_set': counters.peak_working, 'commit': counters.pagefile, 'peak_commit': counters.peak_pagefile}


def write_json(path, obj):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write('\n')


def root(db, n):
    while True:
        parent = db.execute('SELECT parent FROM records WHERE n=?', (n,)).fetchone()[0]
        if parent == n:
            return n
        n = parent


def union(db, a, b):
    a, b = root(db, a), root(db, b)
    if a != b:
        db.execute('UPDATE records SET parent=? WHERE n=?', (min(a, b), max(a, b)))


def run(input_path, output_path, threshold=.8, max_bucket=200, max_pairs=1000000,
        max_disk_mib=2048, max_memory_mib=256, min_available_mib=1024, max_line_bytes=131072, cache_mib=16):
    if not (0 < threshold <= 1 and max_bucket >= 2 and max_pairs >= 0 and max_disk_mib >= 8
            and max_memory_mib >= 32 and min_available_mib >= 0 and 256 <= max_line_bytes <= 1048576
            and 1 <= cache_mib <= max_memory_mib // 4):
        raise ValueError('invalid limits')
    source = Path(input_path).resolve(strict=True)
    out = Path(output_path).resolve()
    if not out.is_relative_to(RUNS) or out == RUNS or source.is_relative_to(out):
        raise ValueError('output must be a new directory below runs and exclude input')
    out.mkdir(parents=False, exist_ok=False)
    params = dict(threshold=threshold, max_bucket=max_bucket, max_pairs=max_pairs, max_disk_mib=max_disk_mib,
                  max_memory_mib=max_memory_mib, min_available_mib=min_available_mib,
                  max_line_bytes=max_line_bytes, cache_mib=cache_mib)
    stage = 'initializing'
    n = 0
    db = None
    peak_disk = 0
    min_available = None
    def guard():
        nonlocal peak_disk, min_available
        m = memory_info()
        min_available = min(m['available_physical'], min_available or m['available_physical'])
        disk = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        peak_disk = max(peak_disk, disk)
        if m['commit'] > max_memory_mib*1048576 or m['available_physical'] < min_available_mib*1048576:
            raise RuntimeError(f'memory guard: {m}')
        if disk > max_disk_mib*1048576 or shutil.disk_usage(out).free < 16*1048576:
            raise RuntimeError(f'disk guard: output bytes={disk}')
    try:
        guard()
        db = sqlite3.connect(out/'index.sqlite3')
        db.execute('PRAGMA journal_mode=DELETE')
        db.execute('PRAGMA temp_store=FILE')
        db.execute(f'PRAGMA cache_size=-{cache_mib*1024}')
        db.execute(f'PRAGMA max_page_count={max_disk_mib*1048576//4096}')
        db.executescript('''
          CREATE TABLE records(n INTEGER PRIMARY KEY, raw BLOB NOT NULL, text TEXT NOT NULL, sha TEXT NOT NULL, prefix TEXT NOT NULL, parent INTEGER NOT NULL);
          CREATE INDEX exact_idx ON records(sha);
          CREATE INDEX prefix_idx ON records(prefix);
          CREATE TABLE buckets(band INTEGER, sig BLOB, n INTEGER, PRIMARY KEY(band,sig,n)) WITHOUT ROWID;
          CREATE TABLE pairs(a INTEGER, b INTEGER, PRIMARY KEY(a,b)) WITHOUT ROWID;
          CREATE TABLE risks(n INTEGER, reason TEXT, PRIMARY KEY(n,reason)) WITHOUT ROWID;
          CREATE TABLE membership(n INTEGER PRIMARY KEY, root INTEGER NOT NULL);
          CREATE INDEX membership_root ON membership(root,n);
        ''')
        stage = 'ingest'
        raw_hash = hashlib.sha256()
        with source.open('rb') as f:
            while True:
                raw = f.readline(max_line_bytes+1)
                if not raw:
                    break
                n += 1
                if len(raw) > max_line_bytes:
                    raise ValueError(f'line {n}: exceeds max_line_bytes')
                raw_hash.update(raw)
                row = json.loads(raw)
                if not isinstance(row, dict) or any(not isinstance(row.get(k), str) for k in ('text','group','sha256')):
                    raise ValueError(f'line {n}: expected string text/group/sha256')
                text = row['text']
                raw_sha = digest(text)
                if 'source_original_sha256' in row:
                    key = re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).lower()
                    if row['source_original_sha256'] != raw_sha or row['sha256'] != digest(key):
                        raise ValueError(f'line {n}: P112 sha256 mismatch')
                elif row['sha256'] != raw_sha:
                    raise ValueError(f'line {n}: sha256 mismatch')
                prefix = norm(text)[:96]
                exact = db.execute('SELECT n FROM records WHERE sha=? AND text=? ORDER BY n LIMIT 1', (raw_sha, text)).fetchone()
                pref = db.execute('SELECT n FROM records WHERE prefix=? ORDER BY n LIMIT 1', (prefix,)).fetchone()
                db.execute('INSERT INTO records VALUES(?,?,?,?,?,?)', (n, raw, text, raw_sha, prefix, n))
                if exact:
                    union(db, n, exact[0])
                if pref:
                    union(db, n, pref[0])
                if not exact:
                    gs = grams(text)
                    if not gs:
                        db.execute('INSERT OR IGNORE INTO risks VALUES(?,?)', (n, 'fewer_than_five_chinese_characters'))
                    else:
                        sig = signature(gs)
                        for band in range(8):
                            bs = b''.join(x.to_bytes(8, 'big') for x in sig[4*band:4*band+4])
                            db.execute('INSERT INTO buckets VALUES(?,?,?)', (band, bs, n))
                if n % 100 == 0:
                    db.commit(); guard()
        db.commit(); guard()
        stage = 'buckets'
        skipped = bucket_count = 0
        reserved = 0
        with (out/'unchecked_buckets.jsonl').open('x', encoding='utf-8', newline='\n') as unchecked:
            cursor = db.execute('SELECT band,sig,COUNT(*) FROM buckets GROUP BY band,sig HAVING COUNT(*)>1 ORDER BY band,sig')
            for band, sig, size in cursor:
                bucket_count += 1
                possible = size*(size-1)//2
                reason = 'oversized_bucket' if size > max_bucket else ('pair_budget' if reserved+possible > max_pairs else None)
                if reason:
                    skipped += 1
                    db.execute('INSERT OR IGNORE INTO risks SELECT n,? FROM buckets WHERE band=? AND sig=?', (reason, band, sig))
                    unchecked.write(json.dumps(dict(band=band, signature=sig.hex(), size=size, possible_pairs=possible, reason=reason))+'\n')
                else:
                    members = [r[0] for r in db.execute('SELECT n FROM buckets WHERE band=? AND sig=? ORDER BY n', (band, sig))]
                    db.executemany('INSERT OR IGNORE INTO pairs VALUES(?,?)', itertools.combinations(members, 2))
                    reserved += possible
                if bucket_count % 100 == 0:
                    db.commit(); guard()
        db.commit(); guard()
        stage = 'verify'
        verified = accepted = 0
        for a, b in db.execute('SELECT a,b FROM pairs ORDER BY a,b'):
            ta = db.execute('SELECT text FROM records WHERE n=?', (a,)).fetchone()[0]
            tb = db.execute('SELECT text FROM records WHERE n=?', (b,)).fetchone()[0]
            ga, gb = grams(ta), grams(tb)
            if ga and gb and len(ga & gb) / len(ga | gb) >= threshold:
                union(db, a, b); accepted += 1
            verified += 1
            if verified % 100 == 0:
                db.commit(); guard()
        db.commit(); guard()
        stage = 'membership'
        for (i,) in db.execute('SELECT n FROM records ORDER BY n'):
            db.execute('INSERT INTO membership VALUES(?,?)', (i, root(db, i)))
            if i % 100 == 0:
                db.commit(); guard()
        db.execute('CREATE TABLE group_risks(root INTEGER, reason TEXT, PRIMARY KEY(root,reason)) WITHOUT ROWID')
        db.execute('INSERT OR IGNORE INTO group_risks SELECT m.root,r.reason FROM membership m JOIN risks r ON m.n=r.n')
        db.commit(); guard()
        stage = 'output'
        kept = isolated = groups = 0
        with (out/'candidates.jsonl').open('xb') as candidates, (out/'quarantine.jsonl').open('xb') as quarantine, (out/'groups.jsonl').open('x', encoding='utf-8', newline='\n') as details:
            for i, r, raw in db.execute('SELECT m.n,m.root,r.raw FROM membership m JOIN records r ON r.n=m.n ORDER BY m.n'):
                reasons = [x[0] for x in db.execute('SELECT reason FROM group_risks WHERE root=? ORDER BY reason', (r,))]
                if i == r:
                    groups += 1
                if reasons:
                    quarantine.write(raw.rstrip(b'\r\n') + b'\n'); isolated += 1
                elif i == r:
                    candidates.write(raw.rstrip(b'\r\n') + b'\n'); kept += 1
                details.write(json.dumps(dict(source_line=i, representative_line=r, status='quarantine' if reasons else 'candidate', review_reasons=reasons), ensure_ascii=False)+'\n')
                if i % 100 == 0:
                    candidates.flush(); quarantine.flush(); details.flush(); guard()
        guard()
        hashes = {}
        for name in ('candidates.jsonl', 'quarantine.jsonl', 'groups.jsonl', 'unchecked_buckets.jsonl'):
            h = hashlib.sha256()
            with (out/name).open('rb') as stream:
                for block in iter(lambda: stream.read(1048576), b''):
                    h.update(block)
                    guard()
            hashes[name] = h.hexdigest()
        result = dict(status='candidate_not_trained', input=dict(path=str(source), sha256=raw_hash.hexdigest(), records=n),
                      parameters=params, counts=dict(groups=groups, candidates=kept, quarantined_records=isolated,
                      candidate_pairs_verified=verified, similar_pairs=accepted, unchecked_buckets=skipped, pair_budget_reserved=reserved),
                      output_sha256=hashes,
                      resources=dict(**memory_info(), min_available_physical_sampled=min_available, peak_output_bytes_sampled=peak_disk),
                      limitations=['LSH can miss similar pairs; no semantic or cross-dataset isolation is guaranteed.',
                                   'Transitive groups can contain dissimilar endpoints; high-frequency features are retained.',
                                   'Resource guards are sampled, not hard process or filesystem quotas.'])
        write_json(out/'summary.json', result)
        return result
    except Exception as exc:
        try:
            write_json(out/'failure.json', dict(status='failed_do_not_use_partial_outputs', stage=stage, records_seen=n,
                       error_type=type(exc).__name__, error=str(exc), parameters=params, peak_output_bytes_sampled=peak_disk))
        except OSError:
            pass
        raise
    finally:
        if db is not None:
            db.close()


def main():
    p = argparse.ArgumentParser(description='Disk-backed candidate-only story grouping; never trains.')
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--threshold', type=float, default=.8)
    p.add_argument('--max-bucket', type=int, default=200)
    p.add_argument('--max-pairs', type=int, default=1000000)
    p.add_argument('--max-disk-mib', type=int, default=2048)
    p.add_argument('--max-memory-mib', type=int, default=256)
    p.add_argument('--min-available-mib', type=int, default=1024)
    p.add_argument('--max-line-bytes', type=int, default=131072)
    p.add_argument('--cache-mib', type=int, default=16)
    try:
        args = vars(p.parse_args())
        args['input_path'] = args.pop('input')
        args['output_path'] = args.pop('output')
        print(json.dumps(run(**args), ensure_ascii=False, indent=2))
    except (ValueError, OSError, RuntimeError, sqlite3.Error) as exc:
        p.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    main()
