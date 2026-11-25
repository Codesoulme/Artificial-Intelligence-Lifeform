"""Exercise live CPU ticks and tensor-backed recent recall through the Windows launcher."""

import argparse
from contextlib import contextmanager
from hashlib import file_digest
import json
from pathlib import Path
from queue import Queue
import sqlite3
import subprocess
from threading import Thread
from time import sleep

from scripts.audit_spora_version_local_chat import ROOT
from scripts.train_spora_recurrent_counterfactual import store_json


RECALL = '再说一遍刚才查到的那项。'
CURRENT = '请查贺听竹的饮品当前版。'
PREVIOUS = '请查贺听竹的饮品上一版。'


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


@contextmanager
def open_chat(profile, state, recurrent):
    command = ['cmd.exe', '/d', '/c', 'spora_version_chat.cmd', '--profile', str(profile),
               '--awake-state', str(state), '--recurrent', *map(str, recurrent), '--json']
    process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               text=True, encoding='utf-8', bufsize=1)
    pending = Queue()

    def receive():
        for line in process.stdout:
            pending.put(line)
        pending.put(None)

    worker = Thread(target=receive, daemon=True)
    worker.start()

    def send(text):
        process.stdin.write(text + '\n')
        process.stdin.flush()
        line = pending.get(timeout=160)
        if line is None:
            raise RuntimeError(f'Chat stopped with status {process.poll()}')
        reply = json.loads(line)
        print(text, reply.get('kind'), reply.get('answer', ''), flush=True)
        return reply

    try:
        yield send
    finally:
        if process.poll() is None:
            process.stdin.write('/退出\n')
            process.stdin.flush()
        process.stdin.close()
        try:
            code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
        worker.join(timeout=5)
        process.stdout.close()
        if code:
            raise RuntimeError(f'Chat exited with status {code}')


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    profile, state = args.output / 'profile', args.output / 'awake.pt'
    database = profile / 'versions.sqlite3'
    rows, checks = {}, {}

    def step(label, request):
        rows[label] = send(request)
        return rows[label]

    def read(label, request, number, value, *, awake=False):
        answer = step(label, request)
        checks[label] = (answer['kind'] == 'read' and answer['selected'] == [number]
                         and answer['answer'] == '饮品：' + value + '。'
                         and (not awake or answer['reasons'] == ['awake_recall']))

    with open_chat(profile, state, args.recurrent) as send:
        first = step('first_status', '/status')['status']
        sleep(.9)
        relaxed = step('relaxed_status', '/status')['status']
        focus_start = step('focus_start', '/focus')['status']
        sleep(.9)
        focused = step('focused_status', '/status')['status']
        checks['background_cadence'] = (
            first['mode'] == relaxed['mode'] == 'relax' and focused['mode'] == 'focus'
            and relaxed['neural_steps'] > first['neural_steps']
            and focused['neural_steps'] - focus_start['neural_steps'] > relaxed['neural_steps'] - first['neural_steps']
        )
        step('relax', '/relax')
        for number, value in enumerate(('谷香茶', '温杏乳'), 1):
            result = step(f'write_{number}', f'/记 贺听竹 | 饮品 | {value}')
            checks[f'write_{number}'] = (result['kind'], result.get('record')) == ('saved', number)
        read('current', CURRENT, 2, '温杏乳')
        sleep(.75)
        active = step('active', '/status')['status']
        checks['topic_active'] = active['topics']['activity']['drinks'] >= .2
        read('recent_current', RECALL, 2, '温杏乳', awake=True)
        checks['single_use'] = step('consumed', RECALL)['kind'] == 'clarify'
        read('previous', PREVIOUS, 1, '谷香茶')
        sleep(.75)
        read('recent_previous', RECALL, 1, '谷香茶', awake=True)
        read('before_expiry', CURRENT, 2, '温杏乳')
        sleep(4.5)
        expired = step('expired_status', '/status')['status']
        checks['expired'] = (expired['topics']['activity']['drinks'] < .2
                             and step('expired', RECALL)['kind'] == 'clarify')
        read('before_sleep', CURRENT, 2, '温杏乳')
        sleep(.75)
        asleep = step('sleep', '/sleep')['status']
        sleep(.9)
        still_asleep = step('sleep_status', '/status')['status']
        checks['sleep'] = (asleep['mode'] == still_asleep['mode'] == 'sleep'
                           and still_asleep['neural_steps'] == asleep['neural_steps']
                           and still_asleep['sleep_seconds'] > asleep['sleep_seconds']
                           and not any(still_asleep['topics']['activity'].values()))
        step('wake', '/wake')
        checks['wake_clears_recent'] = step('after_wake', RECALL)['kind'] == 'clarify'
        read('before_exit', CURRENT, 2, '温杏乳')
        sleep(.75)
    before, awake_before = digest(database), digest(state)
    with open_chat(profile, state, args.recurrent) as send:
        restarted = step('restart_status', '/status')['status']
        checks['restart_clears_recent'] = (restarted['neural_steps'] >= still_asleep['neural_steps']
                                           and restarted['gap_seconds'] > 0
                                           and not any(restarted['topics']['activity'].values())
                                           and step('after_restart', RECALL)['kind'] == 'clarify')
        read('explicit_after_restart', PREVIOUS, 1, '谷香茶')
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as conn:
        checks['archive_valid'] = (conn.execute('pragma integrity_check').fetchone() == ('ok',)
                                   and conn.execute('select count(*) from versions').fetchone() == (2,))
    checks['independent_persistence'] = digest(database) == before and digest(state) != awake_before
    store_json(args.output / 'result.json', {'checks': checks, 'rows': rows, 'archive_sha256': before})
    print(checks, flush=True)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--recurrent', required=True, nargs=3, type=Path)
    main(parser.parse_args())
