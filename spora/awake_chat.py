"""Drive a model's finite awake module independently of chat decoding."""

import hashlib
import queue
import threading
from contextlib import contextmanager
from io import BytesIO

import torch

from .awake import AwakeConfig, AwakeCore, attach_awake
from .awake_session import AwakeSession
from .language_training import atomic_save
from .working_checkpoint_lock import lock_working_checkpoint


FORMAT = 'spora-awake-chat-v1'
CONTROLS = {'/relax': 'relax', '/focus': 'focus', '/sleep': 'sleep', '/wake': 'relax'}
SAVE_INTERVAL_NS = 5_000_000_000


@contextmanager
def awake_chat_lines(model, lines, path, trained_core, individual, report):
    core_data = trained_core.read_bytes()
    core_digest = hashlib.sha256(core_data).hexdigest()
    provenance = dict(format=FORMAT, individual=individual, core_sha256=core_digest)
    with lock_working_checkpoint(path):
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if saved['provenance'] != provenance:
                raise ValueError('Awake state belongs to a different chat individual or core')
            payload = saved['core']
        else:
            payload = torch.load(BytesIO(core_data), map_location='cpu', weights_only=True)
        core = AwakeCore(AwakeConfig(**payload['config']))
        core.load_state_dict(payload['state'])
        attach_awake(model, core.requires_grad_(False))
        session = AwakeSession(model)

        def save():
            core.last_utc_ns.fill_(session.utc())
            atomic_save(dict(provenance=provenance, core=dict(config=core.config_dict(), state=core.state_dict())), path)

        inbox = queue.Queue(maxsize=16)
        stop = threading.Event()
        state_changed = threading.Condition()
        failures = []

        def run_clock():
            with state_changed:
                next_save = session.clock() + SAVE_INTERVAL_NS
                try:
                    while not stop.is_set():
                        next_tick = session.anchor + int(core.interval() * 1e9)
                        now = session.clock()
                        delay = min(next_tick, next_save) - now
                        if delay > 0:
                            state_changed.wait(delay / 1e9)
                            continue
                        if now >= next_tick:
                            session.tick()
                        if now >= next_save:
                            save()
                            next_save = session.clock() + SAVE_INTERVAL_NS
                    session.tick()
                    save()
                except Exception as error:
                    failures.append(error)
                    stop.set()

        def enqueue(item):
            while not stop.is_set():
                try:
                    inbox.put(item, timeout=.25)
                    return
                except queue.Full:
                    continue

        def read_input():
            try:
                for line in lines:
                    if stop.is_set():
                        return
                    enqueue(line)
            except Exception as error:
                enqueue(error)
            finally:
                enqueue(None)

        def active_lines():
            while True:
                if stop.is_set():
                    return
                try:
                    line = inbox.get(timeout=.25)
                except queue.Empty:
                    continue
                if line is None:
                    return
                if isinstance(line, Exception):
                    raise line
                command = line.strip()
                if command == '/status' or command in CONTROLS:
                    with state_changed:
                        mode = CONTROLS.get(command)
                        status = core.status()
                        if mode is not None and status['mode'] != mode:
                            session.tick()
                            session.set_mode(mode)
                            state_changed.notify()
                            status = core.status()
                    report(status)
                elif command:
                    yield line

        worker = threading.Thread(target=run_clock, name='spora-awake-clock')
        threading.Thread(target=read_input, daemon=True).start()
        worker.start()
        try:
            yield active_lines()
        finally:
            stop.set()
            with state_changed:
                state_changed.notify()
            worker.join()
        if failures:
            raise failures[0]
