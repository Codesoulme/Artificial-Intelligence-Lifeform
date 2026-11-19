"""Drive a model's finite awake module while a chat waits for input."""

import hashlib
import queue
import threading
import time
from contextlib import contextmanager

import torch

from .awake import AwakeConfig, AwakeCore, attach_awake
from .awake_session import AwakeSession
from .language_training import atomic_save
from .working_checkpoint_lock import lock_working_checkpoint


FORMAT = 'spora-awake-chat-v1'
CONTROLS = {'/relax': 'relax', '/focus': 'focus', '/sleep': 'sleep', '/wake': 'relax'}


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
            from io import BytesIO
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
            saved_at = time.monotonic()
            while True:
                try:
                    line = inbox.get(timeout=core.interval())
                except queue.Empty:
                    line = ''
                session.tick()
                if line is None:
                    return
                if isinstance(line, Exception):
                    raise line
                command = line.strip()
                if command in CONTROLS:
                    session.set_mode(CONTROLS[command])
                if command == '/status' or command in CONTROLS:
                    report(core.status())
                elif command:
                    yield line
                now = time.monotonic()
                if now - saved_at >= 5:
                    save()
                    saved_at = now

        threading.Thread(target=read_input, daemon=True).start()
        try:
            yield active_lines()
        finally:
            stop.set()
            save()
