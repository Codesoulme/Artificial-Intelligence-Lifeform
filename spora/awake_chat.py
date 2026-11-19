"""Drive a model's finite awake module independently of chat decoding."""

import hashlib
import queue
import threading
from contextlib import contextmanager
from io import BytesIO

import torch

from .awake import AwakeConfig, AwakeCore, attach_awake
from .awake_session import AwakeSession
from .awake_topics import AwakeTopics
from .language_training import atomic_save
from .working_checkpoint_lock import lock_working_checkpoint


FORMAT = 'spora-awake-chat-v1'
CONTROLS = {'/relax': 'relax', '/focus': 'focus', '/sleep': 'sleep', '/wake': 'relax'}
SAVE_INTERVAL_NS = 5_000_000_000


@contextmanager
def awake_chat_lines(model, lines, path, trained_core, individual, report, *, topic_labels=(), clear_topics_on_sleep=False,
                     on_sleep=None):
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
            saved = {}
            payload = torch.load(BytesIO(core_data), map_location='cpu', weights_only=True)
        core = AwakeCore(AwakeConfig(**payload['config']))
        core.load_state_dict(payload['state'])
        attach_awake(model, core.requires_grad_(False))
        session = AwakeSession(model)
        topics = AwakeTopics(topic_labels) if topic_labels else None
        if 'topics' in saved:
            if topics is None or tuple(saved['topics']['labels']) != topics.labels:
                raise ValueError('Awake state has different topic labels')
            topics.load_state_dict(saved['topics']['state'])
            if int(topics.observations) < 0:
                raise ValueError('Negative topic observation count')
            topics.invalidate()
        if topics is not None:
            model.awake_topics = topics

        def tick():
            awake_ns, gap_ns = int(core.awake_ns), int(core.gap_ns)
            session.tick()
            if topics is not None:
                if int(core.gap_ns) != gap_ns:
                    topics.invalidate()
                elif int(core.awake_ns) > awake_ns:
                    topics.advance(int(core.awake_ns) - awake_ns)

        def save():
            core.last_utc_ns.fill_(session.utc())
            payload = dict(provenance=provenance, core=dict(config=core.config_dict(), state=core.state_dict()))
            if topics is not None:
                payload['topics'] = dict(labels=topics.labels, state=topics.state_dict())
            atomic_save(payload, path)

        inbox = queue.Queue(maxsize=16)
        stop = threading.Event()
        state_changed = threading.Condition()
        failures = []

        def observe(probabilities):
            with state_changed:
                if topics is not None and int(core.mode) != 2 and not stop.is_set():
                    topics.observe(probabilities)

        def consume_topic():
            with state_changed:
                if topics is not None and int(core.mode) != 2 and not stop.is_set():
                    return topics.consume_read()
            return None

        def bind_topic(entity, field, remaining=1):
            with state_changed:
                if topics is None or int(core.mode) == 2 or stop.is_set():
                    return False
                topics.bind_read(entity, field, remaining)
                return True

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
                            tick()
                        if now >= next_save:
                            save()
                            next_save = session.clock() + SAVE_INTERVAL_NS
                    tick()
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
                    entering_sleep = False
                    with state_changed:
                        mode = CONTROLS.get(command)
                        status = core.status()
                        if mode is not None and status['mode'] != mode:
                            entering_sleep = mode == 'sleep'
                            tick()
                            if clear_topics_on_sleep and topics is not None and mode == 'sleep':
                                topics.invalidate()
                            session.set_mode(mode)
                            state_changed.notify()
                            status = core.status()
                        if topics is not None:
                            status['topics'] = topics.status()
                    if entering_sleep and on_sleep is not None:
                        status['replay'] = on_sleep()
                    report(status)
                elif command:
                    yield line

        worker = threading.Thread(target=run_clock, name='spora-awake-clock')
        threading.Thread(target=read_input, daemon=True).start()
        worker.start()
        try:
            yield active_lines(), observe, consume_topic, bind_topic
        finally:
            stop.set()
            with state_changed:
                state_changed.notify()
            worker.join()
        if failures:
            raise failures[0]
