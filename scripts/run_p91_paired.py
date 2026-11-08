"""P91 paired training entry. A new output directory is required per seed."""
import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import os
import platform
import shutil
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
from tokenizers import Tokenizer
import spora.source_review as source_review
import spora.paired_source_gate as source_gate

from spora.conversation import ConversationConfig
from spora.document_stream import DocumentStream
from spora.growing import load, query
from spora.language_training import frozen_digest
from spora.paired_experiment import PairedTrainer, coverage
from spora.paired_source_gate import verify_p91_inputs
from spora.source_review import file_sha256


class HashMeter:
    """Count logical bytes returned by reads, including repeated hashes."""
    def __init__(self):
        self.files = 0; self.bytes = 0; self.seconds = 0.0

    def hash(self, path):
        started = time.perf_counter(); count = 0
        try:
            digest = hashlib.sha256()
            with Path(path).open('rb') as stream:
                while chunk := stream.read(1024 * 1024):
                    count += len(chunk); digest.update(chunk)
            return digest.hexdigest()
        finally:
            self.files += 1; self.bytes += count; self.seconds += time.perf_counter()-started

    def snapshot(self):
        return dict(files=self.files, bytes=self.bytes, seconds=self.seconds)


def file_metadata(paths):
    result = {}
    for path in paths:
        try:
            s = Path(path).stat()
            result[str(path)] = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        except OSError:
            result[str(path)] = None
    return result


def append_cost(path, event):
    with path.open('a', encoding='utf8') as stream:
        stream.write(json.dumps(event, ensure_ascii=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())


def protected_files(root, output):
    """Fingerprint existing individual and historical model artifacts."""
    paths = list((root/'checkpoints').rglob('*.pt')) + list((root/'checkpoints').rglob('*.sqlite'))
    paths += [p for p in (root/'runs').rglob('*.pt') if output not in p.resolve().parents]
    paths = [p.resolve() for p in sorted(set(paths)) if p.is_file()]
    before = file_metadata(paths)
    result = {str(p): file_sha256(p) for p in paths}
    if before != file_metadata(paths):
        raise ValueError('Protected files changed during preflight hashing')
    return result


def verify_files(files):
    before = file_metadata(files)
    failures = []
    for name, expected in files.items():
        try:
            if file_sha256(name) != expected: failures.append(name)
        except OSError:
            failures.append(name)
    if failures:
        raise ValueError('Protected files changed: '+', '.join(failures))
    if before != file_metadata(files):
        raise ValueError('Files changed while hashing')


def output_path(root, value, resume):
    runs = (root/'runs').resolve()
    path = Path(value)
    path = (root/path).resolve() if not path.is_absolute() else path.resolve()
    if path.parent != runs or path == runs:
        raise ValueError('Output must be a direct new child of runs')
    if resume:
        if not path.is_dir() or not (path/'checkpoint.pt').is_file():
            raise ValueError('Resume requires an existing experiment checkpoint')
    else:
        path.mkdir(exist_ok=False)
    return path


@contextmanager
def directory_lock(directory):
    """Nonblocking OS lock held for the complete run or verified read."""
    lock = (Path(directory)/'.p91.lock').open('a+b')
    acquired = False
    try:
        if os.name == 'nt':
            import msvcrt
            if lock.seek(0, os.SEEK_END) == 0:
                lock.write(b'\0'); lock.flush()
            lock.seek(0)
            try: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc: raise RuntimeError('Experiment directory is locked') from exc
        else:
            import fcntl
            try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc: raise RuntimeError('Experiment directory is locked') from exc
        acquired = True
        yield
    finally:
        if acquired:
            if os.name == 'nt':
                lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def read_verified_summary(directory):
    """Read only a published summary matching the current checkpoint."""
    directory = Path(directory)
    with directory_lock(directory):
        result = json.loads((directory/'summary.json').read_text(encoding='utf8'))
        if result.get('format') != 'spora-p91-summary-v1' or result.get('status') not in ('partial', 'complete'):
            raise ValueError('Summary is unavailable')
        if result.get('checkpoint_sha256') != file_sha256(directory/'checkpoint.pt'):
            raise ValueError('Summary does not match current checkpoint')
        envelope = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=True)
        payload_bytes = envelope['payload']
        if hashlib.sha256(payload_bytes).hexdigest() != envelope['sha256']:
            raise ValueError('Checkpoint integrity failure')
        payload = torch.load(io.BytesIO(payload_bytes), map_location='cpu', weights_only=True)
        expected = {'steps':payload['steps'], 'targets':payload['targets'], 'seed':payload['spec']['seed'],
                    'coverage':coverage(payload['spans']), 'batches':payload['trace'],
                    'updates':payload['logs'], 'evaluations':payload['evaluations'],
                    'status':'complete' if payload['steps'] == payload['spec']['schedule_steps'] else 'partial'}
        for key, value in expected.items():
            if json.dumps(result.get(key), sort_keys=True) != json.dumps(value, sort_keys=True):
                raise ValueError('Summary content differs from checkpoint: '+key)
        return result


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    try:
        with tmp.open('w', encoding='utf8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()


def retention_probe(parent):
    model, state, _ = load(parent, 'cpu')
    before = frozen_digest(model, state)
    with torch.no_grad():
        answers = query(model, state, torch.tensor([12]*4), torch.tensor([0,1,1,1]),
                        torch.tensor([0,0,1,2])).argmax(-1).tolist()
    if before != frozen_digest(model, state) or answers != [7,5,9,3]:
        raise ValueError('Individual retention probe failed')
    return {'actual': answers, 'expected': [7,5,9,3], 'unchanged': True}


def run_pair(trainer, development, checkpoint, until, check, *, eval_steps,
             full_check=None, interval=10, on_commit=None, segment_id=None):
    """Only commit complete paired updates; recover by reloading the last file."""
    full_check = full_check or check
    def commit():
        started = time.perf_counter()
        trainer.save(checkpoint)
        if on_commit is not None:
            on_commit(trainer.steps, time.perf_counter()-started, checkpoint.stat().st_size)
    if not checkpoint.exists():
        check(); started = time.perf_counter()
        baseline = trainer.evaluate(development)
        baseline['seconds'] = time.perf_counter()-started
        trainer.evaluations.append(baseline)
        check(); commit()
    if until < trainer.steps or until > trainer.spec['schedule_steps']:
        raise ValueError('Invalid stopping step')
    while trainer.steps < until:
        check()
        trainer.step()
        check()
        if trainer.steps in eval_steps:
            full_check(); check()
            started = time.perf_counter()
            report = trainer.evaluate(development)
            report['seconds'] = time.perf_counter()-started
            trainer.evaluations.append(report)
        if next(trainer.models[0].parameters()).is_cuda:
            trainer.logs[-1]['process_segment_cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            trainer.logs[-1]['process_segment_cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        trainer.logs[-1]['process_segment_id'] = segment_id
        check()
        if trainer.steps % interval == 0 or trainer.steps in eval_steps or trainer.steps == until:
            commit()
    check()


def _main(meter):
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True); parser.add_argument('--review', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--until', type=int, required=True); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan_path = (ROOT/args.plan).resolve(); review_path = (ROOT/args.review).resolve()
    source_dir = ROOT/source_gate.DATA
    source_paths = [plan_path, review_path, ROOT/source_gate.P54, ROOT/source_gate.P28,
                    source_dir/'manifest.json', source_dir/'lineage-audit.json',
                    source_dir/'pretrain-train.bin', source_dir/'story-development.bin',
                    source_dir/'tokenizer.json']
    source_before = file_metadata(source_paths)
    provenance = verify_p91_inputs(ROOT, plan_path, review_path)
    if source_before != file_metadata(source_paths):
        raise ValueError('Source files changed during preflight hashing')
    plan = json.loads(plan_path.read_text(encoding='utf8'))
    if args.seed not in plan['seeds'] or not 0 <= args.until <= plan['schedule_steps']:
        raise ValueError('Seed or stopping step not covered by reviewed plan')
    if plan['device'] not in ('cpu', 'cuda') or (plan['device'] == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Reviewed device unavailable')
    parent = (ROOT/plan['parent']).resolve()
    if file_sha256(parent) != plan['parent_sha256']:
        raise ValueError('Parent individual changed')
    # Resolve and validate before making any output; resume never creates a directory.
    out = output_path(ROOT, args.output, args.resume)
    with directory_lock(out):
        if parent == out or out in parent.parents:
            raise ValueError('Parent must be outside output directory')
        code = [*sorted((ROOT/'spora').glob('*.py')), Path(__file__).resolve()]
        code_before = file_metadata(code)
        fingerprints = {str(p.resolve()): file_sha256(p) for p in code}
        if code_before != file_metadata(code):
            raise ValueError('Implementation changed during preflight hashing')
        metadata_path = out/'metadata.json'
        if args.resume:
            metadata = json.loads(metadata_path.read_text(encoding='utf8'))
            if fingerprints != metadata['code'] or provenance != metadata['source_review'] or metadata['seed'] != args.seed:
                raise ValueError('Source, implementation or seed changed since start')
            protected = metadata['protected']; verify_files(protected)
            if metadata['torch'] != str(torch.__version__) or metadata['cuda'] != torch.version.cuda or metadata['device'] != plan['device'] or metadata['device_name'] != (torch.cuda.get_device_name() if plan['device'] == 'cuda' else platform.processor()):
                raise ValueError('Runtime device or framework changed')
        else:
            protected = protected_files(ROOT, out)
            if str(parent) not in protected or protected[str(parent)] != plan['parent_sha256']:
                raise ValueError('Parent not included in protected files')
            device_name = torch.cuda.get_device_name() if plan['device'] == 'cuda' else platform.processor()
            metadata = dict(format='spora-p91-run-v1', source_review=provenance, protected=protected,
                            code=fingerprints, seed=args.seed, python=sys.version, torch=str(torch.__version__),
                            numpy=str(np.__version__), platform=platform.platform(), device=plan['device'],
                            device_name=device_name, cuda=torch.version.cuda,
                            disk_free_bytes=shutil.disk_usage(out).free,
                            preflight_seconds=time.perf_counter()-started)
            atomic_json(metadata_path, metadata)
        protected_before = file_metadata(protected)
        data = ROOT/plan['data']
        tokenizer = Tokenizer.from_file(str(data/'tokenizer.json'))
        bos, eos, pad = (tokenizer.token_to_id(s) for s in ('<|bos|>', '<|eos|>', '<|pad|>'))
        if None in (bos, eos, pad) or len({bos,eos,pad}) != 3 or tokenizer.get_vocab_size() != plan['model']['vocabulary']:
            raise ValueError('Tokenizer symbols or vocabulary mismatch')
        def stream(name):
            return DocumentStream(np.memmap(data/name, dtype='<u2', mode='r'), bos, eos, pad)
        train, dev = stream('pretrain-train.bin'), stream('story-development.bin')
        if max(int(train.stream.max()), int(dev.stream.max())) >= plan['model']['vocabulary']:
            raise ValueError('Token outside reviewed vocabulary')
        probe = retention_probe(parent)
        organism, state, _ = load(parent, plan['device'])
        config = ConversationConfig(**{**plan['model'], 'memory_layers': tuple(plan['model']['memory_layers'])})
        torch.manual_seed(args.seed)
        provenance = dict(source=provenance, code=fingerprints, protected=protected,
                          parent=plan['parent_sha256'], metadata_sha256=file_sha256(metadata_path))
        trainer = PairedTrainer(organism, state, config, train, seed=args.seed,
            data_seed=plan['data_seeds'][str(args.seed)], provenance=provenance,
            batch_size=plan['microbatch'], length=plan['length'], budget=plan['target_budget'],
            schedule_steps=plan['schedule_steps'], warmup=plan['warmup'], lr=plan['learning_rate'], device=plan['device'])
        checkpoint = out/'checkpoint.pt'
        if args.resume:
            trainer.restore(checkpoint)
        watched = set(protected) | set(fingerprints) | {str(plan_path), str(review_path), str(metadata_path),
            str(ROOT/source_gate.P54), str(ROOT/source_gate.P28), str(data/'manifest.json'),
            str(data/'lineage-audit.json'), str(data/'pretrain-train.bin'),
            str(data/'story-development.bin'), str(data/'tokenizer.json')}
        watched_state = file_metadata(watched)
        if any(value is None for value in watched_state.values()):
            raise ValueError('Watched source or protected file missing')
        if source_before != file_metadata(source_paths) or code_before != file_metadata(code) or protected_before != file_metadata(protected):
            raise ValueError('Inputs changed during preflight')
        cost_path = out/'io-cost.jsonl'
        anomaly_checked = False
        append_cost(cost_path, dict(event='preflight', hashes=meter.snapshot(),
            protected_files=len(protected), protected_bytes=sum(Path(p).stat().st_size for p in protected),
            watched_files=len(watched), scope='logical bytes read for SHA-256; may include repeat reads'))

        def full_check(reason='checkpoint'):
            check_started = time.perf_counter()
            before = meter.snapshot(); failures = []
            if reason == 'metadata_change':
                # An anomaly gets a complete scan, even when an early comparison fails.
                for name in sorted(watched):
                    try: file_sha256(name)
                    except OSError as exc: failures.append(str(exc))
            for action in (trainer.check_frozen, lambda: verify_files(protected),
                           lambda: verify_files(fingerprints)):
                try: action()
                except (ValueError, OSError) as exc: failures.append(str(exc))
            try:
                if verify_p91_inputs(ROOT, plan_path, review_path) != metadata['source_review']:
                    failures.append('Source binding changed')
            except (ValueError, OSError, KeyError) as exc: failures.append(str(exc))
            try:
                if file_sha256(metadata_path) != provenance['metadata_sha256']:
                    failures.append('Run metadata changed')
            except OSError as exc: failures.append(str(exc))
            elapsed = time.perf_counter()-check_started
            trainer.guard_seconds += elapsed
            after = meter.snapshot()
            append_cost(cost_path, dict(event='full_check', reason=reason, step=trainer.steps,
                hashes={k: after[k]-before[k] for k in before}, seconds=elapsed, passed=not failures))
            if failures: raise ValueError('; '.join(failures))

        def check():
            nonlocal anomaly_checked
            check_started = time.perf_counter()
            current = file_metadata(watched)
            trainer.guard_seconds += time.perf_counter()-check_started
            if current != watched_state:
                anomaly_checked = True
                try:
                    full_check('metadata_change')
                finally:
                    raise ValueError('Watched file metadata changed; run stopped after full verification')

        def on_commit(step, seconds, size):
            append_cost(cost_path, dict(event='checkpoint', step=step, seconds=seconds,
                                       logical_bytes_written=size))
        if args.until < trainer.steps:
            raise ValueError('Invalid stopping step')
        # Invalidate before any update or checkpoint replacement. A crash leaves
        # a tombstone, never an apparently current previous summary.
        atomic_json(out/'summary.json', {'format':'spora-p91-summary-v1', 'status':'unavailable'})
        segment_id = uuid.uuid4().hex
        segment_start_step = trainer.steps
        if plan['device'] == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        run_pair(trainer, dev, checkpoint, args.until, check,
                 full_check=lambda: full_check('before_evaluation'),
                 eval_steps=set(plan['evaluation_steps']), interval=10, on_commit=on_commit,
                 segment_id=segment_id)
        final_probe = retention_probe(parent)
        check()
        if trainer.steps == plan['schedule_steps']:
            full_check('completion'); check()
        segment_peak = dict(id=segment_id, start_step=segment_start_step, end_step=trainer.steps,
            scope='current process segment since reset, including evaluations; not per-arm or across resumes',
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if plan['device'] == 'cuda' else None,
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if plan['device'] == 'cuda' else None)
        summary = dict(format='spora-p91-summary-v1', status='complete' if trainer.steps == plan['schedule_steps'] else 'partial',
            seed=args.seed, steps=trainer.steps, targets=trainer.targets, coverage=coverage(trainer.spans),
            batches=trainer.trace, updates=trainer.logs, evaluations=trainer.evaluations,
            retention_probe_before=probe, retention_probe_after=final_probe,
            parameters=[sum(p.numel() for p in m.trainable_parameters()) for m in trainer.models],
            checkpoint_sha256=file_sha256(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
            invocation_seconds=time.perf_counter()-started, guard_seconds=trainer.guard_seconds,
            invocation_hashes=meter.snapshot(), io_cost_log='io-cost.jsonl',
            process_segment_memory=segment_peak,
            disk_free_bytes=shutil.disk_usage(out).free)
        atomic_json(out/'summary.json', summary)
        print(json.dumps({'output':str(out), 'steps':trainer.steps, 'targets':trainer.targets}))


def main():
    global file_sha256
    meter = HashMeter()
    previous = file_sha256, source_review.file_sha256, source_gate.file_sha256
    file_sha256 = source_review.file_sha256 = source_gate.file_sha256 = meter.hash
    try:
        return _main(meter)
    finally:
        file_sha256, source_review.file_sha256, source_gate.file_sha256 = previous


if __name__ == '__main__':
    main()
