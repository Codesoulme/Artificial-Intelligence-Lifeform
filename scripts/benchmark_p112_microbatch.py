"""Read-only P34 forward/backward microbatch benchmark; no optimizer exists."""
import hashlib
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from tokenizers import Tokenizer
from spora.conversation import restore_conversation
from spora.document_stream import trim_ignored_tail
from spora.growing import load
from spora.language_training import frozen_digest
from spora.token_objective import backward_token_budget
from spora.uniform_document_stream import UniformDocumentStream


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def split_batches(x, y, size):
    if x.shape != y.shape or x.ndim != 2 or size < 1:
        raise ValueError('Invalid batch')
    return [trim_ignored_tail(x[i:i+size], y[i:i+size])
            for i in range(0, len(x), size)]


def batch_digest(x, y):
    h = hashlib.sha256()
    for tensor in (x, y):
        h.update(tensor.contiguous().numpy().tobytes())
    return h.hexdigest()


def gradients(model):
    return {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()
            if p.requires_grad and p.grad is not None}


def compare(reference, actual):
    if reference.keys() != actual.keys():
        raise AssertionError('Gradient parameter sets differ')
    max_abs = 0.0
    squared_diff = squared_ref = 0.0
    for name in reference:
        a, b = reference[name], actual[name]
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-6, msg=lambda m: name + ': ' + m)
        d = (a - b).double()
        max_abs = max(max_abs, float(d.abs().max()))
        squared_diff += float(d.square().sum())
        squared_ref += float(a.double().square().sum())
    return {'max_abs': max_abs, 'relative_l2': (squared_diff / squared_ref) ** .5}


def main():
    plan_path = ROOT / 'configs/p109-sampling-uniform.json'
    plan = json.loads(plan_path.read_text(encoding='utf8'))
    out = ROOT / 'runs' / ('p112-microbatch-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(exist_ok=False)
    result = {'status': 'running', 'results': [], 'seed': 112391, 'warmups': 2,
              'repetitions': 2, 'effective_rows': 16, 'script_sha256': sha(Path(__file__)),
              'plan_sha256': sha(plan_path), 'pytorch': torch.__version__,
              'scope': 'Separate fixed story and old-stream batches; forward/backward only, no optimizer, clipping or update.'}
    def save():
        (out / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
    save()
    parent, checkpoint = ROOT / plan['parent'], ROOT / plan['initial_language']
    original = {p.relative_to(ROOT).as_posix(): sha(p) for p in (parent, checkpoint)}
    result['protected_sha256'] = original
    if original[plan['parent']] != plan['parent_sha256'] or original[plan['initial_language']] != plan['initial_language_sha256']:
        raise ValueError('Protected input differs from plan')
    data, base = ROOT / plan['data'], ROOT / plan['base_data']
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf8'))
    base_manifest = json.loads((base / 'manifest.json').read_text(encoding='utf8'))
    for directory, meta, names in ((data, manifest, ('tokenizer.json', 'pretrain-train.bin')),
                                   (base, base_manifest, ('tokenizer.json', 'pretrain-train.bin'))):
        for name in names:
            if sha(directory / name) != meta['files'][name]:
                raise ValueError('Training input differs: ' + str(directory / name))
    if manifest['tokenizer_sha256'] != base_manifest['tokenizer_sha256']:
        raise ValueError('Different vocabularies')
    tokenizer = Tokenizer.from_file(str(data / 'tokenizer.json'))
    story_stream = np.memmap(data / 'pretrain-train.bin', mode='r', dtype=np.uint16)
    old_stream = np.memmap(base / 'pretrain-train.bin', mode='r', dtype=np.uint16)
    sampler = UniformDocumentStream(story_stream, *(tokenizer.token_to_id(s) for s in ('<|bos|>', '<|eos|>', '<|pad|>')), plan['length'])
    rng = random.Random(result['seed'])
    sx, sy = sampler.batch(rng, 16, plan['length'], 'cpu')
    starts = [rng.randrange(len(old_stream) - plan['length']) for _ in range(16)]
    seq = np.stack([np.array(old_stream[i:i+plan['length']+1], dtype=np.int64) for i in starts])
    ox, oy = torch.tensor(seq[:, :-1].copy()), torch.tensor(seq[:, 1:].copy())
    batches = {'story': (sx, sy), 'old_stream': (ox, oy)}
    result['samples'] = {name: {'sha256': batch_digest(x, y), 'valid_targets': int((y != -100).sum()),
                                'row_lengths': (y != -100).sum(dim=1).tolist()}
                         for name, (x, y) in batches.items()}
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    result['gpu'] = torch.cuda.get_device_name(0)
    organism, state, _ = load(parent, 'cuda')
    frozen = frozen_digest(organism, state)
    model = restore_conversation(organism, torch.load(checkpoint, map_location='cpu', weights_only=True),
                                 plan['parent_sha256'], manifest['tokenizer_sha256'], 'cuda')
    model.train()
    current = None
    try:
        references = {}
        for micro in (4, 8, 16):
            for name, (x, y) in batches.items():
                current = {'source': name, 'microbatch': micro}
                pieces = split_batches(x, y, micro)
                total = int((y != -100).sum())
                if sum(int((b != -100).sum()) for _, b in pieces) != total:
                    raise AssertionError('Target count changed')
                # Check values and locations, including trimmed ignored tails.
                for i, (a, b) in enumerate(pieces):
                    start = i * micro
                    if not torch.equal(a, x[start:start+len(a), :a.shape[1]]) or not torch.equal(b, y[start:start+len(b), :b.shape[1]]):
                        raise AssertionError('Targets or context changed')
                def run():
                    model.zero_grad(set_to_none=True)
                    loss, count = backward_token_budget(model, state, pieces)
                    if count != total:
                        raise AssertionError('Wrong denominator')
                    return float(loss)
                for _ in range(result['warmups']):
                    run()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                timings = []
                for _ in range(result['repetitions']):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    loss = run()
                    torch.cuda.synchronize()
                    timings.append(time.perf_counter() - t)
                peak_allocated = torch.cuda.max_memory_allocated()
                peak_reserved = torch.cuda.max_memory_reserved()
                actual = gradients(model)
                if name not in references:
                    references[name] = actual
                    diff = {'max_abs': 0.0, 'relative_l2': 0.0}
                else:
                    diff = compare(references[name], actual)
                row = {**current, 'targets': total, 'widths': [a.shape[1] for a, _ in pieces],
                       'seconds': timings, 'targets_per_second': [total / t for t in timings],
                       'median_targets_per_second': total / statistics.median(timings),
                       'peak_allocated_bytes': peak_allocated, 'peak_reserved_bytes': peak_reserved,
                       'loss': loss, 'gradient_vs_micro4': diff, 'gradients_close': True}
                result['results'].append(row)
                save()
                print(json.dumps(row), flush=True)
        result['status'] = 'complete'
    except torch.cuda.OutOfMemoryError as exc:
        result['status'] = 'oom'
        result['oom'] = {**(current or {}), 'message': str(exc)}
        print(json.dumps(result['oom']), flush=True)
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    finally:
        result['frozen_unchanged'] = frozen_digest(organism, state) == frozen
        result['protected_unchanged'] = all(sha(ROOT / name) == digest for name, digest in original.items())
        save()
        print('RESULT ' + str(out / 'result.json'), flush=True)
    if result['status'] != 'complete' or not result['frozen_unchanged'] or not result['protected_unchanged']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
