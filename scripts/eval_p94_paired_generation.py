"""Read-only P91 paired story continuation diagnostic."""
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True

import numpy as np
import torch
from tokenizers import Tokenizer
from spora.conversation import ConversationConfig
from spora.document_stream import DocumentStream
from spora.growing import load
from spora.language_training import frozen_digest, interface_state
from spora.paired_experiment import coverage, language_mode
from spora.paired_source_gate import verify_p91_inputs
from spora.random_baseline import make_paired_models
from spora.source_review import file_sha256

PLAN = ROOT / 'configs/p91-paired-random-story.json'
REVIEW = ROOT / 'reports/P91-paired-random-story-review.json'


@contextmanager
def read_lock(directory):
    # Existing lock only: never create or modify a training artifact.
    with (directory / '.p91.lock').open('rb') as stream:
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('Training directory is locked; wait for training to finish') from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError('Training directory is locked; wait for training to finish') from exc
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def read_pair(directory, plan, source):
    meta_path, summary_path, checkpoint = (directory / x for x in ('metadata.json', 'summary.json', 'checkpoint.pt'))
    metadata = json.loads(meta_path.read_text(encoding='utf8'))
    summary = json.loads(summary_path.read_text(encoding='utf8'))
    raw = checkpoint.read_bytes()
    checkpoint_hash = hashlib.sha256(raw).hexdigest()
    if summary.get('format') != 'spora-p91-summary-v1' or summary.get('checkpoint_sha256') != checkpoint_hash:
        raise ValueError('Summary does not match checkpoint')
    envelope = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    payload_bytes = envelope['payload']
    if hashlib.sha256(payload_bytes).hexdigest() != envelope['sha256']:
        raise ValueError('Checkpoint integrity failure')
    p = torch.load(io.BytesIO(payload_bytes), map_location='cpu', weights_only=True)
    s = p['spec']
    expected_spec = dict(seed=s['seed'], data_seed=plan['data_seeds'][str(s['seed'])],
        batch_size=plan['microbatch'], length=plan['length'], budget=plan['target_budget'],
        schedule_steps=plan['schedule_steps'], warmup=plan['warmup'], lr=plan['learning_rate'],
        device=plan['device'], torch_version=metadata['torch'], model_config=plan['model'])
    # Dataclass tuples are retained by torch serialization, while JSON uses lists.
    expected_spec['model_config'] = dict(plan['model'], memory_layers=tuple(plan['model']['memory_layers']))
    if p['format'] != 'spora-paired-v1' or s != expected_spec or s['seed'] not in plan['seeds']:
        raise ValueError('Checkpoint specification mismatch')
    expected_provenance = dict(source=source, code=metadata['code'], protected=metadata['protected'],
        parent=plan['parent_sha256'], metadata_sha256=file_sha256(meta_path))
    if metadata['format'] != 'spora-p91-run-v1' or metadata['source_review'] != source or metadata['seed'] != s['seed'] or p['provenance'] != expected_provenance:
        raise ValueError('Checkpoint provenance mismatch')
    steps, targets = p['steps'], p['targets']
    if type(steps) is not int or steps != s['schedule_steps'] or len(p['trace']) != steps or len(p['logs']) != steps or targets != sum(r['targets'] for r in p['trace']):
        raise ValueError('Incomplete or invalid checkpoint counters')
    if p['spans'] != [span for row in p['trace'] for span in row['spans']] or targets != sum(end-start for _, start, end in p['spans']):
        raise ValueError('Invalid checkpoint coverage')
    expected = dict(steps=steps, targets=targets, seed=s['seed'], coverage=coverage(p['spans']),
        batches=p['trace'], updates=p['logs'], evaluations=p['evaluations'], status='complete')
    for key, value in expected.items():
        if json.dumps(summary.get(key), sort_keys=True) != json.dumps(value, sort_keys=True):
            raise ValueError('Summary content differs from checkpoint: '+key)
    if len(p['models']) != 2 or len(p['optimizers']) != 2:
        raise ValueError('Expected exactly two arms')
    def finite(value):
        if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite checkpoint tensor')
        if isinstance(value, dict):
            for item in value.values(): finite(item)
        elif isinstance(value, (list, tuple)):
            for item in value: finite(item)
    finite(p['optimizers'])
    return p, metadata, dict(checkpoint=checkpoint_hash, summary=file_sha256(summary_path), metadata=file_sha256(meta_path))


def restore_models(organism, state, payload, device='cpu'):
    config = ConversationConfig(**payload['spec']['model_config'])
    models = make_paired_models(organism, config, seed=payload['spec']['seed'])
    if frozen_digest(organism, state) != payload['frozen']:
        raise ValueError('Individual state mismatch')
    for model, values in zip(models, payload['models'], strict=True):
        expected = interface_state(model)
        if values.keys() != expected.keys():
            raise ValueError('Invalid model tensors')
        for name, value in values.items():
            if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape or value.dtype != expected[name].dtype or not bool(torch.isfinite(value).all()):
                raise ValueError('Invalid model tensor: '+name)
        merged = model.state_dict(); merged.update(values); model.load_state_dict(merged, strict=True)
        model.to(device)
        language_mode(model, False)
    return models


def tensor_hash(values):
    h = hashlib.sha256()
    for name, value in sorted(values.items()):
        h.update(name.encode('utf8')); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def prompts_from_stream(stream, prefix_tokens, count):
    eligible = [(i, int(a), int(b)) for i, (a, b) in enumerate(zip(stream.starts, stream.ends)) if int(b)-int(a) > prefix_tokens+1]
    if len(eligible) < count:
        raise ValueError('Not enough development documents for requested prefix length')
    # Evenly spaced, deterministic document selection; each prefix starts with BOS.
    picks = [eligible[i * len(eligible) // count] for i in range(count)]
    return [(i, np.asarray(stream.stream[a:a+prefix_tokens+1], dtype=np.int64).tolist()) for i, a, _ in picks]


@torch.inference_mode()
def generate(model, state, prefix, tokenizer, bos, eos, pad, max_new, device='cpu'):
    tokens = torch.tensor([prefix], dtype=torch.long, device=device)
    if len(prefix) + max_new > model.config.context:
        raise ValueError('Prefix plus generation exceeds model context')
    result, ended = [], False
    for _ in range(max_new):
        logits = model(tokens, state)[0, -1].clone()
        if not bool(torch.isfinite(logits).all()): raise ValueError('Nonfinite logits')
        logits[bos] = logits[pad] = -torch.inf
        token = int(logits.argmax())
        if token == eos:
            ended = True; break
        result.append(token)
        tokens = torch.cat((tokens, tokens.new_tensor([[token]])), dim=1)
    def repeated(n):
        grams = [tuple(result[i:i+n]) for i in range(max(0, len(result)-n+1))]
        return dict(total=len(grams), repeated=len(grams)-len(set(grams)))
    return dict(text=tokenizer.decode(result, skip_special_tokens=False), token_ids=result,
        generated_tokens=len(result), ended_with_eos=ended, budget_exhausted=not ended,
        repetition={str(n): repeated(n) for n in (1, 2, 3, 4)})


def validate_device(device):
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; choose --device cpu')
    if device not in ('cpu', 'cuda'):
        raise ValueError('Unsupported device')
    return device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', default='runs/p94-paired-seed-391109')
    parser.add_argument('--second', default='runs/p94-paired-seed-391110')
    parser.add_argument('--output', required=True, help='new JSON file directly under runs/')
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--prefix-tokens', type=int, default=32, help='development tokens after BOS')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu',
                        help='evaluation device (default: cpu; cuda requires availability)')
    args = parser.parse_args()
    validate_device(args.device)
    if not 1 <= args.count <= 256 or not 1 <= args.prefix_tokens <= 256 or not 1 <= args.max_new_tokens <= 256:
        raise ValueError('Invalid finite diagnostic settings')
    output = (ROOT / args.output).resolve()
    if output.parent != (ROOT/'runs').resolve() or output.suffix.lower() != '.json' or output.exists():
        raise ValueError('Output must be a new JSON file directly under runs')
    directories = [(ROOT / p).resolve() for p in (args.first, args.second)]
    if len(set(directories)) != 2 or any(p.parent != (ROOT/'runs').resolve() for p in directories):
        raise ValueError('Two distinct run directories directly under runs required')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = json.loads(PLAN.read_text(encoding='utf8'))
    if args.prefix_tokens + 1 + args.max_new_tokens > plan['model']['context']:
        raise ValueError('Prefix plus budget exceeds model context')
    with ExitStack() as stack:
        for directory in sorted(directories): stack.enter_context(read_lock(directory))
        source = verify_p91_inputs(ROOT, PLAN, REVIEW)
        pairs = [read_pair(d, plan, source) for d in directories]
        if [p[0]['spec']['seed'] for p in pairs] != plan['seeds']:
            raise ValueError('Runs must cover reviewed seeds in plan order')
        parent = ROOT / plan['parent']
        if file_sha256(parent) != plan['parent_sha256']:
            raise ValueError('Individual file changed')
        for _, metadata, _ in pairs:
            for group in ('code', 'protected'):
                for name, expected in metadata[group].items():
                    if file_sha256(name) != expected: raise ValueError('Protected input changed: '+name)
        data = ROOT / plan['data']
        tokenizer = Tokenizer.from_file(str(data/'tokenizer.json'))
        bos, eos, pad = (tokenizer.token_to_id(s) for s in ('<|bos|>', '<|eos|>', '<|pad|>'))
        if None in (bos,eos,pad) or len({bos,eos,pad}) != 3 or tokenizer.get_vocab_size() != plan['model']['vocabulary']:
            raise ValueError('Tokenizer mismatch')
        stream = DocumentStream(np.memmap(data/'story-development.bin', dtype='<u2', mode='r'), bos, eos, pad)
        if int(stream.stream.max()) >= plan['model']['vocabulary']: raise ValueError('Token outside vocabulary')
        prompts = prompts_from_stream(stream, args.prefix_tokens, args.count)
        prompt_rows = [dict(document=i, token_ids=t, text=tokenizer.decode(t, skip_special_tokens=True)) for i,t in prompts]
        rows = []
        for payload, metadata, hashes in pairs:
            organism, state, _ = load(parent, args.device)
            models = restore_models(organism, state, payload, args.device)
            before = frozen_digest(organism, state)
            arm_rows = []
            for label, model in zip(('memory_bridge', 'standard_causal'), models, strict=True):
                language_before = tensor_hash(interface_state(model))
                results = [generate(model, state, tokens, tokenizer, bos, eos, pad, args.max_new_tokens, args.device) for _,tokens in prompts]
                language_after = tensor_hash(interface_state(model))
                if language_before != language_after or frozen_digest(organism, state) != before:
                    raise ValueError('Model or individual state changed during generation')
                arm_rows.append(dict(arm=label, language_sha256_before=language_before, language_sha256_after=language_after,
                    training=model.training, device=args.device, outputs=results))
            rows.append(dict(seed=payload['spec']['seed'], steps=payload['steps'], targets=payload['targets'],
                inputs=hashes, individual_sha256_before=before, individual_sha256_after=frozen_digest(organism, state), arms=arm_rows))
        # Recheck the exact pair, source binding and individual after generation.
        if verify_p91_inputs(ROOT, PLAN, REVIEW) != source or file_sha256(parent) != plan['parent_sha256']:
            raise ValueError('Input changed during evaluation')
        for directory, (_, _, hashes) in zip(directories, pairs, strict=True):
            for name, expected in hashes.items():
                filename = {'checkpoint':'checkpoint.pt','summary':'summary.json','metadata':'metadata.json'}[name]
                if file_sha256(directory/filename) != expected: raise ValueError('Run changed during evaluation')
        report = dict(format='spora-paired-generation-v1', scope='development story continuation diagnostic; not chat evaluation',
            source=source, parent_sha256=plan['parent_sha256'], tokenizer_sha256=file_sha256(data/'tokenizer.json'),
            development_sha256=file_sha256(data/'story-development.bin'), script_sha256=file_sha256(__file__),
            python=sys.version, torch=str(torch.__version__), device=args.device,
            settings=dict(prompt_selection='evenly spaced eligible development documents, ordered', prefix_tokens=args.prefix_tokens,
                max_new_tokens=args.max_new_tokens, decoding='greedy argmax; BOS and PAD masked; stop on EOS', count=args.count),
            prompts=prompt_rows, runs=rows,
            limitations='Two seeds and selected development prompts are exploratory; prompts within a seed are not independent training replicates. Repetition counts and EOS do not establish quality, significance, generalization, or chat ability. Development reuse can bias comparisons.')
        # Exclusive creation ensures no existing user file can be overwritten.
        with output.open('x', encoding='utf8') as out:
            json.dump(report, out, ensure_ascii=False, indent=2, allow_nan=False)
    print(str(output))


if __name__ == '__main__':
    main()
