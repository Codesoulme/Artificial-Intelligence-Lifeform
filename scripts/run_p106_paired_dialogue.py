"""Matched dialogue updates from a conversation checkpoint and a frozen individual.

Inputs are development-side paired JSONL from build_matched_dialogue.py. This
entry forks the language weights and intentionally starts new AdamW optimizers.
Use --help for the explicit group-selection and resume options.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from spora.conversation import restore_conversation
from spora.growing import load
from spora.language_training import atomic_save, frozen_digest, interface_state
from spora.paired_order import PairedOrder, padded
from spora.state_codec import flatten


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def select(context, control, cap, seed):
    # Validate the complete upstream pair set even when selecting a subset.
    groups = defaultdict(list)
    for row in context:
        groups[row['group']].append(row['pair_id'])
    selected = set()
    for group, ids in groups.items():
        ids = sorted(ids, key=lambda k: (hashlib.sha256(
            json.dumps([seed, group, k], ensure_ascii=False).encode()).hexdigest(), k))
        selected.update(ids if cap is None else ids[:cap])
    return ([r for r in context if r['pair_id'] in selected],
            [r for r in control if r['pair_id'] in selected])


def run(args):
    if args.batch_size < 1 or args.epochs < 1 or args.length < 1 or args.seed < 0 or args.threads < 1 or (args.until is not None and args.until < 1):
        raise ValueError('Positive batch, epochs, length and threads and nonnegative seed required')
    if args.max_per_group is not None and args.max_per_group < 1:
        raise ValueError('Group cap must be positive')
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('Learning rate must be finite and positive')
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only CPU and CUDA are supported')
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    source = Path(args.data).resolve()
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format') != 'matched-dialogue-v1':
        raise ValueError('Expected matched-dialogue-v1 manifest')
    inputs = {'individual': Path(args.individual).resolve(), 'language': Path(args.language).resolve(),
              'tokenizer': Path(args.tokenizer).resolve(), 'manifest': source / 'manifest.json',
              'context': source / 'context.jsonl', 'control': source / 'last-user.jsonl'}
    hashes = {name: sha(path) for name, path in inputs.items()}
    if hashes['tokenizer'] != manifest['tokenizer_sha256'] or any(
        hashes[key] != manifest['files'][file] for key, file in
        [('context', 'context.jsonl'), ('control', 'last-user.jsonl')]):
        raise ValueError('Data or tokenizer differs from its manifest')
    tokenizer = Tokenizer.from_file(str(inputs['tokenizer']))
    pad = tokenizer.token_to_id('<|pad|>')
    if pad is None:
        raise ValueError('Tokenizer needs <|pad|>')
    context, control = rows(inputs['context']), rows(inputs['control'])
    PairedOrder(context, control, seed=args.seed, epochs=args.epochs, length=args.length)
    all_groups = dict(sorted(Counter(r['group'] for r in context).items()))
    context, control = select(context, control, args.max_per_group, args.seed)
    order = PairedOrder(context, control, seed=args.seed, epochs=args.epochs, length=args.length)
    selected_groups = dict(sorted(Counter(r['group'] for r in context).items()))
    organism, state, _ = load(inputs['individual'], str(device))
    identity = frozen_digest(organism, state)
    individual_schema, individual_tensors = flatten(organism, state)
    individual_snapshot = dict(schema=individual_schema, tensors={k: v.detach().cpu().clone() for k, v in individual_tensors.items()})
    language = torch.load(inputs['language'], map_location='cpu', weights_only=True)
    models = [restore_conversation(organism, language, hashes['individual'], hashes['tokenizer'], str(device)) for _ in range(2)]
    if args.length > models[0].config.context or pad >= models[0].config.vocabulary:
        raise ValueError('Length or padding exceeds model dimensions')
    if any(t >= models[0].config.vocabulary for row in context + control for t in row['tokens']):
        raise ValueError('Token outside model vocabulary')
    optimizers = [torch.optim.AdamW(m.trainable_parameters(), lr=args.lr, weight_decay=.01) for m in models]
    root = Path(__file__).resolve().parents[1]
    code_files = ('spora/paired_order.py', 'spora/conversation.py', 'spora/growing.py',
                  'spora/language_training.py', 'spora/state_codec.py', 'spora/temporal.py',
                  'spora/model.py', 'spora/sleep_memory.py', 'spora/typed_memory.py')
    contract = dict(format='p106-paired-v1', inputs=hashes, code=sha(__file__),
                    dependencies={p: sha(root / p) for p in code_files},
                    frozen_digest=identity, seed=args.seed, epochs=args.epochs, length=args.length,
                    batch_size=args.batch_size, lr=args.lr, device=str(device), threads=args.threads,
                    max_per_group=args.max_per_group, all_pairs=args.all_pairs,
                    full_groups=all_groups, selected_groups=selected_groups,
                    selection_fingerprint=order.fingerprint, pad=pad)
    out = Path(args.output).resolve()
    if any(out == p or out in p.parents for p in inputs.values()):
        raise ValueError('Output cannot contain an input')
    step, logs, elapsed = 0, [], 0.0
    if args.resume:
        checkpoints = sorted(out.glob('step-*.pt'))
        if not checkpoints:
            raise ValueError('No completed checkpoint to resume')
        snapshot = torch.load(checkpoints[-1], map_location='cpu', weights_only=True)
        if snapshot['contract'] != contract:
            raise ValueError('Resume contract differs')
        order.load_state_dict(snapshot['sampler'])
        for model, opt, arm in zip(models, optimizers, snapshot['arms']):
            model.load_state_dict({**model.state_dict(), **arm['state']}, strict=True)
            opt.load_state_dict(arm['optimizer'])
        step, logs, elapsed = snapshot['step'], snapshot['logs'], snapshot['seconds']
        random.setstate(snapshot['python_rng'])
        torch.set_rng_state(snapshot['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(snapshot['cuda_rng'])
        elif snapshot['cuda_rng']:
            raise ValueError('CUDA random state on CPU')
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out / 'contract.json').write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.until is not None and args.until <= step:
        raise ValueError('until must exceed the last completed step')
    print(json.dumps({'selected_groups': selected_groups, 'all_groups': all_groups,
                      'selected_pairs': len(context), 'total_exposures': order.total}, ensure_ascii=False), flush=True)
    start = time.perf_counter()
    while order.cursor < order.total and (args.until is None or step < args.until):
        batch = order.prepare(args.batch_size)
        tensors = []
        for arm in ('context', 'last_user'):
            x, y = padded(batch[arm], args.length, pad)
            tensors.append((torch.tensor(x, device=device), torch.tensor(y, device=device)))
        for i in range(len(batch['events'])):
            if not torch.equal(tensors[0][1][i][tensors[0][1][i] != -100],
                               tensors[1][1][i][tensors[1][1][i] != -100]):
                raise ValueError('Final supervised target sequence differs')
        losses = []
        step_start = time.perf_counter()
        for model, optimizer, (x, y) in zip(models, optimizers, tensors):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x, state).flatten(0, 1), y.flatten(), ignore_index=-100)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss; resume from the last completed checkpoint')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if not all(bool(torch.isfinite(p).all()) for p in model.trainable_parameters()):
                raise ValueError('Nonfinite updated weights; resume from the last completed checkpoint')
            losses.append(float(loss.detach()))
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        if frozen_digest(organism, state) != identity:
            raise ValueError('Frozen individual changed')
        order.commit()
        step += 1
        exposure = dict(sorted(Counter(e['group'] for e in order.state_dict()['events']).items()))
        logs.append(dict(step=step, context_loss=losses[0], last_user_loss=losses[1],
                         targets=sum(e['target_count'] for e in batch['events']),
                         seconds=time.perf_counter()-step_start, group_exposures=exposure))
        snapshot = dict(format='p106-paired-v1', contract=contract, step=step, sampler=order.state_dict(),
                        individual=individual_snapshot,
                        arms=[dict(format='spora-conversation-v1', organism_sha256=hashes['individual'],
                                   tokenizer_sha256=hashes['tokenizer'], config=asdict(m.config),
                                   state=interface_state(m), optimizer=o.state_dict()) for m, o in zip(models, optimizers)],
                        python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                        logs=logs, seconds=elapsed+time.perf_counter()-start)
        path = out / f'step-{step:08d}.pt'
        if path.exists():
            raise FileExistsError(path)
        atomic_save(snapshot, path)
        print(json.dumps(logs[-1], ensure_ascii=False), flush=True)
    return snapshot if step else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('individual', 'language', 'tokenizer', 'data', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=106)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--length', type=int, required=True)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--threads', type=int, default=2)
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--max-per-group', type=int, help='deterministically limit each group')
    choice.add_argument('--all-pairs', action='store_true', help='explicitly accept original group distribution')
    p.add_argument('--until', type=int, help='stop after this many joint updates')
    p.add_argument('--resume', action='store_true')
    run(p.parse_args())


if __name__ == '__main__':
    main()
