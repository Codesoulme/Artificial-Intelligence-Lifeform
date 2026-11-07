"""Real single-thread CPU replay of two independent frozen v8 restores.

Only writes a new directory under runs/p27b-resume-check and the designated
review report. Never invokes the training entrypoint or any CUDA API.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import sys
sys.dont_write_bytecode = True
import argparse
import ast
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from tokenizers import Tokenizer
from spora.conversation import restore_conversation
from spora.conversation_data import DialogueRows
from spora.growing import load
from spora.language_training import frozen_digest, interface_state
from spora.token_objective import backward_token_budget
from spora.training_mix import choose_source

SOURCE = ROOT/'scripts/train_conversation_v8.py'
REPORT = ROOT/'reports/P27b-resume-review.json'
OUTPUT_ROOT = ROOT/'runs/p27b-resume-check'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def log(**values):
    print(json.dumps(values, ensure_ascii=True), flush=True)


def production_code():
    """Copy AST nodes, not a handwritten approximation of the sampler.

    Stop immediately after targets += actual. GPU setup, monitoring, snapshots,
    and the trainer's filesystem writes are outside this compiled block.
    """
    source = SOURCE.read_text('utf-8-sig')
    tree = ast.parse(source)
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name in ('load_rows', 'padded', 'text_batch')]
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    loop = next(n for n in main.body if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 'step')
    stop = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.AugAssign)
                and isinstance(n.target, ast.Name) and n.target.id == 'targets')
    body = loop.body[:stop+1]
    assert isinstance(body[-1].value, ast.Name) and body[-1].value.id == 'actual'
    module = ast.Module(body=helpers+body, type_ignores=[])
    assert not any(isinstance(n, ast.Attribute) and n.attr == 'cuda' for n in ast.walk(module))
    assert not any(isinstance(n, ast.Constant) and n.value == 'cuda' for n in ast.walk(module))
    sources = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'SOURCES' for t in n.targets)))
    return compile(module, str(SOURCE), 'exec'), {
        'path': str(SOURCE.relative_to(ROOT)), 'sha256': sha(SOURCE),
        'first_update_line': body[0].lineno, 'last_update_line': body[-1].end_lineno,
        'ast_sha256': hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest(),
        'method': 'Execute unchanged helper/update AST from v8; pass CPU model and restored state. No trainer main, monitor, or GPU API executed.',
    }, sources


def tensor_bytes(tensor):
    assert tensor.device.type == 'cpu'
    return tensor.detach().contiguous().numpy().tobytes()


def compare(a, b, path='root', mismatches=None):
    if mismatches is None:
        mismatches = []
    if type(a) is not type(b):
        mismatches.append(path+': type')
    elif isinstance(a, torch.Tensor):
        if a.dtype != b.dtype or a.shape != b.shape or tensor_bytes(a) != tensor_bytes(b):
            mismatches.append(path+': tensor bytes')
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            mismatches.append(path+': keys')
        for key in a.keys() & b.keys():
            compare(a[key], b[key], path+'/'+str(key), mismatches)
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b):
            mismatches.append(path+': length')
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, path+'/'+str(i), mismatches)
    elif isinstance(a, float):
        if struct.pack('!d', a) != struct.pack('!d', b):
            mismatches.append(path+': float bits')
    elif a != b:
        mismatches.append(path+': value')
    return mismatches


def optimizer_steps(optimizer_state):
    return {str(k): int(v['step']) for k, v in optimizer_state['state'].items()}


def wait_checkpoint(path, seconds):
    deadline = time.monotonic()+seconds
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError('B100 checkpoint not available; no replay performed')
        log(status='waiting_for_checkpoint', checkpoint=str(path))
        time.sleep(min(15, max(0, deadline-time.monotonic())))
    # Production atomic_save publishes with rename; do not use checkpoint.pt,
    # which may be replaced by ongoing training.


def replay(attempt, path, plan, inputs, code, steps, out):
    started = time.perf_counter()
    # Both attempts reread the checkpoint AND parent independently from disk.
    payload = torch.load(path, map_location='cpu', weights_only=True)
    organism, state, _ = load(ROOT/plan['parent'], 'cpu')
    original = frozen_digest(organism, state)
    model = restore_conversation(organism, payload, plan['parent_sha256'], inputs['tokenizer_sha256'], 'cpu')
    assert all(p.device.type == 'cpu' for p in model.parameters())
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=plan['learning_rate'],
                                 weight_decay=.01, betas=(.9, .95))
    optimizer.load_state_dict(copy.deepcopy(payload['optimizer']))
    initial_optimizer_steps = optimizer_steps(optimizer.state_dict())
    assert initial_optimizer_steps and all(v == payload['steps'] for v in initial_optimizer_steps.values())
    rng = random.Random()
    rng.setstate(payload['python_rng'])
    torch.set_rng_state(payload['torch_rng'])
    # CUDA RNG tensors are preserved as opaque CPU bytes. CPU dropout/randomness
    # would use torch_rng; the current forward/backward has no stochastic layers.
    env = dict(torch=torch, np=np, math=math, hashlib=hashlib, DialogueRows=DialogueRows,
        choose_source=choose_source, backward_token_budget=backward_token_budget,
        model=model, state=state, optimizer=optimizer, plan=plan, rng=rng,
        stream=inputs['stream'], teacher=inputs['teacher'], foundation=inputs['foundation'],
        sft=inputs.get('dialogue'), focus=inputs['focus'], pad=inputs['pad'],
        counts=copy.deepcopy(payload['counts']), focus_cursor=payload['focus_cursor'],
        targets=payload['targets'], batch_trace=payload['batch_trace_sha256'])
    trace = []
    log(status='replay_start', attempt=attempt, source_step=payload['steps'], steps=steps)
    for step in range(payload['steps']+1, payload['steps']+steps+1):
        before_step = interface_state(model)
        start_counts = copy.deepcopy(env['counts'])
        start_trace = env['batch_trace']
        start_targets = env['targets']
        env['step'] = step
        step_started = time.perf_counter()
        exec(code, env)
        elapsed = time.perf_counter()-step_started
        actual = env['actual']
        batches = env['batches']
        counts = env['counts']
        # Independently recompute target count and chained digest from the real
        # sampled tensors; the production body controls both RNG and counters.
        chain = start_trace
        batch_evidence = []
        for x, y in batches:
            raw = x.numpy().tobytes()+y.numpy().tobytes()
            chain = hashlib.sha256(bytes.fromhex(chain)+raw).hexdigest()
            batch_evidence.append({'shape': list(x.shape), 'targets': int((y != -100).sum()),
                                   'sha256': hashlib.sha256(raw).hexdigest()})
        changed = [name for name, value in interface_state(model).items()
                   if tensor_bytes(value) != tensor_bytes(before_step[name])]
        increments = {k: counts.get(k, 0)-start_counts.get(k, 0) for k in counts}
        source_targets = sum(v for k, v in increments.items() if k.endswith('_source_targets'))
        focus_targets = increments.get('focus_targets', 0)
        assert actual == sum(b['targets'] for b in batch_evidence) == source_targets+focus_targets
        assert env['targets']-start_targets == actual and chain == env['batch_trace']
        assert actual >= plan['target_budget'] and actual-batch_evidence[-1]['targets'] < plan['target_budget']
        assert changed, 'No language tensor changed: this cannot count as a replay'
        assert all(v == step for v in optimizer_steps(optimizer.state_dict()).values())
        item = {'step': step, 'loss': float(env['loss']), 'lr': optimizer.param_groups[0]['lr'],
                'actual_targets': actual, 'budget_overshoot': actual-plan['target_budget'],
                'microbatches': batch_evidence, 'counts_increment': increments,
                'targets': env['targets'], 'focus_cursor': env['focus_cursor'],
                'batch_trace_sha256': chain, 'changed_language_tensors': changed}
        trace.append(item)
        log(status='update_completed', attempt=attempt, step=step, loss=item['loss'],
            microbatches=len(batches), actual_targets=actual, seconds=elapsed,
            substantial_cpu_runtime=elapsed > 20)
    snapshot = {
        'language': interface_state(model), 'optimizer': copy.deepcopy(optimizer.state_dict()),
        'python_rng': rng.getstate(), 'torch_rng': torch.get_rng_state().clone(),
        'cuda_rng_opaque_unexecuted': copy.deepcopy(payload['cuda_rng']),
        'counts': copy.deepcopy(env['counts']), 'focus_cursor': env['focus_cursor'],
        'targets': env['targets'], 'steps': payload['steps']+steps,
        'batch_trace_sha256': env['batch_trace'], 'trace': trace,
    }
    old_unchanged = original == frozen_digest(organism, state)
    final_steps = optimizer_steps(snapshot['optimizer'])
    assert all(final_steps[k]-v == steps for k, v in initial_optimizer_steps.items())
    snapshot_path = out/f'attempt-{attempt}-final.pt'
    torch.save(snapshot, snapshot_path)
    summary = {'attempt': attempt, 'seconds': time.perf_counter()-started,
        'old_state_before_sha256': original, 'old_state_unchanged': old_unchanged,
        'language_tensors': len(snapshot['language']), 'optimizer_parameter_states': len(final_steps),
        'optimizer_steps_advanced_exactly': steps, 'source_step': payload['steps'],
        'python_rng_advanced': snapshot['python_rng'] != payload['python_rng'],
        'cpu_rng_changed': not torch.equal(snapshot['torch_rng'], payload['torch_rng']),
        'snapshot': str(snapshot_path.relative_to(ROOT)), 'snapshot_sha256': sha(snapshot_path)}
    del env, snapshot, optimizer, model, organism, state, payload
    gc.collect()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default='configs/p27b-token-budget.json')
    parser.add_argument('--checkpoint', default='runs/p27b-token-budget/step000100.pt')
    parser.add_argument('--output', default='runs/p27b-resume-check')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--expected-step', type=int, default=100)
    parser.add_argument('--wait-seconds', type=int, default=1800)
    args = parser.parse_args()
    if not 1 <= args.steps <= 3 or not 0 <= args.wait_seconds <= 3600:
        raise ValueError('At most three updates per restore; bounded wait required')
    out = (ROOT/args.output).resolve()
    if out != OUTPUT_ROOT.resolve() and OUTPUT_ROOT.resolve() not in out.parents:
        raise ValueError('Output must stay inside runs/p27b-resume-check')
    if out.exists() or REPORT.exists():
        raise FileExistsError('Preserve existing checker artifacts; output/report already exists')
    path = (ROOT/args.checkpoint).resolve()
    if path.name != 'step000100.pt' and args.expected_step == 100:
        raise ValueError('Use immutable step000100.pt for the B100 review')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    code, extraction, source_names = production_code()
    source_hashes = {name: sha(ROOT/name) for name in source_names}
    plan_path = ROOT/args.plan
    plan = read_json(plan_path)
    plan_sha = sha(plan_path)
    wait_checkpoint(path, args.wait_seconds)
    checkpoint_sha = sha(path)
    header = torch.load(path, map_location='cpu', weights_only=True)
    assert header['steps'] == args.expected_step
    assert header['plan_sha256'] == plan_sha
    assert sha(ROOT/plan['parent']) == plan['parent_sha256']
    data = ROOT/plan['data']
    curriculum = ROOT/plan['curriculum']
    manifest = read_json(data/'manifest.json')
    assert header['manifest_sha256'] == sha(data/'manifest.json')
    config = read_json(path.parent/'config.json')
    assert config['source_code_sha256'] == source_hashes
    assert config['plan_sha256'] == plan_sha and config['plan'] == plan
    assert header['tokenizer_sha256'] == manifest['tokenizer_sha256']
    assert sha(curriculum/'manifest.json') == plan['curriculum_manifest_sha256']
    cm = read_json(curriculum/'manifest.json')
    read_inputs = {data/name: manifest['files'][name] for name in
                   ('tokenizer.json', 'pretrain-train.bin', 'focus-tokens.json')}
    read_inputs.update({curriculum/name: cm['files'][name] for name in
                       ('teacher-tokens.jsonl', 'foundations-tokens.jsonl')})
    if 'dialogue' in plan['target_shares']:
        read_inputs[data/'dialogue-train-tokens.jsonl'] = manifest['files']['dialogue-train-tokens.jsonl']
    for filename, expected in read_inputs.items():
        assert sha(filename) == expected, 'Training input changed: '+str(filename)
    input_stats = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in read_inputs}
    tokenizer = Tokenizer.from_file(str(data/'tokenizer.json'))
    inputs = {'tokenizer_sha256': manifest['tokenizer_sha256'], 'pad': tokenizer.token_to_id('<|pad|>'),
        'stream': np.memmap(data/'pretrain-train.bin', dtype=np.uint16, mode='r'),
        'teacher': DialogueRows(curriculum/'teacher-tokens.jsonl', plan['length']),
        'foundation': DialogueRows(curriculum/'foundations-tokens.jsonl', plan['length']),
        'focus': read_json(data/'focus-tokens.json')}
    if 'dialogue' in plan['target_shares']:
        inputs['dialogue'] = DialogueRows(data/'dialogue-train-tokens.jsonl', plan['length'])
    out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'passed': False, 'checkpoint': str(path.relative_to(ROOT)),
        'checkpoint_sha256': checkpoint_sha, 'source_step': header['steps'], 'updates_per_restore': args.steps,
        'cpu_threads': 1, 'gpu_used': False, 'production_ast': extraction,
        'source_sha256': source_hashes, 'plan_sha256': plan_sha,
        'checker_sha256': sha(Path(__file__)), 'attempts': [],
        'scope': 'Two independent CPU restores of the same GPU-produced B100 checkpoint, followed by real production-sampler/token-budget backward/AdamW updates. Not CPU-vs-GPU or uninterrupted-run equivalence.'}
    del header
    try:
        for attempt in (1, 2):
            report['attempts'].append(replay(attempt, path, plan, inputs, code, args.steps, out))
        a = torch.load(out/'attempt-1-final.pt', map_location='cpu', weights_only=True)
        b = torch.load(out/'attempt-2-final.pt', map_location='cpu', weights_only=True)
        fields = list(a)
        discrepancies = {name: compare(a[name], b[name], name) for name in fields}
        report['bitwise_equal'] = {name: not differences for name, differences in discrepancies.items()}
        report['discrepancies'] = {k: v for k, v in discrepancies.items() if v}
        report['trace'] = a['trace']
        report['final_counts'] = a['counts']
        report['final_targets'] = a['targets']
        report['final_batch_trace_sha256'] = a['batch_trace_sha256']
        report['maximum_language_tensor_difference'] = max(float((v-b['language'][k]).abs().max()) for k,v in a['language'].items())
        report['checkpoint_unchanged'] = sha(path) == checkpoint_sha
        report['parent_file_unchanged'] = sha(ROOT/plan['parent']) == plan['parent_sha256']
        report['production_sources_unchanged'] = {name: sha(ROOT/name) == h for name,h in source_hashes.items()}
        report['plan_unchanged'] = sha(plan_path) == plan_sha
        report['training_inputs_stat_unchanged'] = all((Path(p).stat().st_size, Path(p).stat().st_mtime_ns) == v for p,v in input_stats.items())
        report['old_state_unchanged'] = all(r['old_state_unchanged'] for r in report['attempts'])
        report['limitations'] = [
            'CUDA RNG bytes are compared and preserved, but never installed on a GPU or consumed; CUDA replay was not tested.',
            'CPU RNG can remain unchanged because current training has no stochastic forward layers. Python sampler RNG must advance.',
            'Steps101-103 do not reach the focus_every=10 trigger. Focus logic is copied exactly but this replay does not dynamically exercise an injection.',
            'No separately uninterrupted training trajectory was compared, and CPU arithmetic is not claimed bitwise equivalent to GPU arithmetic.',
            'Data inputs were fully hashed before replay and checked for size/mtime changes afterward; frozen production sources, parent and B100 were rehashed.',
            'Wall-clock accounting and monitoring are excluded because they are not deterministic update state. No final test texts were opened.',
        ]
        report['old_checker_discrepancies'] = [
            'verify_coverage_resume.py hard-codes CUDA despite read-only framing; it cannot satisfy this CPU-only request.',
            'Old checker uses one-source-per-update v7 sampling and does not compare complete optimizer/RNG/count/trace state; it is not a valid v8 verifier.',
        ]
        report['passed'] = (all(report['bitwise_equal'].values()) and report['checkpoint_unchanged']
            and report['parent_file_unchanged'] and report['old_state_unchanged']
            and all(report['production_sources_unchanged'].values()) and report['plan_unchanged']
            and report['training_inputs_stat_unchanged'] and all(r['python_rng_advanced'] for r in report['attempts']))
        report['status'] = 'passed' if report['passed'] else 'failed'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        report['seconds_including_wait'] = time.perf_counter()-started
        for name in ('teacher', 'foundation', 'dialogue'):
            if name in inputs:
                inputs[name].close()
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        (out/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        log(status=report['status'], report=str(REPORT), seconds=report['seconds_including_wait'])
    if not report['passed']:
        raise SystemExit('CPU resume replay did not pass')


if __name__ == '__main__':
    main()
