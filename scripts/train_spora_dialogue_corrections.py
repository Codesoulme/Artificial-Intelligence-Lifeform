"""Compare authored and student-generated conversation histories on one SPORA checkpoint."""

import argparse
from collections import deque
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from audit_awake_multiturn_layers import encode
from scripts.audit_spora_chat_backbones import DEVELOPMENT, evaluate
from scripts.audit_spora_native_query_key import write
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_calendar_history_chat import _chat_inputs, ordinary_reply
from spora.durable import TensorStore
from spora.integrated_checkpoint import load_integrated_runtime
from spora.integrated_durable import _bootstrap_metadata, _observation_calendar
from spora.language_training import frozen_digest
from spora.named_address import address_basis_sha256
from spora.pretrained_attention import SPORAPretrainedBlock, SPORAResidualBlock
from train_spora_query_key_dialogue import answer_request, prediction, teacher_targets


TRAINING = ROOT / 'data/dialogue-correction-training-v1.json'
FOUNDATIONS = ROOT / 'data/chinese-foundations.json'
LAYERS = (24, 27)
RANK = 16
SEED = 22221


def capture_histories(decoder, tokenizer, training, excluded, ordinary_limit, recorded=None):
    samples = []
    for family, cases in training.items():
        for first, first_answer, update, update_answer, followup, target in cases:
            if {first, update, followup} & excluded:
                raise ValueError('A training question also appears in evaluation')
            if first == update or not target.strip():
                raise ValueError('A correction needs distinct user turns and a target')
            record = {'family': family, 'canonical': [[first, first_answer], [update, update_answer]],
                      'followup': followup, 'target': target}
            if recorded is None:
                student = deque()
                for question in (first, update):
                    generated = ordinary_reply(decoder, tokenizer, student, question)
                    student.append((question, generated['answer'] or ''))
                record['student'] = [list(pair) for pair in student]
            else:
                saved = recorded[len(samples)]
                if ({key: saved[key] for key in record} != record
                        or [user for user, _ in saved['student']] != [first, update]
                        or any(not isinstance(answer, str) for _, answer in saved['student'])):
                    raise ValueError('Recorded histories do not match the labeled training set')
                record['student'] = saved['student']
            for history in (record['canonical'], record['student']):
                inputs, omitted = _chat_inputs(tokenizer, history, followup, ordinary_limit)
                if inputs is None or omitted:
                    raise ValueError('Training prefix does not fit the decoder context')
            samples.append(record)
        print('Verified student histories', family, len(cases), flush=True)
    return samples


def retention_requests(tokenizer, excluded, *, diverse):
    rows = json.loads(FOUNDATIONS.read_text(encoding='utf-8'))['dialogues']
    topics = {'greeting', 'reflection', 'explanation', 'writing', 'value'}
    if diverse:
        topics = {'animals', 'arithmetic', 'cause', 'clarify', 'constraint', 'correction', 'description',
                  'emotion', 'evidence', 'fairness', 'format', 'greeting', 'honesty', 'planning',
                  'preference_change', 'reference', 'rewriting', 'story', 'summary', 'support',
                  'uncertainty', 'unknown_private', 'values_focus'}
    candidates = [row for row in rows if row['topic'] in topics
                  and all(question not in excluded for question, _ in row['turns'])]
    requests = []
    for row in candidates:
        history = []
        for question, answer in row['turns']:
            history.append({'role': 'user', 'content': question})
            requests.append(answer_request(tokenizer, encode(tokenizer, history), answer, {}))
            history.append({'role': 'assistant', 'content': answer})
    if not requests:
        raise ValueError('Missing broad dialogue retention examples')
    return requests


def diverse_prediction(decoder, request):
    """Project four answer positions at most, keeping the response start and EOS."""
    ids, labels, _ = request
    device = decoder.base.device
    count = len(labels)
    indices = torch.tensor([i * (count - 1) // max(1, min(count, 4) - 1)
                            for i in range(min(count, 4))], device=device)
    inputs = torch.tensor([ids], device=device)
    positions = indices + len(ids) - count
    logits = decoder(input_ids=inputs, logits_to_keep=positions, use_cache=False).logits[0].float()
    return logits, torch.tensor(labels, device=device).index_select(0, indices)


def fit(decoder, originals, tokenizer, samples, retention, references, arm, steps, *, diverse):
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    layers = decoder.base.model.layers
    for index in LAYERS:
        layers[index] = SPORAResidualBlock(originals[index], rank=RANK, ordinary_only=True,
                                          memory_context=decoder._memory_context).eval()
    parameters = [parameter for index in LAYERS for part in (layers[index].down, layers[index].up)
                  for parameter in part.parameters()]
    requests = []
    for row in samples:
        messages = [{'role': role, 'content': text} for user, answer in row[arm]
                    for role, text in (('user', user), ('assistant', answer))]
        messages.append({'role': 'user', 'content': row['followup']})
        requests.append(answer_request(tokenizer, encode(tokenizer, messages), row['target'], {}))
    optimizer = torch.optim.AdamW(parameters, lr=3.5e-4, weight_decay=0.)
    references = [value.float().log_softmax(-1) for value in references]
    loss_log, correction_losses, replay_losses = [], [], []
    interval = 2 if diverse else 4
    replay_prediction = diverse_prediction if diverse else prediction
    replay_steps, supervised_tokens = 0, 0
    for step in range(steps):
        if step % interval == interval - 1:
            index = rng.randrange(len(retention))
            logits, labels = replay_prediction(decoder, retention[index])
            reference = references[index].to(logits.device)
            loss = .75 * F.kl_div(logits.log_softmax(-1), reference, log_target=True, reduction='batchmean')
            if diverse:
                loss = .4 * loss + .6 * F.cross_entropy(logits, labels)
                supervised_tokens += labels.numel()
            replay_steps += 1
            replay_losses.append(float(loss.detach()))
        else:
            index = rng.randrange(len(requests))
            logits, targets = prediction(decoder, requests[index])
            loss = F.cross_entropy(logits, targets)
            supervised_tokens += targets.numel()
            correction_losses.append(float(loss.detach()))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % 32 == 0 or step + 1 == steps:
            row = {'step': step + 1,
                   'correction_loss': sum(correction_losses) / len(correction_losses) if correction_losses else None,
                   'replay_loss': sum(replay_losses) / len(replay_losses) if replay_losses else None}
            loss_log.append(row)
            correction_losses.clear()
            replay_losses.clear()
            print('Correction fit', arm, row, flush=True)
    decoder.requires_grad_(False)
    return {'parameters': sum(p.numel() for p in parameters), 'trace': loss_log,
            'target_tokens': sum(len(request[1]) for request in requests),
            'sampled_supervised_tokens': supervised_tokens, 'replay_steps': replay_steps}


def verify_residual_only(parent, candidate):
    old, new = (torch.load(path, map_location='cpu', weights_only=True) for path in (parent, candidate))
    left, right = dict(old['metadata']), dict(new['metadata'])
    old_residuals, new_residuals = left.pop('residuals', []), right.pop('residuals')
    required = [spec for spec in new_residuals if spec['index'] in LAYERS]
    if (left != right or old_residuals != [spec for spec in new_residuals if spec['index'] not in LAYERS]
            or [spec['index'] for spec in required] != list(LAYERS)
            or any(spec.get('ordinary_only') is not True for spec in required)):
        raise ValueError('Candidate changed unrelated checkpoint metadata')
    a, b = old['tensors'], new['tensors']
    added = {f'residual/{index}/{part}' for index in LAYERS for part in ('down', 'up')}
    if a.keys() & added or b.keys() != a.keys() | added or any(not torch.equal(a[key], b[key]) for key in a):
        raise ValueError('Candidate changed unrelated checkpoint tensors')


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = [args.checkpoint, args.database, args.fixed, TRAINING, DEVELOPMENT, FOUNDATIONS]
    if args.histories is not None:
        sources.append(args.histories)
    protected = {path: sha(path) for path in sources}
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(args.checkpoint, load_full_streamed(), MODEL)
    decoder = runtime.decoder.eval().requires_grad_(False)
    layers = decoder.base.model.layers
    originals = {i: layers[i] for i in LAYERS}
    if any(type(layer) is not SPORAPretrainedBlock for layer in originals.values()):
        raise ValueError('Correction layers must be unadapted Qwen-compatible SPORA blocks')
    with TensorStore(args.database) as store:
        organism, state, metadata, revision = store.read()
    binding = _bootstrap_metadata(args.checkpoint)
    if {key: value for key, value in metadata.items() if key != 'observation_calendar'} != binding:
        raise ValueError('Training individual must be bound to the parent checkpoint')
    if (runtime.organism.config != organism.config
            or address_basis_sha256(runtime.organism) != address_basis_sha256(organism)
            or _observation_calendar(metadata, state) is None):
        raise ValueError('Training individual differs from bootstrap or lacks the date history')
    individual = frozen_digest(organism, state)
    development = json.loads(DEVELOPMENT.read_text(encoding='utf-8'))['scenarios']
    families = json.loads(args.fixed.read_text(encoding='utf-8'))['families']
    training = json.loads(TRAINING.read_text(encoding='utf-8'))['families']
    exclusions = {question for scene in development for question in scene['turns']}
    for cases in families.values():
        for case in cases:
            exclusions.add(case['question'])
            exclusions.update(pair[0] for pair in case['history'])
    recorded = None
    if args.histories is not None:
        recorded = json.loads(args.histories.read_text(encoding='utf-8'))['samples']
        if len(recorded) != sum(len(cases) for cases in training.values()):
            raise ValueError('Recorded histories do not cover all of the labeled scenarios')
    samples = capture_histories(decoder, tokenizer, training, exclusions, decoder.position_table_max_position - 64,
                                recorded=recorded)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / 'histories.json', {'samples': samples})
    diverse = args.replay == 'diverse'
    retention = retention_requests(tokenizer, exclusions, diverse=diverse)
    if diverse:
        with torch.no_grad():
            reference = [diverse_prediction(decoder, request)[0].log_softmax(-1).half().cpu() for request in retention]
        torch.cuda.empty_cache()
    else:
        reference = teacher_targets(decoder, retention)
    settings = {'device': str(decoder.base.device), 'deepseek': True, 'limit': 64, 'eos': [tokenizer.eos_token_id]}
    arms = ('canonical', 'student') if args.arm == 'both' else (args.arm,)
    for arm in arms:
        result = fit(decoder, originals, tokenizer, samples, retention, reference, arm, args.steps, diverse=diverse)
        rollouts, fixed = evaluate(decoder, tokenizer, development, families, settings)
        directory = args.output / arm
        directory.mkdir()
        checkpoint = runtime.save(directory / 'bootstrap.pt')
        verify_residual_only(args.checkpoint, checkpoint)
        with TensorStore.create(directory / 'individual.sqlite', organism, state,
                                {**metadata, **_bootstrap_metadata(checkpoint)}) as copied:
            model, memory, _, _ = copied.read()
            if frozen_digest(model, memory) != individual:
                raise ValueError('Saving an isolated candidate changed the neural individual')
        write(directory / 'result.json', {'training': result, 'rollouts': rollouts, 'fixed': fixed,
                                          'checkpoint_sha256': sha(checkpoint)})
    if frozen_digest(organism, state) != individual or any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError('Training changed a frozen source')
    write(args.output / 'result.json', {
        'steps_per_arm': args.steps, 'arms': list(arms), 'replay': args.replay,
        'replay_max_answer_positions': 4 if diverse else None,
        'histories': len(samples), 'retention_requests': len(retention),
        'source_revision': revision, 'individual_sha256': individual, 'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
        'seconds': time.perf_counter() - started, 'sources': {str(path): digest for path, digest in protected.items()},
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'database', 'fixed', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--arm', choices=('both', 'canonical', 'student'), default='both')
    parser.add_argument('--replay', choices=('baseline', 'diverse'), default='baseline')
    parser.add_argument('--histories', type=Path, help='Reuse and validate a previous capture of the same training cases.')
    options = parser.parse_args()
    if options.steps < 4:
        parser.error('--steps must be at least four')
    main(options)
