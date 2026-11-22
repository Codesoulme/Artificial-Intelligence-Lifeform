"""Fit complete multi-turn responses while preserving the date-memory branch."""

import argparse
import json
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from audit_awake_multiturn_layers import encode
from scripts.audit_spora_chat_backbones import DEVELOPMENT, evaluate, reply
from scripts.audit_spora_native_query_key import write
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_dialogue_corrections import LAYERS, retention_requests
from spora.durable import TensorStore
from spora.integrated_checkpoint import load_integrated_runtime
from spora.integrated_durable import _bootstrap_metadata, _observation_calendar, load_integrated_durable
from spora.language_training import frozen_digest
from spora.named_address import address_basis_sha256
from spora.pretrained_attention import SPORAResidualBlock
from train_spora_query_key_dialogue import answer_request


DATA = ROOT / 'data/dialogue-composition-v1.json'
FOUNDATIONS = ROOT / 'data/chinese-foundations.json'
SEED = 22242


def settings_for(decoder, tokenizer, limit):
    return {'device': str(decoder.base.device), 'deepseek': True, 'limit': limit,
            'eos': [tokenizer.eos_token_id]}


def parse_answer(answer, case):
    updated = [re.search(pattern, answer) is not None for pattern in case['updated']]
    retained = [re.search(pattern, answer) is not None for pattern in case['retained']]
    stale = any(re.search(pattern, answer) for pattern in case['forbidden'])
    sentences = [part for part in re.split(r'[。！？!?\n]+', answer) if part.strip()]
    concise = 0 < len(sentences) <= case['max_sentences']
    form = 'form' not in case or re.fullmatch(case['form'], answer.strip()) is not None
    return {'updated': updated, 'retained': retained, 'stale': stale, 'concise': concise, 'form': form,
            'complete': all(updated + retained) and not stale and concise and form}


def score_rollouts(decoder, tokenizer, cases, *, settings=None):
    if settings is None:
        settings = settings_for(decoder, tokenizer, 64)
    result = {}
    for case in cases:
        rows, messages = [], []
        for question in case['turns']:
            generated = reply(decoder, tokenizer, messages, question, **settings)
            rows.append({'question': question, **generated})
            messages.extend(({'role': 'user', 'content': question},
                             {'role': 'assistant', 'content': generated['answer']}))
        score = parse_answer(rows[-1]['answer'], case)
        result[case['name']] = {'turns': rows, 'score': score, 'passed': score['complete'] and rows[-1]['eos']}
    print('Composition pass', sum(row['passed'] for row in result.values()), '/', len(result), flush=True)
    return result


def record_requests(decoder, tokenizer, training, previous, excluded):
    update, lead, short = [], [], []
    settings = settings_for(decoder, tokenizer, 32)
    for index, case in enumerate(training):
        canonical, actual = [], []
        for turn, (question, answer) in enumerate(case):
            canonical.append({'role': 'user', 'content': question})
            request = answer_request(tokenizer, encode(tokenizer, canonical), answer, {})
            (lead if turn < 2 else update).append(request)
            canonical.append({'role': 'assistant', 'content': answer})
            if index % 2 == 0:
                actual.append({'role': 'user', 'content': question})
                if turn < 2:
                    generated = reply(decoder, tokenizer, actual[:-1], question, **settings)
                    actual.append({'role': 'assistant', 'content': generated['answer']})
                else:
                    update.append(answer_request(tokenizer, encode(tokenizer, actual), answer, {}))
    for row in previous:
        if {row['followup'], *(question for question, _ in row['student'])} & excluded:
            raise ValueError('Short-answer replay overlaps evaluation')
        messages = [{'role': role, 'content': text} for question, answer in row['student']
                    for role, text in (('user', question), ('assistant', answer))]
        messages.append({'role': 'user', 'content': row['followup']})
        short.append(answer_request(tokenizer, encode(tokenizer, messages), row['target'], {}))
    return update, lead, short, retention_requests(tokenizer, excluded, diverse=True)


def request_tiles(requests):
    # Divide each answer into interleaved blocks, each projecting at most four positions.
    # Walking a shuffled pass covers all answer tokens, including positions between slots.
    tiles = []
    for ids, labels, _ in requests:
        count = (len(labels) + 3) // 4
        tiles.extend((ids, labels, tuple(range(part, len(labels), count))) for part in range(count))
    return tiles


def tile_logits(decoder, tile):
    ids, labels, selected = tile
    device = decoder.base.device
    indices = torch.tensor(selected, device=device)
    inputs = torch.tensor([ids], device=device)
    positions = indices + len(ids) - len(labels)
    output = decoder(input_ids=inputs, logits_to_keep=positions, use_cache=False).logits[0]
    return output.float(), torch.tensor(labels, device=device).index_select(0, indices)


def cycle_tiles(tiles, rng):
    while True:
        shuffled = list(tiles)
        rng.shuffle(shuffled)
        yield from shuffled


def fit(decoder, requests, steps):
    blocks = [decoder.base.model.layers[index] for index in LAYERS]
    if any(type(block) is not SPORAResidualBlock or not block.ordinary_only for block in blocks):
        raise ValueError('Expected persisted ordinary-only dialogue residuals')
    if max(len(ids) for pool in requests for ids, _, _ in pool) > 240:
        raise ValueError('Conversation exceeds the supported training input size')
    params = [parameter.requires_grad_(True) for block in blocks for part in (block.down, block.up)
              for parameter in part.parameters()]
    rng = random.Random(SEED)
    banks = [request_tiles(pool) for pool in requests]
    if not all(banks):
        raise ValueError('Each training pool needs labeled answers')
    streams = [cycle_tiles(tiles, rng) for tiles in banks]
    optimizer = torch.optim.AdamW(params, lr=2.5e-4, weight_decay=0.)
    schedule = (0, 0, 1, 0, 2, 0, 3, 0)
    losses, trace = [], []
    torch.cuda.empty_cache()
    token_counts = [0] * len(streams)
    for step in range(steps):
        pool = schedule[step % len(schedule)]
        logits, labels = tile_logits(decoder, next(streams[pool]))
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        token_counts[pool] += labels.numel()
        losses.append(float(loss.detach()))
        if (step + 1) % 32 == 0 or step + 1 == steps:
            item = {'step': step + 1, 'average_loss': sum(losses) / len(losses)}
            trace.append(item)
            print('Composition fit', item, flush=True)
            losses.clear()
    decoder.requires_grad_(False)
    return {'steps': steps, 'parameters': sum(p.numel() for p in params),
            'requests': [len(pool) for pool in requests], 'tiles': [len(tiles) for tiles in banks],
            'supervised_positions': token_counts, 'trace': trace}


def verify_checkpoint(parent, candidate):
    source, result = (torch.load(path, map_location='cpu', weights_only=True) for path in (parent, candidate))
    before, after = source['tensors'], result['tensors']
    allowed = {f'residual/{index}/{name}' for index in LAYERS for name in ('up', 'down')}
    if (before.keys() != after.keys() or not allowed <= before.keys()
            or any(not torch.equal(before[name], after[name]) for name in before.keys() - allowed)
            or all(torch.equal(before[name], after[name]) for name in allowed)):
        raise ValueError('Composition training must change only the two dialogue residuals')
    left, right = (dict(payload['metadata']) for payload in (source, result))
    previous, fitted = left.pop('residuals'), right.pop('residuals')
    expected = [dict(spec, sha256=new['sha256']) if spec['index'] in LAYERS else spec
                for spec, new in zip(previous, fitted, strict=True)]
    if left != right or expected != fitted:
        raise ValueError('Unexpected checkpoint metadata changes')


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    protected = {path: sha(path) for path in (args.checkpoint, args.database, args.fixed, args.previous, DATA,
                                               FOUNDATIONS, DEVELOPMENT)}
    config = json.loads(DATA.read_text(encoding='utf-8'))
    evaluation = config['evaluation']
    excluded = {user for scene in evaluation for user in scene['turns']}
    excluded.update(user for scenario in json.loads(DEVELOPMENT.read_text(encoding='utf-8'))['scenarios']
                    for user in scenario['turns'])
    fixed = json.loads(args.fixed.read_text(encoding='utf-8'))['families']
    for rows in fixed.values():
        for row in rows:
            excluded.add(row['question'])
            excluded.update(user for user, _ in row['history'])
    if any(user in excluded for scene in config['training'] for user, _ in scene):
        raise ValueError('Composition training overlaps evaluation')
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(args.checkpoint, load_full_streamed(), MODEL)
    decoder = runtime.decoder.eval().requires_grad_(False)
    with TensorStore(args.database) as original:
        organism, state, metadata, revision = original.read()
    if ({key: value for key, value in metadata.items() if key != 'observation_calendar'}
            != _bootstrap_metadata(args.checkpoint) or _observation_calendar(metadata, state) is None
            or organism.config != runtime.organism.config
            or address_basis_sha256(organism) != address_basis_sha256(runtime.organism)):
        raise ValueError('The database must be the bound, dated individual')
    digest = frozen_digest(organism, state)
    args.output.mkdir(parents=True, exist_ok=False)
    before = score_rollouts(decoder, tokenizer, evaluation)
    write(args.output / 'before.json', before)
    history = json.loads(args.previous.read_text(encoding='utf-8'))['samples']
    requests = record_requests(decoder, tokenizer, config['training'], history, excluded)
    training = fit(decoder, requests, args.steps)
    after = score_rollouts(decoder, tokenizer, evaluation)
    _, short = evaluate(decoder, tokenizer, [], fixed, settings_for(decoder, tokenizer, 64))
    checkpoint = runtime.save(args.output / 'bootstrap.pt')
    verify_checkpoint(args.checkpoint, checkpoint)
    with TensorStore.create(args.output / 'individual.sqlite', organism, state,
                            {**metadata, **_bootstrap_metadata(checkpoint)}) as copied:
        copy_model, copy_state, _, _ = copied.read()
    if (frozen_digest(copy_model, copy_state) != digest or frozen_digest(organism, state) != digest
            or any(sha(path) != value for path, value in protected.items())):
        raise ValueError('Training or saving changed the original individual or input files')
    write(args.output / 'result.json', {'training': training, 'before': before, 'after': after, 'short': short,
                                        'sources': {str(path): value for path, value in protected.items()},
                                        'checkpoint_sha256': sha(checkpoint), 'individual_digest': digest,
                                        'individual_revision': revision, 'seconds': time.perf_counter() - started,
                                        'peak_gpu_bytes': torch.cuda.max_memory_allocated()})


def verify(args):
    config = json.loads(DATA.read_text(encoding='utf-8'))
    source = json.loads(args.verify.read_text(encoding='utf-8'))
    expected = source['after']
    cases = {case['name']: case for case in config['evaluation']}
    if (sha(args.checkpoint) != source['checkpoint_sha256'] or expected.keys() != cases.keys()
            or any([turn['question'] for turn in row['turns']] != cases[name]['turns']
                   for name, row in expected.items())):
        raise ValueError('Reload requires the same candidate and evaluation messages')
    evaluation_sha = sha(DATA)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    before = sha(args.database)
    with load_integrated_durable(args.database, args.checkpoint, load_full_streamed(), MODEL) as runtime:
        actual = score_rollouts(runtime.decoder, tokenizer, config['evaluation'])
        for name, row in actual.items():
            for turn in row['turns']:
                turn.pop('seconds')
            for turn in expected[name]['turns']:
                turn.pop('seconds')
            final = expected[name]['turns'][-1]
            score = parse_answer(final['answer'], cases[name])
            expected[name].update(score=score, passed=score['complete'] and final['eos'])
        if actual != expected or frozen_digest(runtime.organism, runtime.state) != source['individual_digest']:
            raise ValueError('The reloaded candidate changed dialogue behavior or the individual')
    if sha(args.database) != before or sha(DATA) != evaluation_sha:
        raise ValueError('A reload input changed during verification')
    write(args.output, {'verified': len(actual), 'report_sha256': sha(args.verify),
                        'database_sha256': before, 'evaluation_sha256': evaluation_sha})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'database', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('fixed', 'previous', 'verify'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--steps', type=int, default=176)
    args = parser.parse_args()
    if args.verify is not None:
        verify(args)
    elif args.fixed is not None and args.previous is not None and args.steps >= 8:
        main(args)
    else:
        parser.error('Training requires --fixed, --previous and at least eight steps')
