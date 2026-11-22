"""Train and evaluate native dialog residuals on the anchored DeepSeek descendant."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from audit_awake_multiturn_layers import CEVAL_SUBJECTS, ceval_questions, encode, score_ceval
from audit_awake_reasoning_residual_holdout_20261121 import DATA, load_scenarios
from audit_spora_anchored_dialogue import SCENARIOS
from audit_spora_dialogue_migration import dialogue
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_qwen_counterfactual_reply_20261122 import HELD, WORDS, evaluate, records
from spora.integrated_checkpoint import load_integrated_runtime
from spora.language_training import frozen_digest
from spora.pretrained_attention import SPORAResidualBlock
from train_awake_absolute_attention import replay_requests
from train_awake_replacement_replay import divergence
from train_spora_query_key_dialogue import answer_request, prediction, teacher_targets


FOUNDATIONS = ROOT / 'data/chinese-foundations.json'
SOURCE = ROOT / 'data/chinese-pilot/source.json'
CURRICULUM = ROOT / 'runs/teaching-curriculum-v5/teacher.jsonl'
LAYERS = (24, 27)
RANK = 16
SEED = 221831


def get_training(tokenizer, organism):
    authored = json.loads(FOUNDATIONS.read_text(encoding='utf-8'))['dialogues']
    histories = [[{'role': role, 'content': content} for pair in item['turns']
                  for role, content in zip(('user', 'assistant'), pair, strict=True)] for item in authored]
    with CURRICULUM.open(encoding='utf-8') as stream:
        histories.extend(json.loads(line)['messages'] for line in stream)
    prose = json.loads(SOURCE.read_text(encoding='utf-8'))['train']
    development = {text for case in load_scenarios() for text in case['turns']}
    development |= {text.replace('{text}\n\n', '').replace('\n\n{text}', '')
                    for _, turns in SCENARIOS for text, _ in turns}
    ordinary, long, seen = [], [], set()
    for index, messages in enumerate(histories):
        if any(msg['role'] == 'user' and msg['content'] in development for msg in messages):
            continue
        first = next(i for i, msg in enumerate(messages) if msg['role'] == 'user')
        if messages[first]['content'] in seen:
            continue
        seen.add(messages[first]['content'])
        passage = '\n'.join(prose[index % len(prose):] + prose[:index % len(prose)])
        filler = tokenizer.decode(tokenizer.encode(passage, add_special_tokens=False)[:290])
        augmented = [dict(message) for message in messages]
        augmented[first]['content'] = filler + '\n\n' + messages[first]['content']
        for position, message in enumerate(messages):
            if position == 0 or message['role'] != 'assistant' or messages[position - 1]['role'] != 'user':
                continue
            for entries, pool, limit in ((messages, ordinary, 176), (augmented, long, 480)):
                prompt = encode(tokenizer, entries[:position])
                sample = answer_request(tokenizer, prompt, message['content'], {})
                if len(sample[0]) <= limit and len(sample[1]) <= 40:
                    pool.append(sample)
    replay = [answer_request(tokenizer, prompt, WORDS[value], controls) for prompt, controls, value in
              replay_requests(organism, tokenizer, value_offset=3, with_answers=True)]
    if not ordinary or not long or not replay:
        raise ValueError('Expected short and long labeled conversations and memory replay')
    return ordinary, long, replay


def language(model, tokenizer):
    data = json.loads(SOURCE.read_text(encoding='utf-8'))
    filler = tokenizer.decode(tokenizer.encode('\n'.join(data['development'] + data['test']),
                                              add_special_tokens=False)[:350])
    names = ('earlier_later', 'schedule_revision', 'conditional_plan', 'negation_and_reason')
    scenarios = {case['name']: case['turns'] for case in load_scenarios() if case['name'] in names}
    responses = {}
    for name, questions in scenarios.items():
        responses[name] = dialogue(model, tokenizer, questions, open_reasoning=False, limit=48, seed=SEED)
    for name, turns in SCENARIOS:
        questions = [question.format(text=filler) for question, _ in turns]
        rows = dialogue(model, tokenizer, questions, open_reasoning=False, limit=48, seed=SEED)
        for row, (_, expected) in zip(rows, turns, strict=True):
            row['expected'] = list(expected)
            row['strict'] = row['answer'].strip('。！!，,；;？? \n') in expected
        responses[name] = rows
    return responses


def assess(runtime, tokenizer, academic):
    decoder = runtime.decoder
    return {'dialogue': language(decoder, tokenizer),
            'ceval': score_ceval(decoder, academic, decoder.position_table_max_position),
            'memory': evaluate(decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))}


def stable(rows):
    return {**rows, 'dialogue': {name: [{key: value for key, value in turn.items() if key != 'seconds'}
                                        for turn in turns] for name, turns in rows['dialogue'].items()}}


def verify_residual_checkpoint(parent, candidate):
    source, fitted = (torch.load(path, map_location='cpu', weights_only=True) for path in (parent, candidate))
    allowed = {f'residual/{index}/{part}' for index in LAYERS for part in ('down', 'up')}
    before, after = source['tensors'], fitted['tensors']
    frozen = before.keys() - allowed
    if (after.keys() != frozen | allowed
            or any(not torch.equal(before[key], after[key]) for key in frozen)):
        raise ValueError('Saving the dialogue candidate changed other tensors')
    left, right = dict(source['metadata']), dict(fitted['metadata'])
    source_specs, candidate_specs = left.pop('residuals', []), right.pop('residuals')
    if left != right or [spec for spec in source_specs if spec['index'] not in LAYERS] != [
        spec for spec in candidate_specs if spec['index'] not in LAYERS
    ]:
        raise ValueError('Saving the dialogue candidate changed other structure metadata')


def rollout_followups(decoder, tokenizer):
    topics = {'reference', 'correction', 'count', 'time', 'preference_change', 'list', 'spatial', 'sequence'}
    authored = json.loads(FOUNDATIONS.read_text(encoding='utf-8'))['dialogues']
    prose = json.loads(SOURCE.read_text(encoding='utf-8'))['train']
    examples = []
    for index, item in enumerate(row for row in authored if row['topic'] in topics):
        questions = [question for question, _ in item['turns']]
        passage = '\n'.join(prose[index:] + prose[:index])
        filler = tokenizer.decode(tokenizer.encode(passage, add_special_tokens=False)[:290])
        for turns in (questions, [filler + '\n\n' + questions[0], *questions[1:]]):
            observed = dialogue(decoder, tokenizer, turns, open_reasoning=False, limit=24, seed=SEED)
            history = []
            for position, (row, (_, expected)) in enumerate(zip(observed, item['turns'], strict=True)):
                history.append({'role': 'user', 'content': row['question']})
                if position:
                    examples.append(answer_request(tokenizer, encode(tokenizer, history), expected, {}))
                history.append({'role': 'assistant', 'content': row['answer']})
    return examples


def fit(runtime, tokenizer, steps, *, on_policy=False):
    decoder = runtime.decoder
    short, long, memory = get_training(tokenizer, runtime.organism)
    rng = random.Random(SEED)
    memory_reference = teacher_targets(decoder, memory)
    retention = rng.sample(short, 24)
    reference = teacher_targets(decoder, retention)
    pools = [short, long, memory, retention]
    schedule = (0, 1, 0, 2, 1, 3, 0, 1)
    if on_policy:
        pools.append(rollout_followups(decoder, tokenizer))
        schedule = (0, 4, 2, 1, 4, 3, 0, 4)
    layers = decoder.base.model.layers
    for index in LAYERS:
        if on_policy:
            if not isinstance(layers[index], SPORAResidualBlock) or layers[index].rank != RANK:
                raise ValueError('Expected a dialogue residual to continue training')
            layers[index].down.requires_grad_(True)
            layers[index].up.requires_grad_(True)
        else:
            layers[index] = SPORAResidualBlock(layers[index], rank=RANK).eval()
    parameters = [param for index in LAYERS for part in (layers[index].down, layers[index].up)
                  for param in part.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=.00035, weight_decay=0.)
    local = []
    trace = []
    for step in range(steps):
        kind = schedule[step % len(schedule)]
        index = rng.randrange(len(pools[kind]))
        logits, labels = prediction(decoder, pools[kind][index])
        if kind in (0, 1, 4):
            loss = F.cross_entropy(logits, labels)
        else:
            target = (memory_reference if kind == 2 else reference)[index].float().log_softmax(-1)
            loss = divergence(logits, target) * (2. if kind == 2 else .65)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        local.append((kind, float(loss.detach())))
        if (step + 1) % 32 == 0 or step + 1 == steps:
            grouped = [[loss for key, loss in local if key == k] for k in range(len(pools))]
            row = {'step': step + 1, 'loss': [sum(v) / len(v) if v else None for v in grouped]}
            trace.append(row)
            local.clear()
            print('Dialogue fit', row, flush=True)
    decoder.requires_grad_(False)
    return {'steps': steps, 'rank': RANK, 'layers': list(LAYERS), 'on_policy': on_policy,
            'pool_sizes': [len(pool) for pool in pools], 'trace': trace}


def main(args):
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    files = (args.checkpoint, DATA, FOUNDATIONS, SOURCE, CURRICULUM,
             *(args.ceval / f'{subject}.parquet' for subject in CEVAL_SUBJECTS))
    inputs = {path: sha(path) for path in files}
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed()
    if args.verify_report is not None:
        original = json.loads(args.verify_report.read_text(encoding='utf-8'))
        if original['checkpoint_sha256'] != sha(args.checkpoint):
            raise ValueError('Candidate checkpoint differs from its training report')
        runtime = load_integrated_runtime(args.checkpoint, base, MODEL)
        runtime.decoder.eval().requires_grad_(False)
        individual = frozen_digest(runtime.organism, runtime.state)
        if individual != original['individual_sha256'] or runtime.decoder.anchored_attention_layout != original['layout']:
            raise ValueError('Reload changed the original individual or attention layout')
        restored = assess(runtime, tokenizer, list(ceval_questions(args.ceval, tokenizer)))
        if (stable(restored) != stable(original['candidate'])
                or individual != frozen_digest(runtime.organism, runtime.state)
                or any(sha(path) != digest for path, digest in inputs.items())):
            raise ValueError('Reload changed candidate behavior or memory')
        output.mkdir(parents=True, exist_ok=False)
        (output / 'result.json').write_text(json.dumps({'exact_reopen': True, 'seconds': time.perf_counter() - started,
                                                       'checkpoint_sha256': original['checkpoint_sha256']}) + '\n',
                                           encoding='utf-8')
        print('Candidate reproduced after reload', flush=True)
        return
    original = None
    if not args.on_policy:
        print('Loaded raw DeepSeek', flush=True)
        original = language(base, tokenizer)
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL)
    runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    if runtime.decoder.anchored_attention_layout is None:
        raise ValueError('Expected an anchored SPORA checkpoint')
    academic = list(ceval_questions(args.ceval, tokenizer))
    parent = assess(runtime, tokenizer, academic)
    print('Evaluated anchored parent', flush=True)
    training = fit(runtime, tokenizer, args.steps, on_policy=args.on_policy)
    candidate = assess(runtime, tokenizer, academic)
    if individual != frozen_digest(runtime.organism, runtime.state) or any(
        sha(path) != digest for path, digest in inputs.items()
    ):
        raise ValueError('The dialogue pilot changed a frozen input or individual')
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / 'candidate.pt'
    runtime.save(checkpoint)
    verify_residual_checkpoint(args.checkpoint, checkpoint)
    report = {'original_dialogue': original, 'parent': parent, 'candidate': candidate,
              'training': training, 'checkpoint_sha256': sha(checkpoint),
              'individual_sha256': individual, 'layout': runtime.decoder.anchored_attention_layout,
              'sources': {str(path): digest for path, digest in inputs.items()},
              'seconds': time.perf_counter() - started, 'peak_gpu_bytes': torch.cuda.max_memory_allocated()}
    (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Wrote', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--ceval', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=192)
    parser.add_argument('--verify-report', type=Path)
    parser.add_argument('--on-policy', action='store_true', help='Train follow-ups on the loaded model’s actual replies')
    args = parser.parse_args()
    if args.steps < 8:
        parser.error('--steps must be at least 8')
    main(args)
