"""Learn when to inject event evidence from the frozen reader's own rollouts."""

import argparse
import gc
import json
import re
import time
from pathlib import Path

from deepseek_focus_timing_gate import (
    CALIBRATION, ORDINARY, TRAIN, complete, encode, train_gate,
)
from deepseek_focus_value_position_20261120 import (
    MODEL, ROOT, SOURCE, SOURCE_RUN, check, fingerprint, focused_memory_args,
    load, load_integrated, prefix, sha, torch, write,
)

PRIOR_GATE = ROOT / 'runs/deepseek-focus-timing-gate/gate.pt'
TRAIN_QUESTIONS = tuple(row[0] for row in TRAIN)
CAL_QUESTIONS = tuple(row[0] for row in CALIBRATION)
HELD_QUESTIONS = (
    '那一段经历里提到的饮料偏好是什么？请把人物和饮品写在同一句话里。',
    '当时那位当事人偏好哪种饮料？请完整回答。',
    '能用一个完整句子说说，在那次记录里对方爱喝哪种饮品吗？',
    '关于这个人的饮品喜好，那份记录是怎么说的？',
)
HELD_ORDINARY = (
    '请用一句话解释饮品和食品有什么区别。',
    '写一句话邀请同事明天来参加讨论。',
    '请问如何计算长方形的面积？',
    '这次经历记录于哪一年？',
)
THRESHOLDS = (0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0)
OPENING = re.compile(r'(?:喜欢喝|爱喝|想喝|愿意喝|偏爱|喜欢的(?:饮品|饮料)?是|爱的是|(?:饮品|饮料|首选|偏好)是)$')
PREDICATE = re.compile(r'喜欢|爱喝|偏爱|最爱|饮品是|饮料是|首选')
SUBJECT = re.compile(r'人|用户|当事|他|她|对方|其')
OTHER_CLAIM = re.compile(r'不喜欢|不爱喝|没有|不知道|不清楚|无法|不能|[?？]')


def screen(answer, target, words):
    text = answer.strip()
    position = text.find(target)
    if position < 0:
        return False
    before = text[:position]
    remaining = text.replace(target, '')
    return (text.endswith(('。', '！', '.')) and bool(SUBJECT.search(before))
            and bool(PREDICATE.search(before)) and not OTHER_CLAIM.search(text)
            and not any(word != target and word in remaining for word in words))


def decode(tokenizer, tokens):
    return tokenizer.decode(tokens, skip_special_tokens=True).strip()


@torch.inference_mode()
def follow_branch(decoder, tokenizer, tensor, controls, prefix_tokens, token, budget):
    generated = [*prefix_tokens, token]
    if token == tokenizer.eos_token_id:
        return decode(tokenizer, generated)
    tensor = torch.cat((tensor, tensor.new_tensor([[token]])), dim=1)
    for _ in range(budget):
        scores = decoder(input_ids=tensor, use_cache=False, **controls).logits[0, -1]
        token = int(scores.argmax())
        generated.append(token)
        if token == tokenizer.eos_token_id:
            break
        tensor = torch.cat((tensor, tensor.new_tensor([[token]])), dim=1)
    return decode(tokenizer, generated)


@torch.inference_mode()
def collect(decoder, tokenizer, prompt, controls, delta, target, words):
    tensor = torch.tensor([prompt], device='cuda')
    generated, hidden, positives, trials = [], [], [], []
    for index in range(40):
        output = decoder(input_ids=tensor, output_hidden_states=True, use_cache=False, **controls)
        scores = output.logits[0, -1].float()
        normal_token = int(scores.argmax())
        hidden.append(output.hidden_states[-1][0, -1].float().cpu())
        if target is not None and OPENING.search(decode(tokenizer, generated)):
            changed_token = int((scores + delta).argmax())
            if normal_token != changed_token:
                candidate = follow_branch(decoder, tokenizer, tensor, controls, generated,
                                          changed_token, 39 - index)
                success = screen(candidate, target, words)
                trials.append({'position': index, 'completion': candidate, 'screen': success})
                if success:
                    positives.append(index)
        generated.append(normal_token)
        if normal_token == tokenizer.eos_token_id:
            break
        tensor = torch.cat((tensor, tensor.new_tensor([[normal_token]])), dim=1)
    answer = decode(tokenizer, generated)
    # Leave already-correct responses alone, including alternative safe continuations.
    if target is not None and screen(answer, target, words):
        positives = []
    return torch.stack(hidden), dict(text=answer, tokens=generated, positive_positions=positives, trials=trials)


def labels_for(trace):
    labels = torch.zeros(len(trace['tokens']))
    labels[trace['positive_positions']] = 1.0
    return labels


def calibrate(gate, timelines):
    probabilities = []
    with torch.no_grad():
        for hidden, trace in timelines:
            scores = gate(torch.nn.functional.normalize(hidden, dim=-1)).flatten().sigmoid()
            probabilities.append((scores.tolist(), set(trace['positive_positions'])))
    results = []
    for threshold in THRESHOLDS:
        good = bad = 0
        for scores, positives in probabilities:
            position = next((i for i, score in enumerate(scores) if score >= threshold), None)
            if position is not None:
                good += position in positives
                bad += position not in positives
        results.append({'threshold': threshold, 'good': good, 'bad': bad})
    selected = max(results, key=lambda row: (row['good'] - row['bad'], -row['bad'], row['threshold']))
    return selected, results


def summarize(rows):
    summaries = {}
    for group in ('memory', 'ordinary'):
        for mode in ('reader', 'prior_gate', 'rollout_gate'):
            subset = [row for row in rows if row['group'] == group and row['mode'] == mode]
            pairs = {}
            for row in subset:
                label = (row['question'], row['fixture'].rsplit('-v', 1)[0])
                pairs.setdefault(label, []).append(row['sentence_screen'])
            summaries[f'{group}/{mode}'] = {
                'rows': len(subset), 'triggered': sum(any(step['applied'] for step in row['steps']) for row in subset),
                'sentence_screen': sum(row['sentence_screen'] for row in subset),
                'complete_pairs': sum(len(items) == 2 and all(items) for items in pairs.values()),
                'pairs': sum(len(items) == 2 for items in pairs.values()),
            }
    return summaries


def main(out):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if out.exists():
        raise FileExistsError(out)
    if torch.cuda.mem_get_info()[0] < 3.7 * 1024**3:
        raise RuntimeError('Insufficient free GPU memory for the frozen model')
    plan = check()
    fixtures = plan['fixtures']
    words = sorted({fixture['expected_word'] for fixture in fixtures})
    train_fixtures = [fixture for fixture in fixtures if fixture['world'] == 0]
    held_fixtures = [fixture for fixture in fixtures if fixture['world'] == 1]
    torch.set_num_threads(2)
    torch.manual_seed(21417)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    out.mkdir()
    write(out / 'plan.json', {
        'source_plan_sha256': sha(SOURCE_RUN / 'plan.json'), 'script_sha256': sha(Path(__file__)),
        'prior_gate_sha256': sha(PRIOR_GATE), 'train_questions': TRAIN_QUESTIONS,
        'calibration_questions': CAL_QUESTIONS, 'held_questions': HELD_QUESTIONS,
        'held_ordinary': HELD_ORDINARY, 'train_fixtures': [f['name'] for f in train_fixtures],
        'held_fixtures': [f['name'] for f in held_fixtures], 'thresholds': THRESHOLDS,
        'max_tokens': 40, 'branch_opening_pattern': OPENING.pattern,
        'calibration': 'maximize successful first triggers minus other first triggers',
    })
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, dtype=torch.float16,
        attn_implementation='sdpa', device_map={'': 'cuda'}, low_cpu_mem_usage=True).eval()
    decoder, parent, old = load_integrated(SOURCE, base, MODEL, individual_device='cuda')
    del parent, old
    decoder.eval().requires_grad_(False)
    gc.collect()
    torch.cuda.empty_cache()
    probe_text = next(row['text'] for row in plan['questions'] if row['name'] == 'previous_word')
    probe = torch.tensor([encode(tokenizer, prefix(tokenizer, probe_text))], device='cuda')
    with torch.inference_mode():
        baseline = decoder(input_ids=probe, use_cache=False).logits[0, -1].float()

    def fixture_state(fixture):
        organism, state, _ = load(SOURCE_RUN / (fixture['name'] + '.pt'), 'cuda')
        if fingerprint(organism, state) != fixture['state_digest']:
            raise ValueError('Fixture digest mismatch')
        controls, selection = focused_memory_args(organism, state)
        if (selection['key'], selection['event_time']) != (fixture['key'], fixture['event']):
            raise ValueError('Wrong event selected')
        kwargs = dict(memory_state=state, **controls)
        with torch.inference_mode():
            evidence = decoder(input_ids=probe, use_cache=False, **kwargs).logits[0, -1].float() - baseline
        if not torch.isfinite(evidence).all():
            raise ValueError('Non-finite event evidence')
        return organism, state, kwargs, evidence

    training, calibration = [], []
    with (out / 'training.jsonl').open('x', encoding='utf8') as stream:
        for index, fixture in enumerate(train_fixtures):
            organism, state, kwargs, evidence = fixture_state(fixture)
            cases = [('train', TRAIN_QUESTIONS[(index + shift) % 8], fixture['expected_word']) for shift in (0, 3, 5)]
            cases += [('calibration', CAL_QUESTIONS[index % 2], fixture['expected_word'])]
            if index % 2 == 0:
                cases.extend((('train', ORDINARY[index % 2][0], None),
                              ('calibration', ORDINARY[2 + index // 2 % 2][0], None)))
            for split, question, target in cases:
                prompt = encode(tokenizer, prefix(tokenizer, question))
                features, trace = collect(decoder, tokenizer, prompt, kwargs, evidence, target, words)
                trace.update(split=split, fixture=fixture['name'], question=question, target=target)
                (training if split == 'train' else calibration).append((features, trace))
                stream.write(json.dumps(trace, ensure_ascii=False) + '\n')
            stream.flush()
            if fingerprint(organism, state) != fixture['state_digest']:
                raise ValueError('Collection mutated fixture')
            print('Collected', fixture['name'], flush=True)

    vectors = torch.cat([features for features, _ in training])
    labels = torch.cat([labels_for(trace) for _, trace in training])
    counts = {'train_trajectories': len(training), 'train_positive_positions': int(labels.sum()),
              'train_positions': len(labels), 'calibration_trajectories': len(calibration),
              'calibration_positive_positions': sum(len(trace['positive_positions']) for _, trace in calibration)}
    print(json.dumps(counts), flush=True)
    if not 0 < int(labels.sum()) < len(labels):
        raise ValueError('Rollouts contained no usable positive/negative training contrast')
    gate = train_gate(vectors, labels)
    selection, grid = calibrate(gate, calibration)
    torch.save(gate.state_dict(), out / 'gate.pt')
    prior = torch.nn.Linear(vectors.shape[-1], 1)
    prior.load_state_dict(torch.load(PRIOR_GATE, weights_only=True))
    gate.to('cuda')
    prior.eval().requires_grad_(False).to('cuda')
    print(json.dumps(selection), flush=True)
    rows = []
    with (out / 'progress.jsonl').open('x', encoding='utf8') as stream:
        for index, fixture in enumerate(held_fixtures):
            organism, state, kwargs, evidence = fixture_state(fixture)
            cases = [('memory', question) for question in HELD_QUESTIONS]
            cases.append(('ordinary', HELD_ORDINARY[index % len(HELD_ORDINARY)]))
            for group, question in cases:
                prompt = encode(tokenizer, prefix(tokenizer, question))
                modes = {'reader': {}, 'prior_gate': dict(gate=prior, delta=evidence, threshold=.8),
                         'rollout_gate': dict(gate=gate, delta=evidence, threshold=selection['threshold'])}
                for mode, options in modes.items():
                    response, steps = complete(decoder, tokenizer, prompt, kwargs, **options)
                    row = {'group': group, 'fixture': fixture['name'], 'question': question, 'mode': mode,
                           'expected': fixture['expected_word'], 'response': response, 'steps': steps,
                           'sentence_screen': group == 'memory' and screen(response, fixture['expected_word'], words)}
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            if fingerprint(organism, state) != fixture['state_digest']:
                raise ValueError('Evaluation mutated fixture')
            print('Evaluated', fixture['name'], len(rows), flush=True)
    report = {'counts': counts, 'threshold_selection': selection, 'threshold_grid': grid,
              'summary': summarize(rows), 'rows': rows, 'seconds': time.perf_counter() - started,
              'plan_sha256': sha(out / 'plan.json'), 'gate_sha256': sha(out / 'gate.pt'),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
    write(out / 'result.json', report)
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/deepseek-focus-rollout-gate')
    main(parser.parse_args().out)
