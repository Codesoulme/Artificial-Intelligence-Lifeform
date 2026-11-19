"""Train a small timing gate on frozen hidden states and evaluate free generation."""
import argparse
import gc
import json
import re
import time
from pathlib import Path

from deepseek_focus_value_position_20261120 import (
    MODEL, ROOT, SOURCE, SOURCE_RUN, check, fingerprint, focused_memory_args,
    load, load_integrated, prefix, sha, torch, write,
)

TRAIN = (
    ('请用一句话描述记录中那个人的饮料喜好。', '那个人爱喝', '果汁'),
    ('当时他最喜欢哪种饮料？请回答一个完整句子。', '他当时最喜欢的饮料是', '牛奶'),
    ('请说说这次经历中此人喜欢的饮品。', '记录显示此人更喜欢喝', '酸奶'),
    ('把当事人的饮品偏好写成一句话。', '这位当事人的首选饮品是', '豆浆'),
    ('那一回她爱喝什么？请完整回答。', '那一回她喜欢的是', '咖啡'),
    ('用一句话描述他的饮料偏好。', '他平时偏爱', '可乐'),
    ('请复述当时记录下来的饮料喜好。', '他比较爱喝', '茶'),
    ('据那段经历，他喜欢的饮料是什么？', '根据那段经历，他最常选的是', '水'),
)
CALIBRATION = (
    ('根据当时的经历复述此人选的饮品。', '此人那时最爱的是', '果汁'),
    ('请把那个人爱喝的饮料写成一句话。', '这位当事人爱喝', '牛奶'),
)
ORDINARY = (
    ('解释喝水为何重要。', '喝水有助于维持身体的正常功能。'),
    ('给这段经历起一个标题。', '一次愉快的相遇。'),
    ('你能帮我写一句问候语吗？', '你好，祝你今天一切顺利。'),
    ('什么是质数？', '质数是只有两个正因数的自然数。'),
)
HELD = '请用完整的话告诉我，那份记录里的当事人最爱哪种饮品？'


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


@torch.inference_mode()
def features(decoder, tokenizer, positives, negatives):
    vectors, labels = [], []
    for question, opening, value in positives:
        prompt = encode(tokenizer, prefix(tokenizer, question))
        beginning = prompt + encode(tokenizer, opening)
        full = beginning + encode(tokenizer, value + '。')
        output = decoder(input_ids=torch.tensor([full], device='cuda'), output_hidden_states=True,
                         use_cache=False).hidden_states[-1][0].float().cpu()
        for position in range(len(prompt) - 1, len(full)):
            vectors.append(output[position])
            labels.append(float(position == len(beginning) - 1))
    for question, answer in negatives:
        prompt = encode(tokenizer, prefix(tokenizer, question))
        full = prompt + encode(tokenizer, answer)
        output = decoder(input_ids=torch.tensor([full], device='cuda'), output_hidden_states=True,
                         use_cache=False).hidden_states[-1][0].float().cpu()
        for position in range(len(prompt) - 1, len(full)):
            vectors.append(output[position])
            labels.append(0.)
    return torch.stack(vectors), torch.tensor(labels)


def train_gate(vectors, labels):
    vectors = torch.nn.functional.normalize(vectors, dim=-1)
    gate = torch.nn.Linear(vectors.shape[-1], 1)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=.02, weight_decay=.05)
    weights = torch.where(labels == 1, (labels == 0).sum() / (labels == 1).sum(), 1.)
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        logits = gate(vectors).flatten()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, weight=weights)
        loss.backward()
        optimizer.step()
    return gate.eval().requires_grad_(False)


@torch.inference_mode()
def complete(decoder, tokenizer, prompt, controls, gate=None, delta=None, threshold=None):
    tokens = torch.tensor([prompt], device='cuda')
    steps = []
    fired = False
    for index in range(40):
        use_gate = gate is not None and not fired
        output = decoder(input_ids=tokens, use_cache=False, output_hidden_states=use_gate, **controls)
        scores = output.logits[0, -1].float()
        probability = None
        if use_gate:
            hidden = torch.nn.functional.normalize(output.hidden_states[-1][0, -1].float(), dim=-1)
            probability = float(gate(hidden).sigmoid())
            if probability >= threshold:
                scores = scores + delta
                fired = True
        token = scores.argmax().reshape(1, 1)
        steps.append({'token': int(token), 'gate_probability': probability,
                      'applied': use_gate and fired})
        if int(token) == tokenizer.eos_token_id:
            break
        tokens = torch.cat((tokens, token), dim=-1)
    response = tokenizer.decode([step['token'] for step in steps], skip_special_tokens=True).strip()
    return response, steps


def screen(text, word, vocabulary):
    others = [other for other in vocabulary if other != word and other in text.replace(word, '')]
    affirmative = not re.search('不喜欢|无法|不能|不知道|没有记录|[?？]', text)
    position = text.find(word)
    predicate = bool(re.search('喜欢|偏爱|爱喝|饮品是|饮料是|首选|选择', text[:position]))
    return {'contains_target': position >= 0, 'conflicting_values': others,
            'sentence_screen': position >= 0 and affirmative and predicate and not others}


def main(out):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if out.exists():
        raise FileExistsError(out)
    if torch.cuda.mem_get_info()[0] < 3.7 * 1024**3:
        raise RuntimeError('Insufficient free GPU memory for this experiment')
    plan = check()
    torch.set_num_threads(2)
    torch.manual_seed(12091)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, dtype=torch.float16,
        attn_implementation='sdpa', device_map={'': 'cuda'}, low_cpu_mem_usage=True).eval()
    decoder, parent, old = load_integrated(SOURCE, base, MODEL, individual_device='cuda')
    del parent, old
    decoder.eval().requires_grad_(False)
    gc.collect()
    torch.cuda.empty_cache()

    train_vectors, train_labels = features(decoder, tokenizer, TRAIN, ORDINARY[:2])
    gate = train_gate(train_vectors.clone(), train_labels.clone())
    validation, validation_labels = features(decoder, tokenizer, CALIBRATION, ORDINARY[2:])
    probabilities = gate(torch.nn.functional.normalize(validation, dim=-1)).flatten().sigmoid()
    threshold = 0.8
    calibration = {'threshold': threshold, 'positives': int(validation_labels.sum()),
                   'true_positive': int(((probabilities >= threshold) & (validation_labels == 1)).sum()),
                   'negatives': int((validation_labels == 0).sum()),
                   'false_positive': int(((probabilities >= threshold) & (validation_labels == 0)).sum())}
    out.mkdir()
    torch.save(gate.state_dict(), out / 'gate.pt')
    gate.load_state_dict(torch.load(out / 'gate.pt', weights_only=True))
    gate.to('cuda')
    questions = {item['name']: item['text'] for item in plan['questions']}
    probe = encode(tokenizer, prefix(tokenizer, questions['previous_word']))
    probe_tensor = torch.tensor([probe], device='cuda')
    prompts = {'development': questions['sentence'], 'held_wording': HELD}
    write(out / 'plan.json', {'script_sha256': sha(Path(__file__)),
                            'source_plan_sha256': sha(SOURCE_RUN / 'plan.json'),
                            'train': TRAIN, 'calibration': CALIBRATION, 'ordinary': ORDINARY,
                            'calibration_result': calibration, 'evaluation_questions': prompts,
                            'gate_sha256': sha(out / 'gate.pt'), 'max_new_tokens': 40,
                            'probe': questions['previous_word']})
    print(json.dumps(calibration), flush=True)
    vocabulary = {fixture['expected_word'] for fixture in plan['fixtures']}
    rows = []
    with torch.inference_mode(), (out / 'progress.jsonl').open('x', encoding='utf-8') as stream:
        baseline = decoder(input_ids=probe_tensor, use_cache=False).logits[0, -1].float()
        for fixture in plan['fixtures']:
            organism, state, _ = load(SOURCE_RUN / (fixture['name'] + '.pt'), 'cuda')
            if fingerprint(organism, state) != fixture['state_digest']:
                raise ValueError('Fixture restore mismatch')
            args, selected = focused_memory_args(organism, state)
            if (selected['key'], selected['event_time']) != (fixture['key'], fixture['event']):
                raise ValueError('Incorrect focus read')
            controls = dict(memory_state=state, **args)
            delta = decoder(input_ids=probe_tensor, use_cache=False, **controls).logits[0, -1].float() - baseline
            if not torch.isfinite(delta).all():
                raise ValueError('Invalid evidence delta')
            for name, question in prompts.items():
                if name == 'held_wording' and fixture['world'] != 1:
                    continue
                prompt = encode(tokenizer, prefix(tokenizer, question))
                for mode in ('reader', 'gated'):
                    kwargs = {'gate': gate, 'delta': delta, 'threshold': threshold} if mode == 'gated' else {}
                    text, steps = complete(decoder, tokenizer, prompt, controls, **kwargs)
                    row = dict(fixture=fixture['name'], question=name, mode=mode,
                               text=text, target=fixture['expected_word'], steps=steps,
                               **screen(text, fixture['expected_word'], vocabulary))
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                if fixture == plan['fixtures'][0 if name == 'development' else 8]:
                    text, steps = complete(decoder, tokenizer, prompt, {})
                    row = dict(fixture=None, question=name, mode='disabled', text=text, steps=steps)
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            if fingerprint(organism, state) != fixture['state_digest']:
                raise ValueError('Generation mutated state')
            print(f'Completed {fixture["name"]}, {len(rows)} rows', flush=True)
    write(out / 'result.json', {'rows': rows, 'calibration': calibration,
                               'seconds': time.perf_counter() - started,
                               'peak_allocated_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/deepseek-focus-timing-gate')
    main(parser.parse_args().out)
