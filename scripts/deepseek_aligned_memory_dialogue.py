"""Probe free generation from a focused entity and output-aligned native matrices."""

import argparse
import gc
import json
import random
import time
from pathlib import Path

from deepseek_focus_sentence_bootstrap import fit_style, gather
from deepseek_focus_timing_gate import train_gate
from deepseek_output_aligned_memory import (
    GAIN, MODEL, ROOT, SEED, SLOTS, SOURCE, encode, fingerprint, fresh, load_integrated,
    prefix, read, read_bank, sha, torch, write, write_bank,
)

OUT = ROOT / 'runs/deepseek-aligned-memory-dialogue'
RANK = 90
RELATIONS = ('drink', 'work', 'place', 'ordinary')
THRESHOLD = .8
EARLY_TIMING_THRESHOLD = .65
TRAIN = {
    'drink': (
        ('请用完整句子说明记录里的人喜欢什么饮料。', '这位当事人喜欢喝'),
        ('说说此人的饮品喜好。', '这个人最喜欢的饮品是'),
        ('那个人偏好喝什么？', '那个人偏爱喝'),
        ('请复述那次记录中的饮料偏好。', '那次记录里，他喜欢喝'),
    ),
    'work': (
        ('请用一个完整句子回答这个人的职业。', '这位当事人的职业是'),
        ('那个人是做什么工作的？', '那个人从事的工作是'),
        ('说说记录里的工作身份。', '记录里的这个人是一名'),
        ('在那段经历中，他从事哪种工作？', '当时他的职业是'),
    ),
    'place': (
        ('请说出此人在那段经历里去过的地方。', '这位当事人去了'),
        ('说明那个人去了哪处地点。', '那个人到访了'),
        ('那段记录里她去了什么地方？', '记录里，她去了'),
        ('当事人那次到访了哪里？请完整回答。', '那次，当事人去了'),
    ),
    'ordinary': (
        ('为什么运动后要适量喝水？', ''),
        ('推荐一个小组开会时间。', ''),
        ('怎么准备第一次求职面试？', ''),
        ('想去哪里度假比较合适？', ''),
        ('请解释一下什么是地理学。', ''),
        ('帮我写一句礼貌的问候。', ''),
    ),
}
HELD = {
    'drink': ('用一句话告诉我，这人爱喝哪种饮品？', '那次经历提到了此人偏爱的饮料吗？请直接说明。'),
    'work': ('这位当事人是干哪一行的？请用完整的话回答。', '按照你记得的经历，当事人以什么工作为业？'),
    'place': ('这人那一次到了什么地方？请完整作答。', '从那次经历来看，TA 的到访地点是哪儿？'),
    'ordinary': ('解释热饮和冷饮在冷却速度上的差异。', '列出写一封求职邮件的三个步骤。',
                 '请推荐一个适合两天旅行的行程。', '三加四等于多少？'),
}
FACTS = {
    17: {
        'drink': ('玫瑰荔枝气泡饮', '蜜柚绿茶'),
        'work': ('地质样本管理员', '深海仪器调试员'),
        'place': ('竹湾旧灯塔', '松岚河口站'),
    },
    33: {
        'drink': ('红枣黑豆饮', '青柠薄荷冰茶'),
        'work': ('古建壁画修复师', '舞台灯光设计师'),
        'place': ('南榆旧车站', '白枫山天文台'),
    },
}


@torch.inference_mode()
def question_features(decoder, prompt):
    output = decoder(input_ids=torch.tensor([prompt], device='cuda'), use_cache=False, output_hidden_states=True)
    return torch.nn.functional.normalize(output.hidden_states[-1][0, -1].float(), dim=-1)


def train_route(vectors, labels):
    route = torch.nn.Linear(vectors.shape[1], len(RELATIONS))
    optimizer = torch.optim.AdamW(route.parameters(), lr=.03, weight_decay=.04)
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(route(vectors), labels)
        loss.backward()
        optimizer.step()
    return route.eval().requires_grad_(False)


@torch.inference_mode()
def answer(decoder, tokenizer, prompt, style=None, columns=None, gate=None, biases=None):
    tokens = torch.tensor([prompt], device='cuda')
    generated, trace = [], []
    offset = None
    for index in range(32):
        need_features = style is not None and offset is None
        output = decoder(input_ids=tokens, use_cache=False, output_hidden_states=need_features)
        scores = output.logits[0, -1].float()
        probability = None
        if need_features:
            hidden = torch.nn.functional.normalize(output.hidden_states[-1][0, -1].float(), dim=-1)
            probability = float(gate(hidden).sigmoid())
            if index > 0 and probability >= THRESHOLD:
                offset = 0
            else:
                scores[columns] += style(hidden)
        position = None
        if offset is not None and biases is not None and offset < SLOTS:
            scores += biases[offset]
            position = offset
            offset += 1
        token = int(scores.argmax())
        generated.append(token)
        trace.append({'gate_probability': probability, 'memory_position': position, 'token': token})
        if token == tokenizer.eos_token_id:
            break
        tokens = torch.cat((tokens, tokens.new_tensor([[token]])), dim=1)
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), trace


@torch.inference_mode()
def generate_memory_suffix(decoder, tokenizer, ids, slot_bias, max_new_tokens):
    from spora.memory_output import MemorySequenceLogitsProcessor

    prefix_length = ids.shape[1]
    processor = MemorySequenceLogitsProcessor(slot_bias, prefix_length)
    generation = decoder.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                  logits_processor=[processor], do_sample=False, use_cache=True,
                                  max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
                                  eos_token_id=tokenizer.eos_token_id)
    return generation[0, prefix_length:].tolist()


@torch.inference_mode()
def generate_with_memory(decoder, tokenizer, prompt, style, columns, gate, slot_bias, max_new_tokens=32,
                         *, timing_threshold=THRESHOLD):
    """Generate the opening, then hand off at the learned injection boundary."""
    ids = torch.tensor([prompt], device=slot_bias.device)
    generated = []
    for index in range(max_new_tokens):
        output = decoder(input_ids=ids, use_cache=False, output_hidden_states=True)
        hidden = torch.nn.functional.normalize(output.hidden_states[-1][0, -1].float(), dim=-1)
        probability = float(gate(hidden).sigmoid())
        if index > 0 and probability >= timing_threshold:
            suffix = generate_memory_suffix(decoder, tokenizer, ids, slot_bias, max_new_tokens - index)
            return generated + suffix, index
        scores = output.logits[0, -1].float()
        scores[columns] += style(hidden)
        token = int(scores.argmax())
        generated.append(token)
        if token == tokenizer.eos_token_id:
            break
        ids = torch.cat((ids, ids.new_tensor([[token]])), dim=1)
    return generated, None


def score(row, all_values):
    if row['expected'] is None or row['start'] is None or row['read_relation'] != row['relation']:
        return False
    before, after = row['opening'], row['fact_suffix']
    predicates = {'drink': ('喝', '饮品', '饮料', '偏爱'), 'work': ('职业', '工作', '一名'),
                  'place': ('去', '到访', '地点')}
    return (after.rstrip('。.!！ ') == row['expected'] and any(x in before for x in predicates[row['relation']])
            and any(subject in before for subject in ('人', '他', '她', '对方', '当事'))
            and not any(mark in before for mark in ('？', '?', '无法', '不知道', '没有记录', '不喜欢'))
            and not any(value in before for value in all_values))


@torch.inference_mode()
def matrix_bias(head, site, bank, address, basis):
    slots = read_bank(site, bank, address).to('cuda')[:, :RANK]
    if slots[0].norm() < .2:
        return None
    restored = torch.nn.functional.normalize(slots, dim=-1, eps=1e-4) @ basis.T
    restored[slots.norm(dim=-1) < .2] = 0
    return head(restored.to(head.weight)).float() * GAIN * (head.weight.shape[1] / RANK) ** .5


def main(out, replay=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if (not replay and out.exists()) or (out / 'reload.json').exists():
        raise FileExistsError(out)
    if torch.cuda.mem_get_info()[0] < 3.7 * 1024**3:
        raise RuntimeError('Insufficient free GPU memory for the frozen decoder')
    torch.set_num_threads(2)
    torch.manual_seed(82418)
    started = time.perf_counter()
    protected = (SOURCE, *(MODEL / f for f in ('model.safetensors', 'config.json', 'tokenizer.json',
                  'tokenizer_config.json')), Path(__file__))
    hashes = fingerprint(protected)
    if replay:
        manifest = read(out / 'plan.json')
        baseline = read(out / 'result.json')
        if manifest['source_sha256'] != hashes or sha(out / 'candidate.pt') != baseline['candidate_sha256']:
            raise ValueError('Candidate or source changed')
    else:
        out.mkdir()
        write(out / 'plan.json', {'source_sha256': hashes, 'training': TRAIN, 'held': HELD, 'facts': FACTS,
                                 'rank': RANK, 'gate_threshold': THRESHOLD, 'focus': 'provided entity only'})

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, dtype=torch.float16,
        attn_implementation='sdpa', device_map={'': 'cuda'}, low_cpu_mem_usage=True).eval()
    decoder, parent, _ = load_integrated(SOURCE, base, MODEL, individual_device='cpu')
    del parent, _
    decoder.eval().requires_grad_(False)
    model, _ = fresh()
    site = model.memories[0]
    width = model.config.memory_width
    del model, _
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    head = base.lm_head
    hidden_width = head.weight.shape[1]
    basis = torch.linalg.qr(torch.randn(hidden_width, RANK, generator=torch.Generator().manual_seed(SEED))).Q.to('cuda')
    all_values = sorted({value for group in FACTS.values() for versions in group.values() for value in versions})

    def prompt(question):
        return encode(tokenizer, prefix(tokenizer, question))

    if replay:
        candidate = torch.load(out / 'candidate.pt', map_location='cpu', weights_only=True)
        column_ids = candidate['columns']
        route = torch.nn.Linear(hidden_width, len(RELATIONS))
        route.load_state_dict(candidate['route'])
        style = torch.nn.Sequential(torch.nn.Linear(hidden_width, 24), torch.nn.Tanh(),
                                    torch.nn.Linear(24, len(column_ids)))
        style.load_state_dict(candidate['style'])
        gate = torch.nn.Linear(hidden_width, 1)
        gate.load_state_dict(candidate['gate'])
        columns = torch.tensor(column_ids, device='cuda')
        banks = candidate['banks']
    else:
        column_ids = sorted({token for group in TRAIN.values() for _, text in group for token in encode(tokenizer, text)})
        columns = torch.tensor(column_ids, device='cuda')
        positions = {token: index for index, token in enumerate(column_ids)}
        vectors, labels, inputs, inside, outside, targets, timing, timing_labels = ([] for _ in range(8))
        for label, relation in enumerate(RELATIONS):
            for question, text in TRAIN[relation]:
                current_prompt = prompt(question)
                vectors.append(question_features(decoder, current_prompt).cpu())
                labels.append(label)
                if text:
                    opening = encode(tokenizer, text)
                    hidden, local_scores, other_scores = gather(decoder, tokenizer, current_prompt, opening, {}, columns)
                    inputs.append(hidden[:-1])
                    inside.append(local_scores)
                    outside.append(other_scores)
                    targets.extend(positions[token] for token in opening)
                    timing.append(hidden)
                    timing_labels.extend([0.] * len(opening) + [1.])
        route = train_route(torch.stack(vectors), torch.tensor(labels))
        style, _ = fit_style(torch.cat(inputs), torch.cat(inside), torch.cat(outside), torch.tensor(targets))
        gate = train_gate(torch.cat(timing), torch.tensor(timing_labels))
        banks = {}
        with torch.inference_mode():
            for relation in RELATIONS[:-1]:
                bank = torch.zeros(SLOTS, width, width)
                for version in range(2):
                    for key, group in FACTS.items():
                        value = group[relation][version]
                        ids = [*encode(tokenizer, value), tokenizer.eos_token_id]
                        if len(ids) > SLOTS or len(ids) < 3 or set(ids[:-1]) & set(tokenizer.all_special_ids):
                            raise ValueError('Value does not fit the continuous bank: ' + value)
                        directions = torch.nn.functional.normalize(head.weight[ids].float() @ basis, dim=-1)
                        targets = torch.zeros(SLOTS, width)
                        targets[:len(ids), :RANK] = directions.cpu()
                        bank = write_bank(site, bank, site.current_basis[key:key + 1], targets)
                    banks[f'{relation}:{version}'] = bank.clone()
                banks[f'{relation}:erased'] = write_bank(site, bank, site.current_basis[17:18], torch.zeros(SLOTS, width))
        torch.save({'route': route.state_dict(), 'style': style.state_dict(), 'gate': gate.state_dict(),
                    'columns': column_ids, 'banks': banks}, out / 'candidate.pt')
    for adapter in (route, style, gate):
        adapter.eval().requires_grad_(False).to('cuda')

    cases = []
    for relation in RELATIONS[:-1]:
        for question in HELD[relation]:
            for key in FACTS:
                for version in range(2):
                    cases.append((relation, question, key, version, FACTS[key][relation][version]))
            cases.append((relation, question, 17, 'erased', None))
            cases.append((relation, question, 45, 1, None))
    cases += [('ordinary', question, 17, 1, None) for question in HELD['ordinary']]
    random.Random(82418).shuffle(cases)
    rows = []
    question_cache = {}
    bias_cache = {}
    with torch.inference_mode(), (out / ('reload-progress.jsonl' if replay else 'progress.jsonl')).open('x', encoding='utf8') as stream:
        for relation, question, key, version, expected in cases:
            if question not in question_cache:
                current_prompt = prompt(question)
                probs = route(question_features(decoder, current_prompt)).softmax(dim=-1)
                selected = RELATIONS[int(probs.argmax())]
                reference, reference_trace = answer(decoder, tokenizer, current_prompt)
                question_cache[question] = (current_prompt, selected, probs.tolist(), reference, reference_trace)
            current_prompt, selected, probabilities, reference, reference_trace = question_cache[question]
            confidence = max(probabilities)
            cache_key = (selected, key, version)
            biases = None
            if selected != 'ordinary' and confidence >= THRESHOLD:
                if cache_key not in bias_cache:
                    bank = banks[f'{selected}:{version}']
                    bias_cache[cache_key] = matrix_bias(head, site, bank, site.current_basis[key:key + 1], basis)
                biases = bias_cache[cache_key]
            trace = reference_trace
            response = reference
            if biases is not None:
                response, trace = answer(decoder, tokenizer, current_prompt, style, columns, gate, biases)
            start = next((i for i, step in enumerate(trace) if step['memory_position'] == 0), None)
            tokens = [step['token'] for step in trace]
            opening = tokenizer.decode(tokens[:start], skip_special_tokens=True).strip() if start is not None else ''
            suffix = tokenizer.decode(tokens[start:], skip_special_tokens=True).strip() if start is not None else ''
            row = {'question': question, 'relation': relation, 'entity': key, 'version': version,
                   'expected': expected, 'read_relation': selected, 'route_probabilities': probabilities,
                   'response': response, 'reference_response': reference, 'trace': trace, 'start': start,
                   'opening': opening, 'fact_suffix': suffix, 'reference_tokens_equal': tokens == [s['token'] for s in reference_trace]}
            row['strict'] = score(row, all_values)
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            if len(rows) % 10 == 0:
                print('Completed', len(rows), 'of', len(cases), flush=True)

    facts = [row for row in rows if row['expected'] is not None]
    summary = {}
    for relation in RELATIONS[:-1]:
        group = [row for row in facts if row['relation'] == relation]
        pairs = [[r for r in group if r['entity'] == key and r['question'] == question]
                 for key in FACTS for question in HELD[relation]]
        summary[relation] = {'total': len(group), 'routed': sum(r['read_relation'] == relation for r in group),
                             'triggered': sum(r['start'] is not None for r in group),
                             'strict': sum(r['strict'] for r in group),
                             'strict_pairs': sum(len(pair) == 2 and all(r['strict'] for r in pair) for pair in pairs),
                             'pairs': len(pairs), 'reference_contains_value': sum(r['expected'] in r['reference_response'] for r in group)}
    for name, group in (('ordinary', [r for r in rows if r['relation'] == 'ordinary']),
                        ('unavailable', [r for r in rows if r['relation'] != 'ordinary' and r['expected'] is None])):
        summary[name] = {'total': len(group), 'no_trigger': sum(r['start'] is None for r in group),
                         'reference_equal': sum(r['reference_tokens_equal'] for r in group)}
    if fingerprint(protected) != hashes:
        raise ValueError('A frozen source changed')
    if replay and (rows != baseline['rows'] or summary != baseline['summary']):
        raise ValueError('Independent replay differs')
    write(out / ('reload.json' if replay else 'result.json'), {'summary': summary, 'rows': rows,
          'candidate_sha256': sha(out / 'candidate.pt'), 'plan_sha256': sha(out / 'plan.json'),
          'exact_replay': replay, 'elapsed_seconds': time.perf_counter() - started,
          'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    main(args.out, args.replay)
