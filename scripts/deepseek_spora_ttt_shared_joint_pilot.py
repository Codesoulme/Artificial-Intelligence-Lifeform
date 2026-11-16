"""Jointly train a reader and writer on shared, corrected fast-memory states.

This is an isolated feasibility experiment. Fact boundaries and event order are
provided by the script; the candidate is never installed into the formal
individual or the DeepSeek base checkpoint.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch.nn.utils.rnn import pad_sequence

from scripts import deepseek_local_baseline as base
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_ttt_joint_writer_reader_pilot import fact_cache
from spora.research_ttt_ntp import (TTTNTPWriteTargetAdapter, attach_read_adapters,
                                    fresh_memory, install, use_memory,
                                    use_memory_with_grad)


OUT = ROOT / 'runs/deepseek-spora-ttt-shared-joint-random-prompts-20261116'
LAYERS = (6, 12)
READ_RANK = 16
WRITE_RANK = 8
STEPS = 256
READER_LR = 7e-4
WRITER_LR = 3e-4
ETA = pilot.ETA
RIDGE = pilot.RIDGE
ALIGNMENT_WEIGHT = 0.02
SEED = 24731

TRAIN = (('小林', '酸奶'), ('阿宁', '豆浆'), ('小雨', '咖啡'), ('阿哲', '果汁'))
HELD = (('小赵', '豆浆'), ('小郑', '咖啡'), ('小王', '酸奶'), ('小孙', '果汁'))
VALUES = ('酸奶', '豆浆', '咖啡', '果汁')
TEMPLATES = ('请只回答{name}喜欢喝什么饮料。', '记录中，{name}爱喝哪一种饮品？')
ALT_TEMPLATE = '请说出{name}偏爱的饮品名称。'
UNKNOWN = ('小雪', '阿松')


def exact(answer, value):
    return value is not None and answer.strip().rstrip('。.!！ ').strip() == value


def sample_episode(rng, step):
    """Vary identity-value bindings; a correction changes one existing value."""
    values = rng.sample(VALUES, len(TRAIN))
    expected = [(name, value) for (name, _), value in zip(TRAIN, values)]
    events = list(expected)
    rng.shuffle(events)
    correction = step % 4 == 3
    if correction:
        index = (step // 4) % len(TRAIN)
        name, old = expected[index]
        replacement = rng.choice([value for value in VALUES if value != old])
        events.append((name, replacement))
        expected[index] = (name, replacement)
    return events, expected, correction


def write_sequence(wrappers, caches, writers, *, differentiable):
    """Write a sequence into one shared fast state, preserving event order."""
    memory = {layer: torch.zeros_like(wrapper.base.weight, dtype=torch.float32)
              for layer, wrapper in wrappers.items()}
    for cache in caches:
        for layer in wrappers:
            key = cache['keys'][layer]
            target = writers[layer](cache['targets'][layer]) if writers is not None else cache['targets'][layer]
            fast = memory[layer]
            for position in range(key.shape[0]):
                current_key = key[position].float()
                current_target = target[position].float()
                prediction = torch.nn.functional.linear(current_key.unsqueeze(0), fast)[0]
                residual = current_target - prediction
                denominator = current_key.square().sum() + RIDGE
                fast = fast + ETA * residual.unsqueeze(1) * current_key.unsqueeze(0) / denominator
            memory[layer] = fast
    if not differentiable:
        memory = {layer: value.detach() for layer, value in memory.items()}
    return memory


def alignment_loss(writers, caches):
    terms = []
    for cache in caches:
        values = cache['value_embeddings']
        for layer, writer in writers.items():
            terms.append(1 - torch.nn.functional.cosine_similarity(
                writer(cache['targets'][layer]), values, dim=-1).mean())
    return torch.stack(terms).mean()


def make_query_batch(tokenizer, cases):
    """Pack teacher-forced answer prompts while retaining each prefix length."""
    rows = []
    for name, value, template in cases:
        question = template.format(name=name)
        prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
        target_ids = tokenizer.encode(value, add_special_tokens=False) + [tokenizer.eos_token_id]
        target = torch.tensor(target_ids, device=prompt['input_ids'].device, dtype=torch.long)
        rows.append((prompt['input_ids'][0], target, question, name, value))
    ids = pad_sequence([torch.cat((prompt, target)) for prompt, target, *_ in rows],
                       batch_first=True, padding_value=tokenizer.pad_token_id)
    mask = torch.zeros_like(ids)
    for row, (prompt, target, *_rest) in enumerate(rows):
        mask[row, :prompt.numel() + target.numel()] = 1
    return {'input_ids': ids, 'attention_mask': mask, 'rows': rows}


def teacher_forced_loss(model, batch, *, return_logits=False):
    result = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                   use_cache=False, return_dict=True)
    losses = []
    for row, (prompt, target, *_rest) in enumerate(batch['rows']):
        prefix = prompt.numel()
        scores = result.logits[row, prefix - 1:prefix - 1 + target.numel()].float()
        losses.append(torch.nn.functional.cross_entropy(scores, target))
    mean_loss = torch.stack(losses).mean()
    return (mean_loss, result.logits) if return_logits else mean_loss


def paired_margin_loss(first_logits, second_logits, old_id, new_id, margin=0.5):
    """Require the same question to favor its own value in each state."""
    if old_id == new_id:
        raise ValueError('Paired facts must have different value tokens')
    first_margin = first_logits[old_id] - first_logits[new_id]
    second_margin = second_logits[new_id] - second_logits[old_id]
    return (torch.nn.functional.softplus(margin - first_margin)
            + torch.nn.functional.softplus(margin - second_margin)) / 2


def change_last_event(events, name, replacement):
    changed = list(events)
    indices = [index for index, (person, _) in enumerate(changed) if person == name]
    if not indices or changed[indices[-1]][1] == replacement:
        raise ValueError('Paired intervention requires an existing different value')
    changed[indices[-1]] = (name, replacement)
    return changed


def evaluate(model, tokenizer, wrappers, writers, caches, cases, *, events=None):
    if events is None:
        events = [(name, value) for name, value, _template in cases]
    memory = write_sequence(wrappers, [caches[key] for key in events], writers,
                            differentiable=False)
    rows = []
    for name, expected, template in cases:
        question = template.format(name=name)
        answer = pilot.generate(model, tokenizer, question, memory)
        rows.append({'name': name, 'expected': expected, 'question': question,
                     'answer': answer, 'exact': exact(answer['answer'], expected)})
    return rows


def main(output=OUT, steps=STEPS, paired_weight=0.0):
    if output.exists():
        raise FileExistsError(output)
    if steps < 1 or not 0 <= paired_weight <= 10:
        raise ValueError('Invalid steps or paired weight')
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    base_path = base.MODEL / 'model.safetensors'
    base_sha = base.sha(base_path)
    if base_sha != base.WEIGHT_SHA:
        raise ValueError('Unexpected base weight')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True,
                                                   trust_remote_code=False)
    model = load_full_streamed().eval()
    wrappers = install(model, LAYERS)
    readers = attach_read_adapters(wrappers, rank=READ_RANK)
    writers = {layer: TTTNTPWriteTargetAdapter(wrappers[layer].base.out_features,
                                                rank=WRITE_RANK).to('cuda')
               for layer in LAYERS}
    model.requires_grad_(False)
    trainable = [parameter for reader in readers.values() for parameter in reader.parameters()]
    trainable += [parameter for writer in writers.values() for parameter in writer.parameters()]
    for parameter in trainable:
        parameter.requires_grad_(True)

    all_people = TRAIN + HELD
    caches = {}
    for name, _ in all_people:
        for value in VALUES:
            caches[(name, value)] = fact_cache(model, tokenizer, wrappers,
                                               f'{name}喜欢喝{value}。', value)
    print(json.dumps({'prepared_states': len(caches)}, ensure_ascii=False), flush=True)

    optimizer = torch.optim.AdamW([
        {'params': [p for reader in readers.values() for p in reader.parameters()], 'lr': READER_LR},
        {'params': [p for writer in writers.values() for p in writer.parameters()], 'lr': WRITER_LR},
    ], weight_decay=1e-4, foreach=False)
    rng = random.Random(SEED)
    trace = []
    ordinary = ('你好，请简单介绍自己。', '请写一句鼓励学习的话。')
    ordinary_before = [pilot.generate(model, tokenizer, question) for question in ordinary]

    for step in range(steps):
        events, expected, correction = sample_episode(rng, step)
        event_caches = [caches[item] for item in events]
        query_cases = [(name, value, rng.choice(TEMPLATES)) for name, value in expected]
        batch = make_query_batch(tokenizer, query_cases)
        optimizer.zero_grad(set_to_none=True)
        memory = write_sequence(wrappers, event_caches, writers, differentiable=True)
        with use_memory_with_grad(memory):
            loss, first_logits = teacher_forced_loss(model, batch, return_logits=True)
        paired = torch.zeros((), device=loss.device)
        if paired_weight:
            target_index = rng.randrange(len(expected))
            name, old_value = expected[target_index]
            new_value = rng.choice([value for value in VALUES if value != old_value])
            paired_events = change_last_event(events, name, new_value)
            paired_memory = write_sequence(wrappers, [caches[item] for item in paired_events], writers,
                                           differentiable=True)
            paired_batch = make_query_batch(tokenizer, [(name, new_value, query_cases[target_index][2])])
            with use_memory_with_grad(paired_memory):
                _paired_ce, second_logits = teacher_forced_loss(model, paired_batch, return_logits=True)
            old_id, new_id = tokenizer.encode(old_value, add_special_tokens=False), tokenizer.encode(new_value, add_special_tokens=False)
            if len(old_id) != 1 or len(new_id) != 1:
                raise ValueError('Pair objective requires single-token values')
            first_pos = batch['rows'][target_index][0].numel() - 1
            second_pos = paired_batch['rows'][0][0].numel() - 1
            paired = paired_margin_loss(first_logits[target_index, first_pos].float(),
                                        second_logits[0, second_pos].float(), old_id[0], new_id[0])
        align = alignment_loss(writers, event_caches)
        regularization = 1e-5 * sum(parameter.square().mean() for parameter in trainable)
        total = loss + ALIGNMENT_WEIGHT * align + regularization + paired_weight * paired
        if not bool(torch.isfinite(total)):
            raise ValueError('Nonfinite shared joint loss')
        total.backward()
        grad = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
        optimizer.step()
        row = {'step': step + 1, 'loss': float(loss.detach()), 'alignment': float(align.detach()),
               'total': float(total.detach()), 'paired': float(paired.detach()),
               'gradient_norm': float(grad), 'correction': correction}
        trace.append(row)
        if (step + 1) % 16 == 0:
            print(json.dumps(row), flush=True)

    output.mkdir()
    payload = {'format': 'spora-ttt-shared-joint-v1', 'base_sha256': base_sha,
               'layers': list(LAYERS), 'read_rank': READ_RANK, 'write_rank': WRITE_RANK,
               'steps': steps, 'seed': SEED, 'alignment_weight': ALIGNMENT_WEIGHT,
               'paired_weight': paired_weight,
               'readers': {str(layer): {name: tensor.detach().cpu()
                                        for name, tensor in reader.state_dict().items()}
                           for layer, reader in readers.items()},
               'writers': {str(layer): {name: tensor.detach().cpu()
                                        for name, tensor in writer.state_dict().items()}
                           for layer, writer in writers.items()}}
    torch.save(payload, output / 'candidate.pt')
    (output / 'training-trace.json').write_text(json.dumps(trace, indent=2) + '\n', encoding='utf-8')
    eval_cases = [(name, value, TEMPLATES[0]) for name, value in TRAIN]
    held_cases = [(name, value, TEMPLATES[0]) for name, value in HELD]
    alternate_cases = [(name, VALUES[(VALUES.index(value) + 1) % len(VALUES)], TEMPLATES[0])
                       for name, value in TRAIN]
    eval_cache = {(name, value): caches[(name, value)] for name, value in all_people for value in VALUES}
    # Use the same trained writer for an unseen person and for a changed value.
    train_rows = evaluate(model, tokenizer, wrappers, writers, eval_cache,
                          eval_cases)
    held_rows = evaluate(model, tokenizer, wrappers, writers, eval_cache,
                         held_cases)
    alternate_rows = evaluate(model, tokenizer, wrappers, writers, eval_cache,
                              alternate_cases)
    correction_events = [(name, value) for name, value in TRAIN]
    correction_name, correction_old = TRAIN[0]
    correction_value = VALUES[(VALUES.index(correction_old) + 1) % len(VALUES)]
    correction_events.append((correction_name, correction_value))
    correction_cases = [(name, correction_value if name == correction_name else value, TEMPLATES[0])
                        for name, value in TRAIN]
    correction_rows = evaluate(model, tokenizer, wrappers, writers,
                               eval_cache, correction_cases, events=correction_events)
    unknown = []
    memory = write_sequence(wrappers, [eval_cache[item] for item in TRAIN], writers,
                            differentiable=False)
    for name in UNKNOWN:
        answer = pilot.generate(model, tokenizer, f'请只回答{name}喜欢喝什么饮料。', memory)
        unknown.append({'name': name, 'answer': answer})
    ordinary_after = [pilot.generate(model, tokenizer, question) for question in ordinary]
    if ordinary_before != ordinary_after:
        raise ValueError('Memory-off ordinary control changed')
    if base.sha(base_path) != base_sha:
        raise ValueError('Base model changed')

    summary = {
        'train': {'cases': len(train_rows), 'exact': sum(row['exact'] for row in train_rows)},
        'held_name': {'cases': len(held_rows), 'exact': sum(row['exact'] for row in held_rows)},
        'train_changed_value': {'cases': len(alternate_rows), 'exact': sum(row['exact'] for row in alternate_rows)},
        'correction': {'cases': len(correction_rows), 'exact': sum(row['exact'] for row in correction_rows)},
        'unknown': {'cases': len(unknown), 'nonempty': sum(bool(row['answer']['answer']) for row in unknown)},
    }
    report = {'status': 'ttt_shared_joint_random_bindings_pilot', 'summary': summary,
              'train_rows': train_rows, 'held_rows': held_rows,
              'train_changed_value_rows': alternate_rows, 'correction_rows': correction_rows,
              'unknown': unknown, 'trace': trace, 'base_sha256': base_sha,
              'candidate_sha256': base.sha(output / 'candidate.pt'), 'steps': steps,
              'paired_weight': paired_weight,
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Fact boundaries and event order are supplied by the experiment.',
                              'Training uses four synthetic people and four beverage values in random bindings.',
                              'Held people, correction, unknown rejection and open dialogue are not sealed benchmarks.',
                              'The candidate is separate from the formal individual and base checkpoint.']}
    (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                     encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--steps', type=int, default=STEPS)
    parser.add_argument('--paired-weight', type=float, default=0.0)
    args = parser.parse_args()
    main(args.output, args.steps, args.paired_weight)
