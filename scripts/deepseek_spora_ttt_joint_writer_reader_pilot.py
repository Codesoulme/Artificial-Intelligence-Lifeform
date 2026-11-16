"""Jointly train a TTT write-target correction and its language reader.

The fact boundary is supplied by the experiment and only the selected value
positions are written.  This is an isolated feasibility pilot: it does not
alter the base model, formal individual, or persistent checkpoint.
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

from scripts import deepseek_local_baseline as p
from scripts import deepseek_spora_ttt_crossed_values_pilot as crossed
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.research_ttt_ntp import (TTTNTPWriteTargetAdapter, attach_read_adapters,
                                    fresh_memory, install, sequential_ntp_write,
                                    use_memory, use_memory_with_grad)


OUT = ROOT / 'runs/deepseek-spora-ttt-joint-writer-reader-20261116'
LAYERS = (6, 12)
READ_RANK = 16
WRITE_RANK = 8
STEPS = 192
LR = 5e-4
ETA = pilot.ETA
RIDGE = pilot.RIDGE
SEED = 81731
ALT_TEMPLATE = '请说出{name}偏爱的饮品名称。'


def exact(answer, value):
    return answer.strip().rstrip('。.!！ ').strip() == value


def fact_cache(model, tokenizer, wrappers, fact, value):
    encoded = tokenizer(fact, return_tensors='pt', add_special_tokens=False,
                        return_offsets_mapping=True)
    offsets = encoded.pop('offset_mapping')[0].tolist()
    start = fact.index(value)
    positions = [i - 1 for i, (left, right) in enumerate(offsets)
                 if i > 0 and left < start + len(value) and right > start]
    if not positions:
        raise ValueError('No aligned value token positions')
    inputs = {key: tensor.to('cuda') for key, tensor in encoded.items()}
    captured = {}
    hooks = []
    for layer, wrapper in wrappers.items():
        def capture(module, args, output, layer=layer):
            captured[layer] = args[0][0].detach().float()
        hooks.append(wrapper.base.register_forward_hook(capture))
    try:
        with torch.inference_mode():
            result = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    finally:
        for hook in hooks:
            hook.remove()
    indices = torch.tensor(positions, device='cuda', dtype=torch.long)
    return {
        'keys': {layer: captured[layer].index_select(0, indices) for layer in wrappers},
        'targets': {layer: result.hidden_states[layer + 1][0].detach().float()
                    .index_select(0, indices + 1) for layer in wrappers},
        'value_embeddings': model.model.embed_tokens(inputs['input_ids'][0].index_select(0, indices + 1))
                           .detach().float(),
        'positions': positions,
    }


def write_memory(wrappers, cache, writers, *, differentiable):
    result = {}
    for layer, wrapper in wrappers.items():
        keys = cache['keys'][layer]
        targets = cache['targets'][layer]
        fast = torch.zeros_like(wrapper.base.weight, dtype=torch.float32)
        for index in range(keys.shape[0]):
            key = keys[index].float()
            target = writers[layer](targets[index]) if writers is not None else targets[index]
            prediction = torch.nn.functional.linear(key.unsqueeze(0), fast)[0]
            residual = target.float() - prediction
            denominator = key.square().sum() + RIDGE
            fast = fast + ETA * residual.unsqueeze(1) * key.unsqueeze(0) / denominator
        if not differentiable:
            fast = fast.detach()
        result[layer] = fast
    return result


def value_alignment_loss(writers, cache):
    values = cache['value_embeddings']
    return torch.stack([
        (1 - torch.nn.functional.cosine_similarity(writer(cache['targets'][layer]), values, dim=-1)).mean()
        for layer, writer in writers.items()
    ]).mean()


def main(output=OUT, value_alignment_weight=0.0):
    if output.exists():
        raise FileExistsError(output)
    if not 0 <= value_alignment_weight <= 10:
        raise ValueError('Alignment weight must be between 0 and 10')
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Unexpected base weight')
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True,
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

    caches = {}
    for name in crossed.TRAIN_NAMES + crossed.HELD_NAMES:
        for value in crossed.TRAIN_VALUES + crossed.HELD_VALUES:
            caches[(name, value)] = fact_cache(
                model, tokenizer, wrappers, f'{name}喜欢喝{value}。', value)
        print(json.dumps({'prepared': name, 'states': len(caches)}, ensure_ascii=False), flush=True)

    train = []
    for name in crossed.TRAIN_NAMES:
        for value in crossed.TRAIN_VALUES:
            for template in crossed.TEMPLATES:
                question = template.format(name=name)
                prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
                target_ids = tokenizer.encode(value, add_special_tokens=False) + [tokenizer.eos_token_id]
                target = torch.tensor([target_ids], device='cuda', dtype=torch.long)
                train.append({'name': name, 'value': value, 'prompt': prompt,
                              'ids': torch.cat((prompt['input_ids'], target), dim=1)})

    def eval_memory(name, value):
        return write_memory(wrappers, caches[(name, value)], writers, differentiable=False)

    def generate(question, memory=None):
        return pilot.generate(model, tokenizer, question, memory)

    ordinary = ('你好，请简单介绍自己。', '请写一句鼓励学习的话。')
    ordinary_before = [generate(question) for question in ordinary]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4, foreach=False)
    order = list(range(len(train)))
    trace = []
    for step in range(STEPS):
        if step % len(order) == 0:
            random.shuffle(order)
        item = train[order[step % len(order)]]
        prefix = item['prompt']['input_ids'].shape[1]
        target = item['ids'][:, prefix:]
        optimizer.zero_grad(set_to_none=True)
        memory = write_memory(wrappers, caches[(item['name'], item['value'])], writers,
                              differentiable=True)
        with use_memory_with_grad(memory):
            result = model(input_ids=item['ids'], attention_mask=torch.ones_like(item['ids']),
                           use_cache=False, return_dict=True)
        scores = result.logits[:, prefix - 1:prefix - 1 + target.shape[1]].float()
        loss = torch.nn.functional.cross_entropy(scores.reshape(-1, scores.shape[-1]),
                                                 target.reshape(-1))
        regularization = 1e-5 * sum(parameter.square().mean() for parameter in trainable)
        alignment = value_alignment_loss(writers, caches[(item['name'], item['value'])])
        total = loss + regularization + value_alignment_weight * alignment
        if not bool(torch.isfinite(total)):
            raise ValueError('Nonfinite joint writer loss')
        total.backward()
        gradient = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
        optimizer.step()
        trace.append({'step': step + 1, 'loss': float(loss.detach()),
                      'regularization': float(regularization.detach()),
                      'value_alignment': float(alignment.detach()),
                      'gradient_norm': float(gradient)})
        if (step + 1) % 16 == 0:
            print(json.dumps(trace[-1]), flush=True)

    rows = []
    groups = (
        ('train_pair', crossed.TRAIN_NAMES, crossed.TRAIN_VALUES, crossed.TEMPLATES[0]),
        ('held_name', crossed.HELD_NAMES, crossed.TRAIN_VALUES, crossed.TEMPLATES[0]),
        ('held_value', crossed.HELD_NAMES, crossed.HELD_VALUES, crossed.TEMPLATES[0]),
        ('held_question', crossed.HELD_NAMES, crossed.TRAIN_VALUES, ALT_TEMPLATE),
    )
    for group, names, values, template in groups:
        for name in names:
            question = template.format(name=name)
            baseline = generate(question)
            for value in values:
                answer = generate(question, eval_memory(name, value))
                rows.append({'group': group, 'name': name, 'value': value,
                             'question': question, 'answer': answer,
                             'no_memory': baseline,
                             'exact': exact(answer['answer'], value),
                             'baseline_exact': exact(baseline['answer'], value)})
            print(json.dumps({'evaluated': group, 'name': name}, ensure_ascii=False), flush=True)

    ordinary_after = [generate(question) for question in ordinary]
    if ordinary_before != ordinary_after:
        raise ValueError('Memory-off ordinary control changed')
    if p.sha(p.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Base model changed')
    summary = {}
    for group, names, _, _ in groups:
        subset = [row for row in rows if row['group'] == group]
        summary[group] = {'cases': len(subset), 'exact': sum(row['exact'] for row in subset),
                          'baseline_exact': sum(row['baseline_exact'] for row in subset),
                          'all_values_correct': sum(all(row['exact'] for row in subset
                                                        if row['name'] == name) for name in names),
                          'names': len(names)}

    output.mkdir()
    payload = {
        'format': 'spora-ttt-joint-writer-reader-v1', 'base_sha256': base_sha,
        'layers': list(LAYERS), 'read_rank': READ_RANK, 'write_rank': WRITE_RANK,
        'eta': ETA, 'ridge': RIDGE, 'steps': STEPS, 'learning_rate': LR, 'seed': SEED,
        'value_alignment_weight': value_alignment_weight,
        'readers': {str(layer): {name: tensor.detach().cpu()
                                 for name, tensor in reader.state_dict().items()}
                    for layer, reader in readers.items()},
        'writers': {str(layer): {name: tensor.detach().cpu()
                                 for name, tensor in writer.state_dict().items()}
                    for layer, writer in writers.items()},
    }
    torch.save(payload, output / 'candidate.pt')
    candidate_sha = p.sha(output / 'candidate.pt')
    report = {'status': 'ttt_joint_writer_reader_pilot', 'summary': summary, 'rows': rows,
              'trace': trace, 'train_names': crossed.TRAIN_NAMES, 'held_names': crossed.HELD_NAMES,
              'train_values': crossed.TRAIN_VALUES, 'held_values': crossed.HELD_VALUES,
              'base_sha256': base_sha, 'candidate_sha256': candidate_sha,
              'value_alignment_weight': value_alignment_weight,
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': [
                  'Value boundaries are supplied by the experiment; routing is not learned.',
                  'Only a low-rank correction to the next-state target is trained.',
                  'Synthetic one-fact beverage records and short exact answers only.',
                  'No persistent reload, correction, unknown-fact refusal, or open-domain chat.',
              ]}
    (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                     encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--value-alignment-weight', type=float, default=0.0)
    args = parser.parse_args()
    main(args.output, args.value_alignment_weight)
