"""Train a small frozen-feature temporal intent router and test integrated recall."""
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_integrated_checkpoint_probe import READOUT
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_semantic_value_pilot import WORDS, word_correct
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import load, query
from spora.integrated_checkpoint import load_integrated, save_integrated
from spora.integrated_open_base import ScaledOpenBaseMemory
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256
from spora.temporal_routing import CURRENT, LATER, PAST, route_spec

OUTPUT = ROOT / 'runs/deepseek-temporal-intent-router-predicted-route-20261110'
TRAIN = {
    CURRENT: (f'{NAMES[0]}现在喜欢的饮品是什么？', f'{NAMES[0]}目前偏爱喝什么？',
              f'我记得{NAMES[0]}喜欢哪种饮料？', f'{NAMES[0]}当前最爱喝什么饮品？'),
    PAST: (f'{NAMES[0]}过去喜欢的饮品是什么？', f'{NAMES[0]}以前最爱喝什么？',
           f'更正之前，{NAMES[0]}喜欢哪种饮料？', f'{NAMES[0]}原来偏爱喝什么饮品？'),
    LATER: (f'{NAMES[0]}后来喜欢的饮品是什么？', f'更正之后，{NAMES[0]}最爱喝什么？',
            f'{NAMES[0]}之后改成偏爱什么饮料？', f'{NAMES[0]}更新后喜欢哪种饮品？'),
}
HELD = {
    CURRENT: (f'{NAMES[0]}现阶段爱喝哪种饮料？', f'截至目前，{NAMES[0]}的饮品偏好是什么？',
              f'现在记录的{NAMES[0]}最爱喝什么？'),
    PAST: (f'{NAMES[0]}原先爱喝哪种饮料？', f'变更以前，{NAMES[0]}的饮品偏好是什么？',
           f'之前记录的{NAMES[0]}最爱喝什么？'),
    LATER: (f'{NAMES[0]}变更以后爱喝哪种饮料？', f'后续记录中，{NAMES[0]}的饮品偏好是什么？',
            f'修改完成后，{NAMES[0]}最爱喝什么？'),
}


def examples(templates):
    return [{'text': text, 'label': label} for label, choices in templates.items() for text in choices]


def score(result, expected):
    return {'router_accuracy': sum(row['predicted_slot'] == row['label'] for row in result) / len(result),
            'native_accuracy': sum(row['native_value'] == row['expected_value'] for row in result) / len(result),
            'answer_accuracy': sum(row['answer_correct'] for row in result) / len(result),
            'examples': len(result)}


def current_addresses(organism, key, device):
    return [F.normalize(site.current_basis[key:key + 1].to(device).float(), dim=-1)
            for site in organism.memories]


def temporal_addresses(organism, key, event_time, device):
    keys = torch.tensor([key], dtype=torch.long, device=device)
    modes = torch.ones_like(keys)
    times = torch.tensor([event_time], dtype=torch.long, device=device)
    _, bound = organism.representation(keys, modes, times)
    return [organism.memory_address(site, bound, keys, modes) for site in organism.memories]


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = {'weight': MODEL / 'model.safetensors', 'original': INDIVIDUAL,
             'adapter': READOUT / 'adapter.pt', 'adapter_report': READOUT / 'result.json'}
    before = {name: sha(path) for name, path in files.items()}
    if before['weight'] != WEIGHT_SHA or before['original'] != INDIVIDUAL_SHA:
        raise ValueError('Protected source mismatch')
    with files['adapter_report'].open(encoding='utf-8') as stream:
        if json.load(stream)['adapter_sha256'] != before['adapter']:
            raise ValueError('Reader provenance mismatch')
    adapter_data = torch.load(files['adapter'], map_location='cpu', weights_only=True)
    if (adapter_data.get('format') != 'deepseek-spora-semantic-value-development-v1'
            or adapter_data.get('weight_sha256') != WEIGHT_SHA or adapter_data.get('individual_sha256') != INDIVIDUAL_SHA
            or adapter_data.get('layers') != [5, 17] or adapter_data.get('strength') != 0.5):
        raise ValueError('Adapter provenance mismatch')
    config_sha = feature_files_sha256(MODEL)
    torch.manual_seed(4875)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, source_state, _ = load(INDIVIDUAL, 'cuda')
    original_digest = frozen_digest(organism, source_state)
    state = fresh_controlled_state(organism)
    key = KEYS[0]
    memory_device = next(organism.parameters()).device
    events = []
    for value in (1, 6):
        state, decision = observe_sourced(organism, state, key, value, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('Controlled fact was not written')
        events.append({'time': state.cursor - 1, 'value': value, 'action': decision['action']})
    for event_time, expected in ((0, 1), (1, 6)):
        q = torch.tensor([key], device=memory_device)
        times = torch.tensor([event_time], device=memory_device)
        if int(query(organism, state, q, torch.ones_like(q), times).argmax(-1)) != expected:
            raise ValueError('Native historical memory does not match controlled events')
    train, held = examples(TRAIN), examples(HELD)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    features = {}
    for split, rows in (('train', train), ('held', held)):
        hidden, mask = extract(base, tokenizer, [row['text'] for row in rows], batch_size=2)
        features[split] = pool(hidden, mask).float()
        print(f'features {split}: {len(rows)}', flush=True)
    labels = torch.tensor([row['label'] for row in train], dtype=torch.long)
    torch.manual_seed(4876)
    router = nn.Linear(features['train'].shape[-1], 3)
    optimizer = torch.optim.AdamW(router.parameters(), lr=0.003, weight_decay=0.02)
    losses = []
    for step in range(600):
        optimizer.zero_grad(set_to_none=True)
        logits = router(features['train'])
        loss = F.cross_entropy(logits, labels) + 0.02 * router.weight.square().sum()
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite temporal router loss')
        loss.backward()
        optimizer.step()
        if step in (0, 99, 299, 599):
            losses.append({'step': step + 1, 'loss': float(loss.detach())})
    with torch.no_grad():
        train_predictions = router(features['train']).argmax(-1).tolist()
        held_predictions = router(features['held']).argmax(-1).tolist()
    train_rows = [{**row, 'predicted_slot': prediction} for row, prediction in zip(train, train_predictions)]
    bases = [F.normalize(site.current_basis.to('cuda').float(), dim=-1) for site in organism.memories]
    reader = ScaledOpenBaseMemory(adapter_data['state']['query.0.weight'].shape[1], organism.config.memory_width, 0.5)
    reader.load_state_dict(adapter_data['state'], strict=True)
    checkpoint = OUTPUT / 'historical-individual.pt'
    OUTPUT.mkdir(parents=False, exist_ok=False)
    save_integrated(checkpoint, organism, state, reader, MODEL, (5, 17))
    checkpoint_sha = sha(checkpoint)
    inputs = [closed_think_inputs(tokenizer, row['text'], 'cuda') for row in held]
    max_len = max(item['input_ids'].shape[1] for item in inputs)
    batch = {name: torch.cat([F.pad(item[name], (max_len - item[name].shape[1], 0),
                                    value=tokenizer.pad_token_id if name == 'input_ids' else 0)
                              for item in inputs], dim=0) for name in ('input_ids', 'attention_mask')}
    with torch.inference_mode():
        predicted = router(features['held']).argmax(-1).tolist()
        row_addresses, row_slots, expected_values = [], [], []
        for row, intent in zip(held, predicted):
            label = row['label']
            expected_values.append(6 if label in (CURRENT, LATER) else 1)
            slot, event_time = route_spec(intent)
            row_slots.append(slot)
            if event_time is None:
                row_addresses.append(current_addresses(organism, key, 'cuda'))
            else:
                row_addresses.append(temporal_addresses(organism, key, event_time, 'cuda'))
        addresses = [torch.cat([item[site] for item in row_addresses]) for site in range(2)]
        base_model, loaded_organism, loaded_state = load_integrated(checkpoint, base, MODEL, individual_device='cuda')
        generations = base_model.generate(**batch, memory_state=loaded_state, memory_addresses=addresses,
                                          memory_slots=torch.tensor(row_slots), current_only=False,
                                          max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    results = []
    for i, row in enumerate(held):
        tokens = generations[i, max_len:].tolist()
        answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        native = predicted[i]
        answer_correct = word_correct(answer, WORDS[expected_values[i]])
        results.append({'question': row['text'], 'expected_intent': row['label'],
                        'predicted_intent': native, 'router_correct': native == row['label'],
                        'expected_value': expected_values[i], 'answer': answer,
                        'answer_correct': answer_correct, 'ended_with_eos': tokenizer.eos_token_id in tokens})
    if (any(sha(path) != before[name] for name, path in files.items()) or sha(checkpoint) != checkpoint_sha
            or feature_files_sha256(MODEL) != config_sha or frozen_digest(organism, source_state) != original_digest):
        raise ValueError('Protected inputs changed during temporal route experiment')
    report = {'status': 'learned_temporal_address_router_development_only', 'events': events,
              'training_rows': train_rows, 'held_rows': results, 'losses': losses,
              'summary': {'train_router_accuracy': sum(row['predicted_slot'] == row['label'] for row in train_rows) / len(train_rows),
                          'held_router_accuracy': sum(row['router_correct'] for row in results) / len(results),
                          'held_answer_accuracy': sum(row['answer_correct'] for row in results) / len(results),
                          'held_examples': len(results)},
              'component_sha256': before, 'checkpoint_sha256': checkpoint_sha,
              'feature_config_sha256': config_sha, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Only 4 synthetic training templates per time class and 3 held templates per class.',
                              'All questions use one known name, one address and two controlled drink values.',
                              'Time route is a trained experimental classifier, not native SPORA temporal reasoning.',
                              'Small held set with one seed; templates are authored by the experimenter.',
                              'No unknown time, uncertain date, multi-event natural dialogue, writes or open chat tested.',
                              'Base and tokenizer remain separate; state persists only in the integrated checkpoint.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    run()
