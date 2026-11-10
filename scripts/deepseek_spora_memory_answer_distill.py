"""Train paired SPORA blocks to answer from native memory on held entities."""
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.open_base_baseline import INDIVIDUAL_SHA
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_block_distill import _hash
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import grow
from spora.integrated_checkpoint import load_integrated
from spora.internal_memory import observe_sourced
from spora.language_training import atomic_save
from spora.structure_transfer import SPORALayerReplacement

SOURCE_CHECKPOINT = ROOT / 'runs/deepseek-temporal-intent-router-native-spora-site-20261110/historical-individual.pt'
START_CANDIDATES = {
    6: ROOT / 'runs/deepseek-spora-joint-memory-distill-v2-20261111/layer-6.pt',
    12: ROOT / 'runs/deepseek-spora-joint-memory-distill-v2-20261111/layer-12.pt',
}
OUTPUT = ROOT / 'runs/deepseek-spora-memory-answer-distill-20261111'
LAYERS = (6, 12)
SEED = 51093
STEPS = 90
TRAIN_TEMPLATES = {
    'current': ('{name}现在喜欢什么饮品？', '目前{name}偏爱喝什么？'),
    'past': ('{name}过去喜欢哪种饮料？', '更正前{name}喜欢喝什么？'),
    'later': ('更正后{name}改喜欢什么饮品？', '{name}后来最喜欢喝什么？'),
}
HELD_TEMPLATES = {
    'current': ('此刻{name}爱喝哪种饮料？', '最近{name}最喜欢什么饮品？'),
    'past': ('在改口以前，{name}常喝什么？', '{name}较早的饮品偏好是什么？'),
    'later': ('更新以后{name}偏爱什么？', '现在记录里{name}喜欢喝哪种饮料？'),
}


def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _load_candidate(path, layer, base):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    meta = payload.get('metadata', {})
    if (meta.get('base_weight_sha256') != WEIGHT_SHA or meta.get('layer') != layer
            or meta.get('format') != 'spora-qwen-layer-replacement-candidate-v1'):
        raise ValueError(f'Candidate provenance mismatch for layer {layer}')
    block = SPORALayerReplacement(base.config.hidden_size, base.config.num_attention_heads,
                                 max_length=128, layer_idx=layer).to('cuda')
    block.load_state_dict(payload['state'], strict=True)
    return block


def _facts(model, state, value_pairs=None, train_indices=None):
    schedule = []
    if value_pairs is None:
        value_pairs = [((index * 2) % len(WORDS), ((index * 2) % len(WORDS) + 3) % len(WORDS))
                       for index in range(8)]
    if len(value_pairs) != 8 or any(len(pair) != 2 or any(type(v) is not int or not 0 <= v < len(WORDS)
                                                          for v in pair) or pair[0] == pair[1]
                                      for pair in value_pairs):
        raise ValueError('Expected eight distinct two-value fact assignments')
    train_indices = set(range(6)) if train_indices is None else set(train_indices)
    if len(train_indices) != 6 or not train_indices <= set(range(8)):
        raise ValueError('Exactly six of eight entities must be assigned to training')
    for index, (name, key) in enumerate(zip(NAMES[:8], KEYS[:8])):
        old, new = value_pairs[index]
        for value in (old, new):
            if state.cursor >= len(state.pages) * model.config.time_count:
                state = grow(model, state)
            state, decision = observe_sourced(model, state, key, value, 'direct', 1.0)
            if not decision['current_write']:
                raise ValueError(f'Fact write rejected for {name}: {decision}')
        schedule.append({'name': name, 'key': key, 'old_value': old, 'new_value': new,
                         'old_time': 2 * index, 'new_time': 2 * index + 1,
                         'train_split': index in train_indices})
    return state, schedule


def _rows(facts, tokenizer):
    training, held = [], []
    for fact in facts:
        split = training if fact['train_split'] else held
        templates = TRAIN_TEMPLATES if fact['train_split'] else HELD_TEMPLATES
        for intent, variants in templates.items():
            event_time = fact['old_time'] if intent == 'past' else fact['new_time']
            expected = fact['old_value'] if intent == 'past' else fact['new_value']
            for variant_index, template in enumerate(variants):
                question = template.format(name=fact['name'])
                prefix = closed_think_inputs(tokenizer, question, 'cuda')
                answer = tokenizer(WORDS[expected], add_special_tokens=False, return_tensors='pt').input_ids
                answer = answer.to('cuda')
                answer = torch.cat((answer, torch.tensor([[tokenizer.eos_token_id]], device='cuda')), dim=1)
                other = fact['old_value'] if intent != 'past' else fact['new_value']
                split.append({'name': fact['name'], 'key': fact['key'], 'intent': intent,
                              'question': question, 'value': expected, 'answer_ids': answer,
                              'other_value': other, 'prefix': prefix,
                              'event_time': event_time, 'variant': variant_index})
    return training, held


def _memory_args(organism, state, row):
    key = row['key']
    keys = torch.tensor([key], dtype=torch.long, device='cuda')
    if row['intent'] == 'current':
        addresses = [torch.nn.functional.normalize(site.current_basis[key:key + 1].to('cuda').float(), dim=-1)
                     for site in organism.memories]
        slot = 0
    else:
        mode = torch.ones_like(keys)
        times = torch.tensor([row['event_time']], dtype=torch.long, device='cuda')
        _, bound = organism.representation(keys, mode, times % organism.config.time_count)
        addresses = [organism.memory_address(site, bound, keys, mode) for site in organism.memories]
        slot = 1
    return {'memory_state': state, 'memory_addresses': addresses,
            'memory_slots': torch.tensor([slot], dtype=torch.long, device='cuda'),
            'current_only': False, 'use_cache': False}


def _set_layers(base, blocks, original):
    for layer in LAYERS:
        base.model.layers[layer] = blocks[layer]


def _evaluate(integrated, base, organism, state, blocks, original, tokenizer, rows):
    _set_layers(base, blocks, original)
    output = []
    try:
        with torch.inference_mode():
            for row in rows:
                prompt = row['prefix']
                generated = integrated.generate(
                    **prompt, **_memory_args(organism, state, row), max_new_tokens=8, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                answer = tokenizer.decode(generated[0, prompt['input_ids'].shape[1]:],
                                          skip_special_tokens=True).strip()
                expected = WORDS[row['value']]
                conflicting_values = [WORDS[row['other_value']]]
                correct = expected in answer and not any(word in answer for word in conflicting_values)
                output.append({'name': row['name'], 'intent': row['intent'],
                               'question': row['question'], 'expected': expected,
                               'answer': answer, 'correct': correct})
    finally:
        for layer in LAYERS:
            base.model.layers[layer] = original[layer]
    return {'correct': sum(row['correct'] for row in output), 'total': len(output),
            'accuracy': sum(row['correct'] for row in output) / len(output),
            'rows': output}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base model hash mismatch')
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float16,
        attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    integrated, organism, initial_state = load_integrated(SOURCE_CHECKPOINT, base, MODEL,
                                                          individual_device='cuda')
    integrated.reader.requires_grad_(False)
    state, fact_schedule = _facts(organism, fresh_controlled_state(organism))
    train_rows, held_rows = _rows(fact_schedule, tokenizer)
    original = {layer: base.model.layers[layer] for layer in LAYERS}
    replacements = {layer: _load_candidate(START_CANDIDATES[layer], layer, base) for layer in LAYERS}
    # Evaluate the actual frozen base first.
    before_base = _evaluate(integrated, base, organism, state, original, original, tokenizer, held_rows)
    # The paired joint candidates start from the most memory-compatible previous round.
    before_candidate = _evaluate(integrated, base, organism, state, replacements, original, tokenizer, held_rows)

    # Frozen base teacher creates compact gold answers on the training entities.
    teacher_rows = []
    with torch.inference_mode():
        for row in train_rows:
            prompt = row['prefix']
            generated = integrated.generate(
                **prompt, **_memory_args(organism, state, row), max_new_tokens=8, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            teacher_text = tokenizer.decode(generated[0, prompt['input_ids'].shape[1]:],
                                            skip_special_tokens=True).strip()
            teacher_rows.append({**row, 'teacher_text': teacher_text})
    # Use ground-truth value tokens as supervised targets; the teacher text is retained for audit.
    targets = []
    for row in train_rows:
        full_ids = torch.cat((row['prefix']['input_ids'], row['answer_ids']), dim=1)
        full_mask = torch.ones_like(full_ids)
        targets.append({'row': row, 'input_ids': full_ids, 'attention_mask': full_mask,
                        'prefix_len': row['prefix']['input_ids'].shape[1],
                        'answer_len': row['answer_ids'].shape[1]})

    parameters = [parameter for module in replacements.values() for parameter in module.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=2e-6, weight_decay=0.01, foreach=False)
    order = list(range(len(targets)))
    rng = random.Random(SEED + 1)
    trace = []
    for step in range(STEPS):
        if step % len(order) == 0:
            rng.shuffle(order)
        item = targets[order[step % len(order)]]
        row = item['row']
        _set_layers(base, replacements, original)
        output = integrated(input_ids=item['input_ids'], attention_mask=item['attention_mask'],
                            **_memory_args(organism, state, row))
        start = item['prefix_len'] - 1
        logits = output.logits[:, start:start + item['answer_len']].float()
        labels = item['input_ids'][:, item['prefix_len']:item['prefix_len'] + item['answer_len']]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        if not bool(torch.isfinite(loss)):
            for layer in LAYERS:
                base.model.layers[layer] = original[layer]
            raise ValueError('Nonfinite memory-answer distillation loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        for layer in LAYERS:
            base.model.layers[layer] = original[layer]
        if step in (0, 9, 19, 39, 59, 89):
            trace.append({'step': step + 1, 'answer_cross_entropy': float(loss.detach())})
    for module in replacements.values():
        module.eval()
    after_candidate = _evaluate(integrated, base, organism, state, replacements, original, tokenizer, held_rows)

    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoints = {}
    for layer, module in replacements.items():
        path = OUTPUT / f'layer-{layer}.pt'
        meta = {'format': 'spora-qwen-layer-replacement-candidate-v1',
                'base_weight_sha256': WEIGHT_SHA, 'layer': layer, 'width': base.config.hidden_size,
                'heads': base.config.num_attention_heads, 'max_length': 128, 'seed': SEED,
                'train_steps': STEPS, 'objective': 'native_memory_answer_cross_entropy'}
        atomic_save({'metadata': meta,
                     'state': {name: tensor.detach().cpu() for name, tensor in module.state_dict().items()}}, path)
        checkpoints[str(layer)] = {'file': path.name, 'sha256': sha(path)}
    report = {
        'status': 'memory_answer_supervised_joint_spora_replacement',
        'base_weight_sha256': WEIGHT_SHA,
        'source_individual_sha256': INDIVIDUAL_SHA,
        'source_integrated_checkpoint_sha256': sha(SOURCE_CHECKPOINT),
        'layers': list(LAYERS),
        'facts': fact_schedule,
        'training_queries': len(train_rows),
        'held_queries': len(held_rows),
        'held_base': before_base,
        'held_candidate_before': before_candidate,
        'held_candidate_after': after_candidate,
        'teacher_training_outputs': [{'name': row['name'], 'intent': row['intent'],
                                      'teacher_text': teacher_rows[i]['teacher_text']}
                                     for i, row in enumerate(train_rows)],
        'loss_trace': trace,
        'candidate_checkpoints': checkpoints,
        'elapsed_seconds': time.perf_counter() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'device': torch.cuda.get_device_name(0),
        'limitations': [
            'Six training entities and two held entities from one fixed name/key list.',
            'One corrected fact per entity; drink values are assigned deterministically.',
            'Held prompts use authored templates and evaluate eight-token greedy generations.',
            'This is a narrow memory-answer learning probe, not open-chat acceptance.',
        ],
    }
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'held_base': before_base['accuracy'],
                      'held_before': before_candidate['accuracy'],
                      'held_after': after_candidate['accuracy']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
