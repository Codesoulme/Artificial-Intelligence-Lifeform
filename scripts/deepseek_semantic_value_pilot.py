"""Align finite internal values to Chinese words while replaying numeric reads."""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_chat_interference import SELECTOR
from scripts.deepseek_full_answer import answer_batch
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_named_binding_probe import KEYS, NAMES, UNKNOWN
from scripts.deepseek_named_incremental_probe import NEW_KEYS
from scripts.deepseek_named_readout_alignment import combined_states, snapshot
from scripts.deepseek_read_strength_probe import ScaledReadout
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.open_base_trajectory_comparison import numeric_correct
from spora.growing import load
from spora.language_training import frozen_digest
from spora.open_base_memory import memory_hooks

OUTPUT = ROOT / 'runs/deepseek-semantic-value-pilot-20261110'
WORDS = ('茶', '咖啡', '牛奶', '果汁', '豆浆', '可乐', '水', '酸奶')
TRAIN_TEMPLATES = ('{name}喜欢喝什么？', '请问{name}偏爱的饮品是什么？',
                   '{name}最喜欢的饮料是？', '告诉我{name}通常喜欢喝的饮品。')
HELD_TEMPLATES = ('{name}喜欢的饮品是什么？', '请告诉我{name}爱喝什么。')
HELD_VALUES = (1, 6)
STEPS = 160


def word_correct(text, expected):
    return text.strip() in (expected, expected + '.', expected + '。')


@torch.no_grad()
def evaluate(base, tokenizer, adapter, bases, states, names, keys):
    rows = []
    for group in ('word', 'numeric'):
        for name, key in zip(names, keys):
            templates = HELD_TEMPLATES if group == 'word' else ('对象{key}的值是多少？',)
            for template in templates:
                question = template.format(name=name, key=key)
                addresses = [basis[key:key+1] for basis in bases]
                inputs = closed_think_inputs(tokenizer, question, 'cuda')
                batch = {k: v.expand(len(HELD_VALUES), -1) for k, v in inputs.items()}
                memory = combined_states([states[(key, value)] for value in HELD_VALUES])
                with memory_hooks([base.model.layers[5], base.model.layers[17]], adapter, memory,
                                  current_only=True, addresses=addresses):
                    generated = base.generate(**batch, max_new_tokens=8, do_sample=False,
                                              pad_token_id=tokenizer.eos_token_id)
                outputs = []
                for item, value in zip(generated, HELD_VALUES):
                    tokens = item[inputs['input_ids'].shape[1]:].tolist()
                    text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    expected = WORDS[value] if group == 'word' else value
                    outputs.append({'expected': expected, 'text': text,
                                    'correct': word_correct(text, expected) if group == 'word' else numeric_correct(text, expected),
                                    'ended_with_eos': tokenizer.eos_token_id in tokens})
                row = {'group': group, 'name': name, 'key': key, 'question': question, 'outputs': outputs}
                if group == 'word':
                    empty = SimpleNamespace(current=[torch.zeros_like(states[(key, HELD_VALUES[0])].current[i])
                                                     for i in range(2)], pages=[])
                    with memory_hooks([base.model.layers[5], base.model.layers[17]], adapter, empty,
                                      current_only=True, addresses=addresses):
                        ablated = base.generate(**inputs, max_new_tokens=8, do_sample=False,
                                               pad_token_id=tokenizer.eos_token_id)
                    row['zero_memory_text'] = tokenizer.decode(ablated[0, inputs['input_ids'].shape[1]:],
                                                               skip_special_tokens=True).strip()
                rows.append(row)
    summary = {}
    for group in ('word', 'numeric'):
        selected = [r for r in rows if r['group'] == group]
        summary[group] = {'pairs': len(selected), 'examples': sum(len(r['outputs']) for r in selected),
                          'correct': sum(item['correct'] for r in selected for item in r['outputs']),
                          'both_correct': sum(all(item['correct'] for item in r['outputs']) for r in selected)}
    return {'summary': summary, 'rows': rows}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    protected = {p: sha(p) for p in (MODEL / 'model.safetensors', INDIVIDUAL, SELECTOR)}
    if protected[MODEL / 'model.safetensors'] != WEIGHT_SHA or protected[INDIVIDUAL] != INDIVIDUAL_SHA:
        raise ValueError('Protected model mismatch')
    parent = torch.load(SELECTOR, map_location='cpu', weights_only=True)
    if (parent.get('format') != 'deepseek-spora-question-pooled-finite-address-v1'
            or parent.get('weight_sha256') != WEIGHT_SHA or parent.get('individual_sha256') != INDIVIDUAL_SHA
            or parent.get('layers') != [5, 17]):
        raise ValueError('Readout provenance mismatch')
    if len(set(WORDS)) != 8 or set(NAMES) & set(UNKNOWN[:2]) or set(TRAIN_TEMPLATES) & set(HELD_TEMPLATES):
        raise ValueError('Vocabulary or train/evaluation split mismatch')
    torch.manual_seed(4634)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(organism, original)
    eval_names = (NAMES[0], NAMES[2], UNKNOWN[0], UNKNOWN[1])
    eval_keys = (KEYS[0], KEYS[2], NEW_KEYS[0], NEW_KEYS[1])
    states = {(key, value): snapshot(organism, key, value) for key in sorted(set(KEYS + eval_keys)) for value in range(8)}
    state_digests = {key: frozen_digest(organism, state) for key, state in states.items()}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    adapter = ScaledReadout(base.config.hidden_size, organism.config.memory_width, 0.5).to('cuda')
    adapter.load_state_dict(parent['readout'], strict=True)
    bases = [F.normalize(site.current_basis.to('cuda').float(), dim=-1) for site in organism.memories]
    adapter.eval()
    print('baseline evaluation starting', flush=True)
    baseline = evaluate(base, tokenizer, adapter, bases, states, eval_names, eval_keys)
    adapter.train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.0002, weight_decay=0.)
    tasks = [(name, key, template, value) for name, key in zip(NAMES, KEYS)
             for template in TRAIN_TEMPLATES for value in range(8)]
    generator = torch.Generator().manual_seed(4634)
    order = torch.randperm(len(tasks), generator=generator).tolist()
    log = []
    for step in range(STEPS):
        name, key, template, value = tasks[order[step // 2]]
        other = (value + 1) % 8
        if step % 2 == 0:
            question = template.format(name=name)
            answers = [WORDS[value], WORDS[other]]
            group = 'word'
        else:
            question = f'对象{key}的值是多少？'
            answers = [value, other]
            group = 'numeric_replay'
        inputs, labels = answer_batch(tokenizer, question, answers, 'cuda')
        memory = combined_states([states[(key, value)], states[(key, other)]])
        addresses = [basis[key:key+1] for basis in bases]
        optimizer.zero_grad(set_to_none=True)
        with memory_hooks([base.model.layers[5], base.model.layers[17]], adapter, memory,
                          current_only=True, addresses=addresses):
            logits = base(**inputs, use_cache=False).logits[:, :-1, :].float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step in (0, 1) or (step + 1) % 20 == 0 or (step + 2) % 20 == 0:
            entry = {'step': step + 1, 'group': group, 'loss': float(loss.detach()), 'gradient_norm': float(norm)}
            log.append(entry)
            print(json.dumps(entry), flush=True)
    adapter.eval()
    if (any(p.grad is not None for p in base.parameters()) or frozen_digest(organism, original) != before
            or any(frozen_digest(organism, state) != state_digests[key] for key, state in states.items())
            or any(sha(p) != d for p, d in protected.items())):
        raise ValueError('Protected input changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT / 'adapter.pt'
    torch.save({'format': 'deepseek-spora-semantic-value-development-v1',
                'state': {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA,
                'parent_sha256': protected[SELECTOR], 'layers': [5, 17], 'strength': 0.5}, checkpoint)
    print('adapted checkpoint saved; evaluation starting', flush=True)
    adapted = evaluate(base, tokenizer, adapter, bases, states, eval_names, eval_keys)
    if (frozen_digest(organism, original) != before
            or any(frozen_digest(organism, state) != state_digests[key] for key, state in states.items())
            or any(sha(p) != d for p, d in protected.items())):
        raise ValueError('Protected input changed during evaluation')
    report = {'status': 'finite_semantic_value_development_only', 'weight_sha256': WEIGHT_SHA,
              'individual_sha256': INDIVIDUAL_SHA, 'parent_sha256': protected[SELECTOR],
              'adapter_sha256': sha(checkpoint), 'steps': STEPS, 'learning_rate': 0.0002,
              'train_names': list(NAMES), 'held_names': list(UNKNOWN[:2]), 'vocabulary': WORDS,
              'held_values': HELD_VALUES, 'log': log, 'baseline': baseline, 'adapted': adapted,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Eight manually defined fixed value-word associations, not arbitrary natural-language fact memory.',
                              'The paired state values and correct addresses are provided by the experiment, not inferred from statements.',
                              'Only the readout adapter trains; DeepSeek and the original individual are frozen.',
                              'Two held names and two templates are small synthetic development samples.',
                              'All eight value-word mappings were seen in training; unseen values/relations are not tested.',
                              'Closed-thinking greedy generation and eight-token cap; not open conversation or reasoning.',
                              'Zero-matrix ablation is not a reliable unknown-state mechanism.',
                              'No automatic writes, semantic routing, independently trained individuals or crash recovery.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'baseline': baseline['summary'], 'adapted': adapted['summary']}), flush=True)


if __name__ == '__main__':
    run()
