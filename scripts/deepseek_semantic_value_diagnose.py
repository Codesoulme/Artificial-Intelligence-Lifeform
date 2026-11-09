"""Diagnose whether finite semantic training affects first-token choice logits."""
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
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_named_binding_probe import KEYS, NAMES, UNKNOWN
from scripts.deepseek_named_incremental_probe import NEW_KEYS
from scripts.deepseek_named_readout_alignment import combined_states, snapshot
from scripts.deepseek_read_strength_probe import ScaledReadout
from scripts.deepseek_semantic_value_pilot import HELD_TEMPLATES, OUTPUT as PILOT, TRAIN_TEMPLATES, WORDS
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load
from spora.language_training import frozen_digest
from spora.open_base_memory import memory_hooks

OUTPUT = ROOT / 'runs/deepseek-semantic-value-diagnose-20261110.json'


def training_examples():
    tasks = [(name, key, template, value) for name, key in zip(NAMES, KEYS)
             for template in TRAIN_TEMPLATES for value in range(8)]
    generator = torch.Generator().manual_seed(4634)
    order = torch.randperm(len(tasks), generator=generator).tolist()[:80]
    selected, names = [], set()
    for index in order:
        name, key, template, value = tasks[index]
        if name in names:
            continue
        names.add(name)
        selected.append({'group': 'trained', 'question': template.format(name=name), 'name': name,
                         'key': key, 'values': [value, (value + 1) % 8]})
        if len(selected) == 4:
            break
    if len(selected) != 4:
        raise ValueError('Could not select four distinct trained names')
    return selected


def candidate_metrics(logits, candidate_ids, expected):
    if (logits.ndim != 2 or logits.shape[0] != len(expected) or len(set(candidate_ids)) != len(candidate_ids)
            or any(not 0 <= i < logits.shape[1] for i in candidate_ids) or not bool(torch.isfinite(logits).all())):
        raise ValueError('Invalid choice logits')
    chosen = logits[:, candidate_ids]
    probabilities = chosen.softmax(-1)
    rows = []
    for index, value in enumerate(expected):
        target = candidate_ids[value]
        rows.append({'expected': value, 'best_candidate': int(chosen[index].argmax()),
                     'candidate_correct': int(chosen[index].argmax()) == value,
                     'candidate_probability': float(probabilities[index, value]),
                     'full_vocabulary_rank': int((logits[index] > logits[index, target]).sum()) + 1,
                     'full_vocabulary_top_token': int(logits[index].argmax()),
                     'full_vocabulary_top_is_target': int(logits[index].argmax()) == target})
    return rows


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    candidate, candidate_report = PILOT / 'adapter.pt', PILOT / 'result.json'
    protected = {p: sha(p) for p in (MODEL / 'model.safetensors', INDIVIDUAL, SELECTOR, candidate, candidate_report)}
    if protected[MODEL / 'model.safetensors'] != WEIGHT_SHA or protected[INDIVIDUAL] != INDIVIDUAL_SHA:
        raise ValueError('Protected model mismatch')
    with candidate_report.open(encoding='utf-8') as stream:
        prior = json.load(stream)
    parent = torch.load(SELECTOR, map_location='cpu', weights_only=True)
    adapted = torch.load(candidate, map_location='cpu', weights_only=True)
    if (prior['adapter_sha256'] != protected[candidate] or prior['parent_sha256'] != protected[SELECTOR]
            or parent.get('format') != 'deepseek-spora-question-pooled-finite-address-v1'
            or adapted.get('format') != 'deepseek-spora-semantic-value-development-v1'
            or any(d.get('weight_sha256') != WEIGHT_SHA or d.get('individual_sha256') != INDIVIDUAL_SHA
                   or d.get('layers') != [5, 17] for d in (parent, adapted))):
        raise ValueError('Adapter provenance mismatch')
    cases = training_examples()
    cases += [{'group': 'held', 'question': template.format(name=name), 'name': name, 'key': key, 'values': [1, 6]}
              for name, key in zip((NAMES[0], NAMES[2], UNKNOWN[0], UNKNOWN[1]),
                                   (KEYS[0], KEYS[2], NEW_KEYS[0], NEW_KEYS[1])) for template in HELD_TEMPLATES]
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    original_digest = frozen_digest(organism, original)
    required = {(case['key'], value) for case in cases for value in case['values']}
    states = {item: snapshot(organism, *item) for item in sorted(required)}
    state_digests = {key: frozen_digest(organism, state) for key, state in states.items()}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    pieces = [tokenizer.encode(word, add_special_tokens=False) for word in WORDS]
    if any(not p for p in pieces) or len({p[0] for p in pieces}) != len(pieces):
        raise ValueError('Candidate words require distinct first tokens')
    candidate_ids = [p[0] for p in pieces]
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    adapters = {}
    for name, data in (('parent', parent['readout']), ('adapted', adapted['state'])):
        adapter = ScaledReadout(base.config.hidden_size, organism.config.memory_width, 0.5).to('cuda').eval()
        adapter.load_state_dict(data, strict=True)
        adapters[name] = adapter
    bases = [F.normalize(site.current_basis.to('cuda').float(), dim=-1) for site in organism.memories]
    rows = []
    with torch.inference_mode():
        for case in cases:
            key, values = case['key'], case['values']
            inputs = closed_think_inputs(tokenizer, case['question'], 'cuda')
            batch = {k: v.expand(len(values), -1) for k, v in inputs.items()}
            memory = combined_states([states[(key, value)] for value in values])
            empty = SimpleNamespace(current=[torch.zeros_like(t) for t in memory.current], pages=[])
            addresses = [basis[key:key+1] for basis in bases]
            results = {}
            for name, adapter, active_state in (('parent', adapters['parent'], memory),
                                                 ('adapted', adapters['adapted'], memory),
                                                 ('zero_memory', adapters['adapted'], empty)):
                with memory_hooks([base.model.layers[5], base.model.layers[17]], adapter, active_state,
                                  current_only=True, addresses=addresses):
                    logits = base(**batch, use_cache=False).logits[:, -1, :].float()
                results[name] = candidate_metrics(logits, candidate_ids, values)
            rows.append(dict(case, results=results))
            print(f'checked {len(rows)}/{len(cases)}', flush=True)
    if (frozen_digest(organism, original) != original_digest
            or any(frozen_digest(organism, state) != state_digests[key] for key, state in states.items())
            or any(sha(p) != d for p, d in protected.items())):
        raise ValueError('Protected input changed')
    summary = {}
    for group in ('trained', 'held'):
        selected = [row for row in rows if row['group'] == group]
        summary[group] = {}
        for mode in ('parent', 'adapted', 'zero_memory'):
            all_results = [r for row in selected for r in row['results'][mode]]
            summary[group][mode] = {'examples': len(all_results),
                                   'candidate_correct': sum(r['candidate_correct'] for r in all_results),
                                   'both_correct': sum(all(r['candidate_correct'] for r in row['results'][mode]) for row in selected),
                                   'full_vocabulary_top1': sum(r['full_vocabulary_top_is_target'] for r in all_results),
                                   'mean_target_candidate_probability': sum(r['candidate_probability'] for r in all_results) / len(all_results)}
    report = {'status': 'semantic_first_token_diagnosis_only', 'weight_sha256': WEIGHT_SHA,
              'individual_sha256': INDIVIDUAL_SHA, 'parent_sha256': protected[SELECTOR],
              'adapter_sha256': protected[candidate], 'candidate_token_ids': candidate_ids,
              'candidate_token_lengths': [len(p) for p in pieces], 'summary': summary, 'rows': rows,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Restricted eight-candidate first-token ranking is diagnostic, not free generation.',
                              'One word has multiple tokens; first-token success does not imply the complete word or EOS.',
                              'Four actual training questions and eight held questions; all are development templates.',
                              'Held names were not used in this pilot, but names and values were used elsewhere in the project.',
                              'Addresses and value-to-word mapping are externally supervised, not inferred.',
                              'Zero matrices are an ablation, not a proper unknown-state mechanism.',
                              'No chat preservation, natural language fact writes, history or open-world safety evaluated.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    run()
