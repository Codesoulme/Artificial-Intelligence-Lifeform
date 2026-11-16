"""Compare native SPORA layer replacements after interleaved entity updates.

Entity addresses and event times are explicitly supplied. This development
probe does not test autonomous natural-language memory routing.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_memory_answer_distill import LAYERS, _load_candidate
from scripts.deepseek_spora_memory_answer_distill_v4 import VALUE_PAIRS
from scripts.deepseek_spora_online_correction_answer_probe import SOURCE_CHECKPOINT, CANDIDATES, _query
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.integrated_checkpoint import IntegratedSPORARuntime, load_integrated

OUT = ROOT / 'runs/deepseek-spora-interleaved-native-20261116.json'
INDICES = (4, 5, 6, 7)


def schedule(indices=INDICES):
    if not indices or len(set(indices)) != len(indices) or any(index < 0 or index >= len(NAMES) for index in indices):
        raise ValueError('Distinct valid entity indices required')
    return [(index, VALUE_PAIRS[index][phase], phase)
            for phase in (0, 1) for index in indices]


def main(indices=INDICES, output=OUT):
    if output.exists():
        raise FileExistsError(output)
    schedule(indices)
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    protected = {'base': MODEL / 'model.safetensors', 'integrated': SOURCE_CHECKPOINT,
                 **{f'layer_{layer}': CANDIDATES / f'layer-{layer}.pt' for layer in LAYERS}}
    hashes = {key: sha(path) for key, path in protected.items()}
    if hashes['base'] != WEIGHT_SHA:
        raise ValueError('Base digest mismatch')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float16,
        attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    decoder, organism, _ = load_integrated(SOURCE_CHECKPOINT, model, MODEL, individual_device='cuda')
    runtime = IntegratedSPORARuntime(decoder, organism, fresh_controlled_state(organism), MODEL)
    runtime._probe_blocks = {layer: _load_candidate(CANDIDATES / f'layer-{layer}.pt', layer, model)
                             for layer in LAYERS}
    times, events = {}, []
    for index, value, phase in schedule(indices):
        event_time = runtime.state.cursor
        decision = runtime.observe_sourced(KEYS[index], value, 'direct', 1.0)
        if not decision['current_write'] or runtime.state.cursor != event_time + 1:
            raise ValueError(f'Write rejected for index {index}: {decision}')
        times[(index, phase)] = event_time
        events.append({'name': NAMES[index], 'key': KEYS[index], 'value': WORDS[value],
                       'phase': phase, 'time': event_time, 'action': decision['action']})
    rows = []
    for index in indices:
        for phase, intent, template in ((0, 'past', '最早的记录里{name}喜欢什么饮料？'),
                                        (1, 'current', '更正后{name}现在爱喝什么？')):
            value = VALUE_PAIRS[index][phase]
            for condition in ('base', 'spora_pair'):
                row = _query(runtime, tokenizer, KEYS[index], template.format(name=NAMES[index]),
                             intent, times[(index, phase)], value, condition)
                answer = row['answer'].strip().rstrip('。.!！ ').strip()
                row.update({'name': NAMES[index], 'phase': phase, 'event_time': times[(index, phase)],
                            'exact': answer == WORDS[value],
                            'contains_other_value': WORDS[VALUE_PAIRS[index][1-phase]] in row['answer']})
                rows.append(row)
        print(json.dumps({'completed_entity': NAMES[index]}, ensure_ascii=False), flush=True)
    summary = {}
    for condition in ('base', 'spora_pair'):
        selected = [row for row in rows if row['condition'] == condition]
        summary[condition] = {'total': len(selected), 'exact': sum(row['exact'] for row in selected),
                              'contains': sum(row['correct'] for row in selected),
                              'contains_other': sum(row['contains_other_value'] for row in selected)}
    if any(sha(path) != hashes[key] for key, path in protected.items()):
        raise ValueError('Protected input changed')
    result = {'summary': summary, 'events': events, 'rows': rows, 'source_sha256': hashes,
              'seconds': time.perf_counter() - start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': [f'{len(indices)} previously used named entities, {2 * len(indices)} authored events and fixed question forms.',
                              'Entity identity, memory address and historical event time are supplied by the test.',
                              'Some names and value pairs were used in v4 training; no independent generalization claim.',
                              'Eight-token substring and exact scores are separate, neither is open-dialogue assessment.',
                              'State is isolated and not saved or reloaded; no crash recovery or unknown test.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-entities', action='store_true')
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    main(tuple(range(len(NAMES))) if args.all_entities else INDICES, args.output)
