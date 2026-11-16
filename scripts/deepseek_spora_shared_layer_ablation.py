"""Matched multi-person fast-memory layer ablation on a fixed development set."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from scripts import deepseek_local_baseline as base
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_ttt_joint_writer_reader_pilot import fact_cache
from scripts.deepseek_spora_ttt_shared_joint_pilot import write_sequence
from scripts.deepseek_spora_shared_joint_counterfactual import SOURCE, PEOPLE, BASE_VALUES
from spora.research_ttt_ntp import TTTNTPWriteTargetAdapter, attach_read_adapters, install, use_memory

OUT = ROOT / 'runs/deepseek-spora-shared-layer-ablation-20261116.json'


def intervention(index):
    if type(index) is not int or not 0 <= index < len(PEOPLE):
        raise ValueError('Invalid intervention index')
    original = tuple(zip(PEOPLE, BASE_VALUES))
    altered = list(original)
    altered[index] = (PEOPLE[index], BASE_VALUES[(index + 1) % len(BASE_VALUES)])
    return original, tuple(altered)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    paths = {'base': base.MODEL / 'model.safetensors', 'candidate': SOURCE / 'candidate.pt',
             'report': SOURCE / 'result.json'}
    hashes = {name: base.sha(path) for name, path in paths.items()}
    report = json.loads(paths['report'].read_text(encoding='utf-8'))
    payload = torch.load(paths['candidate'], map_location='cpu', weights_only=True)
    if (hashes['base'] != base.WEIGHT_SHA or hashes['candidate'] != report['candidate_sha256']
            or payload['base_sha256'] != hashes['base']):
        raise ValueError('Source provenance mismatch')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    wrappers = install(model, tuple(payload['layers']))
    readers = attach_read_adapters(wrappers, rank=payload['read_rank'])
    writers = {layer: TTTNTPWriteTargetAdapter(wrappers[layer].base.out_features,
                                                rank=payload['write_rank']).to('cuda')
               for layer in wrappers}
    for layer in wrappers:
        readers[layer].load_state_dict(payload['readers'][str(layer)], strict=True)
        writers[layer].load_state_dict(payload['writers'][str(layer)], strict=True)
        writers[layer].requires_grad_(False)
    model.requires_grad_(False)
    cases = [intervention(index) for index in range(len(PEOPLE))]
    pairs = {record for pair in cases for events in pair for record in events}
    caches = {item: fact_cache(model, tokenizer, wrappers,
                               f'{item[0]}喜欢喝{item[1]}。', item[1]) for item in pairs}
    ids = {value: tokenizer.encode(value, add_special_tokens=False) for value in BASE_VALUES}
    if any(len(value) != 1 for value in ids.values()):
        raise ValueError('Probe requires single-token answers')
    conditions = {'layer6': (6,), 'layer12': (12,), 'both': (6, 12)}
    rows = []
    for index, (original, altered) in enumerate(cases):
        states = {label: write_sequence(wrappers, [caches[item] for item in events], writers,
                                        differentiable=False)
                  for label, events in (('original', original), ('altered', altered))}
        question = f'请只回答{PEOPLE[index]}喜欢喝什么饮料。'
        prompt = closed_think_inputs(tokenizer, question, 'cuda')
        old_value, new_value = original[index][1], altered[index][1]
        results = {}
        for condition, layers in conditions.items():
            results[condition] = {}
            for label, state in states.items():
                with torch.inference_mode(), use_memory({layer: state[layer] for layer in layers}):
                    logits = model(**prompt, use_cache=False, return_dict=True).logits[0, -1].float()
                probabilities = torch.softmax(logits, dim=-1)
                results[condition][label] = {
                    'top_id': int(logits.argmax()),
                    'old_probability': float(probabilities[ids[old_value][0]]),
                    'new_probability': float(probabilities[ids[new_value][0]]),
                    'old_logit': float(logits[ids[old_value][0]]),
                    'new_logit': float(logits[ids[new_value][0]]),
                }
            results[condition]['both_first_tokens_correct'] = (
                results[condition]['original']['top_id'] == ids[old_value][0]
                and results[condition]['altered']['top_id'] == ids[new_value][0])
        rows.append({'person': PEOPLE[index], 'old': old_value, 'new': new_value, 'conditions': results})
        print(json.dumps({'completed': index + 1}, ensure_ascii=False), flush=True)
    if any(base.sha(path) != hashes[name] for name, path in paths.items()):
        raise ValueError('Protected input changed')
    summary = {condition: sum(row['conditions'][condition]['both_first_tokens_correct'] for row in rows)
               for condition in conditions}
    result = {'summary': summary, 'rows': rows, 'source_sha256': hashes,
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Four previously used names and values in fixed order and phrasing.',
                              'Ablations use a reader trained with both layers; no retraining.',
                              'Measures first-token probabilities, not completed free generation.',
                              'Fact boundaries supplied; not autonomous writing or independent generalization.',
                              'Layer 12 inputs change when layer 6 is active; effects need not be additive.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
