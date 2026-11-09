"""Persist an experimental named binding with its SPORA state and verify reload."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_named_binding_probe import HELD_TEMPLATES, NAMES, OUTPUT as SOURCE, UNKNOWN, pooled
from scripts.deepseek_pooled_address import extract
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import load, query, save
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest
from spora.named_address import NamedAddressBinding

OUTPUT = ROOT / 'runs/deepseek-named-individual-20261110'


def verify(individual, audit):
    model, state, _ = load(individual, 'cpu')
    data = torch.load(audit, map_location='cpu', weights_only=True)
    binding = model.named_address_binding
    with torch.inference_mode():
        result = binding(data['features'], model, WEIGHT_SHA, allow_legacy_unbound=True)
        keys = data['keys']
        zeros = torch.zeros_like(keys)
        values = query(model, state, keys, zeros, zeros).argmax(-1)
    if (not torch.equal(result['indices'], data['indices'])
            or not torch.equal(result['accepted'], data['accepted'])
            or not torch.allclose(result['scores'], data['scores'], atol=1e-6, rtol=0)
            or not torch.equal(values, data['values'])):
        raise ValueError('Reloaded binding or memory differs')
    return {'predictions': len(data['features']), 'memory_values': len(keys),
            'accepted': int(result['accepted'].sum()), 'state_digest': frozen_digest(model, state)}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_report, source_binding = SOURCE / 'result.json', SOURCE / 'binding.pt'
    protected = {p: sha(p) for p in (MODEL / 'model.safetensors', INDIVIDUAL, source_report, source_binding)}
    if protected[MODEL / 'model.safetensors'] != WEIGHT_SHA or protected[INDIVIDUAL] != INDIVIDUAL_SHA:
        raise ValueError('Protected model mismatch')
    with source_report.open(encoding='utf-8') as stream:
        previous = json.load(stream)
    data = torch.load(source_binding, map_location='cpu', weights_only=True)
    if (previous['binding_sha256'] != protected[source_binding]
            or data.get('format') != 'deepseek-spora-named-binding-experiment-v1'
            or data.get('weight_sha256') != WEIGHT_SHA or data.get('individual_sha256') != INDIVIDUAL_SHA
            or data.get('sites_hidden_indices') != [6, 18]):
        raise ValueError('Binding provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    original_digest = frozen_digest(organism, original)
    state = fresh_controlled_state(organism)
    for event in previous['events']:
        state, decision = observe_sourced(organism, state, event['key'], event['value'], 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('Direct observation was not written')
    weights = data['weights']
    if weights.ndim != 3 or weights.shape[0] != 2 or weights.shape[2] != organism.config.memory_width:
        raise ValueError('Unexpected binding dimensions')
    binding = NamedAddressBinding(weights.shape[1], weights.shape[2], WEIGHT_SHA, data['threshold'])
    with torch.no_grad():
        binding.weights.copy_(weights)
    organism.named_address_binding = binding
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    questions = [template.format(name=name) for name in NAMES + UNKNOWN for template in HELD_TEMPLATES]
    features, masks = extract(base, tokenizer, questions, batch_size=1)
    vectors = pooled(features, masks)
    with torch.inference_mode():
        result = binding(vectors, organism, WEIGHT_SHA, allow_legacy_unbound=True)
        keys = torch.tensor([event['key'] for event in previous['events']], dtype=torch.long)
        zero = torch.zeros_like(keys)
        values = query(organism, state, keys, zero, zero).argmax(-1)
    expected_values = torch.tensor([event['value'] for event in previous['events']])
    if not torch.equal(values, expected_values):
        raise ValueError('Replayed native values differ')
    for prior, indices, accepted in zip(previous['rows'], result['indices'].tolist(), result['accepted'].tolist()):
        if prior['predicted'] != indices or prior['accepted'] != accepted:
            raise ValueError('Previous name prediction did not reproduce')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    individual, audit = OUTPUT / 'individual.pt', OUTPUT / 'features.pt'
    save(individual, organism, state, {'purpose': 'experimental_named_binding_roundtrip'})
    torch.save({'features': vectors, 'indices': result['indices'], 'scores': result['scores'],
                'accepted': result['accepted'], 'keys': keys, 'values': values}, audit)
    same_process = verify(individual, audit)
    process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--verify', str(individual), str(audit)],
                             check=True, capture_output=True, text=True, encoding='utf-8', timeout=90)
    separate_process = json.loads(process.stdout.strip())
    if same_process != separate_process or same_process['state_digest'] != frozen_digest(organism, state):
        raise ValueError('Separate process differs')
    del organism.named_address_binding
    if frozen_digest(organism, original) != original_digest or any(sha(p) != d for p, d in protected.items()):
        raise ValueError('Protected source changed')
    report = {'status': 'experimental_named_individual_roundtrip', 'weight_sha256': WEIGHT_SHA,
              'individual_sha256': INDIVIDUAL_SHA, 'binding_sha256': protected[source_binding],
              'saved_individual_sha256': sha(individual), 'audit_features_sha256': sha(audit),
              'same_process': same_process, 'separate_process': separate_process,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['The individual stores learned address matrices and numeric memory, not raw names or natural-language facts.',
                              'A matching external DeepSeek encoder and tokenizer are still needed to produce features.',
                              'Only encoder weight SHA and declared pooling contract are stored; tokenizer files are not fingerprinted by the binding.',
                              'The fitted binding and questions were used in earlier development, not independent evaluation.',
                              'A successful process reload is not a crash-recovery or atomic transaction test.',
                              'The native and language model are not yet a single fully integrated architecture.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', nargs=2, metavar=('INDIVIDUAL', 'AUDIT'))
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(*map(Path, args.verify))), flush=True)
    else:
        run()
