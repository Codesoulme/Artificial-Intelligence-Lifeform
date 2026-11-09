"""Copy a validated legacy experiment into a fingerprint-bound individual."""
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

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_named_individual_roundtrip import OUTPUT as SOURCE
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load, query, save
from spora.language_training import frozen_digest
from spora.named_address import NamedAddressBinding, address_basis_sha256, feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-named-individual-bound-20261110'


def verify(individual, audit, feature_digest):
    model, state, _ = load(individual, 'cpu')
    data = torch.load(audit, map_location='cpu', weights_only=True)
    with torch.inference_mode():
        result = model.named_address_binding(data['features'], model, WEIGHT_SHA, feature_digest)
        keys = data['keys']
        values = query(model, state, keys, torch.zeros_like(keys), torch.zeros_like(keys)).argmax(-1)
    if (not torch.equal(result['indices'], data['indices'])
            or not torch.equal(result['accepted'], data['accepted'])
            or not torch.allclose(result['scores'], data['scores'], atol=1e-6, rtol=0)
            or not torch.equal(values, data['values'])):
        raise ValueError('Fingerprint-bound state differs from reference')
    return {'predictions': len(data['features']), 'accepted': int(result['accepted'].sum()),
            'memory_values': len(keys), 'state_digest': frozen_digest(model, state)}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source, audit, report_path = SOURCE / 'individual.pt', SOURCE / 'features.pt', SOURCE / 'result.json'
    protected = {p: sha(p) for p in (MODEL / 'model.safetensors', INDIVIDUAL, source, audit, report_path)}
    if protected[MODEL / 'model.safetensors'] != WEIGHT_SHA or protected[INDIVIDUAL] != INDIVIDUAL_SHA:
        raise ValueError('Protected model mismatch')
    feature_digest = feature_files_sha256(MODEL)
    with report_path.open(encoding='utf-8') as stream:
        prior = json.load(stream)
    if (prior['saved_individual_sha256'] != protected[source] or prior['audit_features_sha256'] != protected[audit]
            or prior['weight_sha256'] != WEIGHT_SHA or prior['individual_sha256'] != INDIVIDUAL_SHA):
        raise ValueError('Prior experiment provenance mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    model, state, metadata = load(source, 'cpu')
    data = torch.load(audit, map_location='cpu', weights_only=True)
    legacy = model.named_address_binding
    if legacy.basis_sha256 is not None or legacy.feature_config_sha256 is not None:
        raise ValueError('Expected a legacy binding without fingerprints')
    with torch.inference_mode():
        old = legacy(data['features'], model, WEIGHT_SHA, allow_legacy_unbound=True)
    if (not torch.equal(old['indices'], data['indices']) or not torch.equal(old['accepted'], data['accepted'])
            or not torch.allclose(old['scores'], data['scores'], atol=1e-6, rtol=0)):
        raise ValueError('Legacy predictions do not match saved audit')
    basis_digest = address_basis_sha256(model)
    config = legacy.config_dict()
    config.update(basis_sha256=basis_digest, feature_config_sha256=feature_digest)
    binding = NamedAddressBinding(**config)
    with torch.no_grad():
        binding.weights.copy_(legacy.weights)
    model.named_address_binding = binding
    OUTPUT.mkdir(parents=False, exist_ok=False)
    individual = OUTPUT / 'individual.pt'
    save(individual, model, state, dict(metadata, purpose='fingerprint_bound_named_experiment'))
    same_process = verify(individual, audit, feature_digest)
    process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--verify', str(individual), str(audit),
                              feature_digest], check=True, capture_output=True, text=True, encoding='utf-8', timeout=90)
    separate_process = json.loads(process.stdout.strip())
    if (same_process != separate_process or same_process['state_digest'] != frozen_digest(model, state)
            or feature_files_sha256(MODEL) != feature_digest or any(sha(p) != d for p, d in protected.items())):
        raise ValueError('Protected input or restored state changed')
    report = {'status': 'experimental_fingerprint_bound_named_individual', 'weight_sha256': WEIGHT_SHA,
              'individual_sha256': INDIVIDUAL_SHA, 'source_sha256': protected[source],
              'saved_individual_sha256': sha(individual), 'basis_sha256': basis_digest,
              'feature_config_sha256': feature_digest, 'same_process': same_process,
              'separate_process': separate_process, 'seconds': time.perf_counter() - started,
              'limitations': ['Features and fitted bindings are from an earlier development experiment.',
                              'Fingerprints verify declared files and basis, not that arbitrary supplied features were generated from them.',
                              'The previous experiment did not record configuration hashes at feature extraction time.',
                              'This migration does not improve name accuracy, natural-language values, or route safety.',
                              'A normal separate-process reload is not crash-recovery validation.',
                              'The DeepSeek encoder remains a separate on-disk model.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', nargs=3, metavar=('INDIVIDUAL', 'AUDIT', 'FEATURE_DIGEST'))
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(Path(args.verify[0]), Path(args.verify[1]), args.verify[2])), flush=True)
    else:
        run()
