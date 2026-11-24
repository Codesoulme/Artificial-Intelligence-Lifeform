"""Measure how much of a dialogue update the migrated model can retain."""

import argparse
import json
from pathlib import Path

from scripts.audit_local_named_chat import FRESH_CASES, run
from scripts.audit_spora_chat_backbones import evaluate
from scripts.deepseek_local_baseline import MODEL, sha, torch
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_dialogue_composition import settings_for
from spora.integrated_checkpoint import _digest, load_integrated_runtime
from spora.language_training import frozen_digest
from transformers import AutoTokenizer


def load_payload(path):
    value = torch.load(path, weights_only=True, map_location='cpu')
    if not isinstance(value, dict) or set(value) != {'metadata', 'tensors', 'integrity_sha256'} or (
            _digest(value['metadata'], value['tensors']) != value['integrity_sha256']):
        raise ValueError(f'Invalid integrated checkpoint: {path}')
    return value['metadata'], value['tensors']


def shared_update(parent, update):
    left_meta, left = load_payload(parent)
    right_meta, right = load_payload(update)
    if left.keys() != right.keys():
        raise ValueError('Checkpoints have different tensor sets')

    left_specs, right_specs = left_meta.pop('residuals'), right_meta.pop('residuals')
    if left_meta != right_meta or len(left_specs) != len(right_specs):
        raise ValueError('Checkpoints have different structure or individual metadata')
    allowed = set()
    for a, b in zip(left_specs, right_specs, strict=True):
        if {key: value for key, value in a.items() if key != 'sha256'} != {
                key: value for key, value in b.items() if key != 'sha256'}:
            raise ValueError('Residual configurations differ')
        prefix = f"residual/{a['index']}/"
        parts = {part: prefix + part for part in ('down', 'up')}
        for spec, tensors in ((a, left), (b, right)):
            identity = {key: value for key, value in spec.items() if key != 'sha256'}
            if _digest(identity, {part: tensors[key] for part, key in parts.items()}) != spec['sha256']:
                raise ValueError('Invalid residual digest')
        if a.get('ordinary_only') is True:
            allowed.update(parts.values())

    changed = {}
    for key, old in left.items():
        new = right[key]
        if old.dtype != new.dtype or old.shape != new.shape:
            raise ValueError(f'Incompatible tensor {key}')
        if not torch.equal(old, new):
            if key not in allowed:
                raise ValueError(f'Update changed non-dialogue tensor {key}')
            changed[key] = (old, new)
    if not changed:
        raise ValueError('The supplied update has no changed dialogue weights')
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent', type=Path)
    parser.add_argument('update', type=Path)
    parser.add_argument('development', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--scales', type=float, nargs='+', default=(.25, .5, .75))
    args = parser.parse_args()
    if len(set(args.scales)) != len(args.scales) or any(not 0 < scale < 1 for scale in args.scales):
        parser.error('Use distinct interpolation scales strictly between zero and one')
    inputs = {path: sha(path) for path in (args.parent, args.update, args.development)}
    changes = shared_update(args.parent, args.update)
    cases = json.loads(args.development.read_text(encoding='utf-8'))['scenarios']
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(args.parent, load_full_streamed(), MODEL, disposable_base=True)
    state = frozen_digest(runtime.organism, runtime.state)
    decoder = runtime.decoder.eval().requires_grad_(False)
    settings = settings_for(decoder, tokenizer, 96)
    parameters = {}
    for key in changes:
        _, index, part = key.split('/')
        parameters[key] = getattr(decoder.base.model.layers[int(index)], part).weight
    for scale in args.scales:
        with torch.no_grad():
            for key, parameter in parameters.items():
                old, new = changes[key]
                parameter.copy_(torch.lerp(old, new, scale))
        ordinary = run(decoder, tokenizer)
        fresh = run(decoder, tokenizer, cases=FRESH_CASES)
        opened, _ = evaluate(decoder, tokenizer, cases, {}, settings)
        folder = args.output / f'scale-{scale:g}'
        folder.mkdir()
        candidate = runtime.save(folder / 'candidate.pt')
        if (frozen_digest(runtime.organism, runtime.state) != state
                or any(sha(path) != digest for path, digest in inputs.items())):
            raise ValueError('The screen changed protected inputs')
        report = {'scale': scale, 'sources': {str(path): value for path, value in inputs.items()},
                  'individual_sha256': state, 'candidate_sha256': sha(candidate),
                  'original': ordinary, 'fresh': fresh, 'open': opened}
        with (folder / 'result.json').open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        print(scale, 'original', ordinary['cases_passed'], ordinary['turns_passed'],
              'fresh', fresh['cases_passed'], fresh['turns_passed'], flush=True)


if __name__ == '__main__':
    main()
