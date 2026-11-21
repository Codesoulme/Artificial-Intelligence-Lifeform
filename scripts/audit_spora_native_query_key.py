"""Pack partial query/key positions and audit natural memory after native reload."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoTokenizer

from scripts.audit_qwen_compatible_natural_memory_20261121 import controlled, load_head, score
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_checkpoint import load_integrated_runtime, query_key_position_modules, value_position_modules
from spora.language_training import frozen_digest
from spora.research_partial_query_key import PartialQueryKeyPositionAttention
from spora.value_position_checkpoint import QUERY_KEY_POSITION_PREFIX


def write(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def assess(runtime, tokenizer, router, temporal):
    states = [controlled(runtime.organism, offset) for offset in (0, 2)]
    digests = [frozen_digest(runtime.organism, state) for state in states]
    rows, _ = score(runtime.decoder, tokenizer, runtime.organism, router, temporal, states, fresh_v2=True)
    if digests != [frozen_digest(runtime.organism, state) for state in states]:
        raise ValueError('Natural evaluation changed its controlled memory')
    positive = [row for row in rows if row['expected_key'] is not None]
    negative = [row for row in rows if row['expected_key'] is None]
    return {'summary': {'known': len(positive), 'correct_key': sum(r['key'] == r['expected_key'] for r in positive),
                        'correct_time': sum(r['mode'] == r['expected_mode'] for r in positive),
                        'correct_pairs': sum(r['correct_pair'] for r in positive),
                        'unsupported': len(negative), 'unsupported_routed': sum(r['key'] is not None for r in negative)},
            'rows': rows}


def pack(args, runtime, sources):
    selected = json.loads(args.component.with_suffix('.json').read_text(encoding='utf-8'))
    verified = json.loads(args.component.with_name('verified.json').read_text(encoding='utf-8'))
    component_digest = sha(args.component)
    if (component_digest != selected['component_sha256'] or not verified['selected_reload_exact']
            or not verified['checks']['passed']
            or verified['sources'].get(str(args.component)) != component_digest):
        raise ValueError('A verified and reproduced query/key component is required')
    payload = torch.load(args.component, weights_only=True, map_location='cpu')
    layer = payload['layer']
    parent = value_position_modules(runtime.decoder)[layer]
    if (payload['format'] != 'spora-partial-query-key-v1' or payload['parent_sha256'] != sources['parent_sha256']
            or payload['group'] != parent.group):
        raise ValueError('Component belongs to a different parent or layer')
    runtime.decoder.base.model.layers[layer].self_attn = PartialQueryKeyPositionAttention(
        parent, payload['pairs'], payload['weight'], payload['bias'],
    ).eval()
    checkpoint = args.output / 'bootstrap.pt'
    runtime.save(checkpoint)
    before, after = (torch.load(path, map_location='cpu', weights_only=True) for path in (args.parent, checkpoint))
    other = {k: v for k, v in after['tensors'].items() if not k.startswith(QUERY_KEY_POSITION_PREFIX)}
    expected = dict(before['metadata'], query_key_position_layers={str(layer): list(payload['pairs'])})
    if (after['metadata'] != expected or other.keys() != before['tensors'].keys()
            or any(not torch.equal(tensor, other[key]) for key, tensor in before['tensors'].items())):
        raise ValueError('Packing changed checkpoint data outside query/key positions')
    for name in ('weight', 'bias'):
        if not torch.equal(after['tensors'][f'{QUERY_KEY_POSITION_PREFIX}{layer}/{name}'], payload[name]):
            raise ValueError('Packed query/key tensors differ from the tested component')
    return {'bootstrap_sha256': sha(checkpoint), 'component_sha256': component_digest,
            'layer': layer, 'group': parent.group, 'pairs': payload['pairs']}


def main(args):
    output = args.output
    output_path = output / ('before.json' if args.component is not None else 'reloaded.json')
    if output_path.exists() or (args.component is not None and (output / 'bootstrap.pt').exists()):
        raise FileExistsError(output_path)
    started = time.perf_counter()
    torch.set_num_threads(2)
    training_base = sha(args.route_source)
    router, router_sha = load_head(args.router, 'router.pt', 'router_sha256', training_base)
    temporal, temporal_sha = load_head(args.temporal, 'temporal.pt', 'temporal_sha256', training_base)
    sources = {'parent_sha256': sha(args.parent), 'route_training_checkpoint_sha256': training_base,
               'router_sha256': router_sha, 'temporal_sha256': temporal_sha}
    checkpoint = args.parent if args.component is not None else output / 'bootstrap.pt'
    if args.component is None:
        baseline = json.loads((output / 'before.json').read_text(encoding='utf-8'))
        if baseline['sources'] != sources or sha(checkpoint) != baseline['packed']['bootstrap_sha256']:
            raise ValueError('Reload must use the same evaluated parent, candidate and heads')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL)
    runtime.decoder.eval().requires_grad_(False)
    individual_digest = frozen_digest(runtime.organism, runtime.state)
    report = {'sources': sources, 'individual_sha256': individual_digest}
    if args.component is None:
        packed = baseline['packed']
        restored = query_key_position_modules(runtime.decoder)
        module = restored.get(packed['layer'])
        if (module is None or module.group != packed['group'] or list(module.pairs) != packed['pairs']
                or individual_digest != baseline['individual_sha256']):
            raise ValueError('Reload restored different positional or individual state')
    report['natural'] = assess(runtime, tokenizer, router, temporal)
    print('Natural', report['natural']['summary'], flush=True)
    if args.component is not None:
        output.mkdir(parents=True, exist_ok=True)
        report['packed'] = pack(args, runtime, sources)
    else:
        old, new = baseline['natural']['rows'], report['natural']['rows']
        fields = ('question', 'key', 'proposed_key', 'mode', 'selected_text', 'routing_reason', 'answers', 'correct_pair')
        changed = [{key: after[key] for key in fields}
                   for before, after in zip(old, new, strict=True)
                   if any(before[key] != after[key] for key in fields)]
        report['changed_behaviors'] = changed
        report['matched_behaviors'] = len(new) - len(changed)
    if (frozen_digest(runtime.organism, runtime.state) != individual_digest
            or sha(args.parent) != sources['parent_sha256']
            or (args.component is None and sha(checkpoint) != baseline['packed']['bootstrap_sha256'])):
        raise ValueError('Natural evaluation changed an individual or source checkpoint')
    report['seconds'] = time.perf_counter() - started
    write(output_path, report)
    print('Wrote', output_path, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'route-source', 'router', 'temporal', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--component', type=Path, help='Evaluate the parent and pack this verified candidate')
    main(parser.parse_args())
