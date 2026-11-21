"""Verify native value-position checkpoint packing and durable memory recovery."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoTokenizer

from audit_awake_reasoning_residual_holdout_20261121 import DATA
from audit_spora_dialogue_migration import evaluate_dialogues
from scripts.audit_qwen_compatible_durable_20261121 import EARLIER, KEY, measure
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_qwen_counterfactual_reply_20261122 import HELD, evaluate, records
from spora.durable import TensorStore
from spora.integrated_checkpoint import _digest, load_integrated_runtime, value_position_modules
from spora.integrated_durable import create_integrated_durable, load_integrated_durable
from spora.language_training import frozen_digest
from spora.pretrained_attention import load_value_position_attention
from spora.value_position_checkpoint import VALUE_POSITION_PREFIX


LAYER = 18


def write(path, report):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def component_digest(runtime):
    modules = value_position_modules(runtime.decoder)
    if set(modules) != {LAYER} or modules[LAYER].group != 0:
        raise ValueError('Incorrect value-position layer or group after restoration')
    component = modules[LAYER]
    return _digest({'layer': LAYER, 'group': component.group},
                   {'position': component.position.weight, 'weight': component.weight, 'bias': component.bias})


def pack(args):
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.component / 'packed.json').read_text(encoding='utf-8'))
    verification = json.loads((args.component / 'verified.json').read_text(encoding='utf-8'))
    bootstrap = args.output / 'bootstrap.pt'
    digest = sha(args.parent)
    if not verification['accepted'] or manifest['source_sha256'] != digest:
        raise ValueError('Only the verified value component can be packed into its parent')
    runtime = load_integrated_runtime(args.parent, load_full_streamed(), MODEL)
    before = frozen_digest(runtime.organism, runtime.state)
    layer = runtime.decoder.base.model.layers[LAYER]
    layer.self_attn = load_value_position_attention(args.component / 'value-position.pt',
                                                   layer.self_attn, checkpoint_digest=digest,
                                                   component_digest=manifest['component_sha256'])
    runtime.save(bootstrap)
    parent, updated = (torch.load(path, map_location='cpu', weights_only=True)
                       for path in (args.parent, bootstrap))
    expected = {**parent['metadata'], 'value_position_layers': {str(LAYER): 0}}
    previous = {k: v for k, v in updated['tensors'].items() if not k.startswith(VALUE_POSITION_PREFIX)}
    if (updated['metadata'] != expected or previous.keys() != parent['tensors'].keys()
            or any(not torch.equal(value, previous[key]) for key, value in parent['tensors'].items())
            or frozen_digest(runtime.organism, runtime.state) != before or sha(args.parent) != digest):
        raise ValueError('Packing changed data outside the value-position component')
    report = {'parent_sha256': digest, 'bootstrap_sha256': sha(bootstrap),
              'component_parameters_sha256': component_digest(runtime), 'individual_sha256': before,
              'evaluated_component_sha256': sha(args.component / 'verified.json'), 'dialogue_sha256': sha(DATA)}
    write(args.output / 'packed.json', report)
    print('Saved integrated bootstrap', report['bootstrap_sha256'], flush=True)


def create(args):
    bootstrap, database = args.output / 'bootstrap.pt', args.output / 'state.db'
    destination = args.output / 'initial.json'
    if database.exists() or destination.exists():
        raise FileExistsError(database)
    packed = json.loads((args.output / 'packed.json').read_text(encoding='utf-8'))
    reference = args.component / 'verified.json'
    if (sha(bootstrap) != packed['bootstrap_sha256'] or sha(reference) != packed['evaluated_component_sha256']
            or sha(DATA) != packed['dialogue_sha256']):
        raise ValueError('Candidate or reference changed after packing')
    expected = json.loads(reference.read_text(encoding='utf-8'))['result']
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    with create_integrated_durable(database, bootstrap, load_full_streamed(), MODEL) as runtime:
        if (component_digest(runtime) != packed['component_parameters_sha256']
                or frozen_digest(runtime.organism, runtime.state) != packed['individual_sha256']):
            raise ValueError('The integrated checkpoint restored different tensors')
        dialogue = evaluate_dialogues(runtime.decoder, tokenizer, 'native_value_position',
                                      limit=96, open_reasoning=False)
        for name, turns in dialogue.items():
            if [turn['tokens'] for turn in turns] != [turn['tokens'] for turn in expected['dialogue'][name]]:
                raise ValueError(f'Native value-position dialogue changed: {name}')
        memory = evaluate(runtime.decoder, tokenizer, runtime.organism, records(runtime.organism, HELD))
        if memory != expected['memory']:
            raise ValueError('Native memory answers differ after loading')
        event = runtime.state.cursor
        decision, stats = runtime.observe_sourced(KEY, EARLIER, 'direct', 1.)
        if not decision['current_write'] or stats['revision'] != 1:
            raise ValueError('First isolated memory write failed')
    report = {'dialogue_turns': sum(map(len, dialogue.values())), 'memory_answers': memory['summary']['count'],
              'earlier_event': event, 'first_write': stats}
    write(destination, report)
    print(report, flush=True)


def recover(args):
    bootstrap, database = args.output / 'bootstrap.pt', args.output / 'state.db'
    destination = args.output / 'recovered.json'
    if destination.exists():
        raise FileExistsError(destination)
    packed = json.loads((args.output / 'packed.json').read_text(encoding='utf-8'))
    initial = json.loads((args.output / 'initial.json').read_text(encoding='utf-8'))
    if sha(bootstrap) != packed['bootstrap_sha256']:
        raise ValueError('Integrated bootstrap changed before recovery')
    revisions = {}
    for stage, revision in (('after_manifest', 1), ('after_commit', 2)):
        child = subprocess.run([sys.executable, '-m', 'scripts.audit_qwen_compatible_durable_20261121',
                                '--database', str(database), '--interrupt', stage],
                               cwd=ROOT, check=False)
        with TensorStore(database) as store:
            _, _, _, restored_revision = store.read()
        if child.returncode != 42 or restored_revision != revision:
            raise ValueError(f'Unexpected recovery after {stage}: {child.returncode}, {restored_revision}')
        revisions[stage] = restored_revision
    with load_integrated_durable(database, bootstrap, load_full_streamed(), MODEL) as runtime:
        if component_digest(runtime) != packed['component_parameters_sha256']:
            raise ValueError('Crash recovery did not restore the value component')
        answers = measure(runtime, initial['earlier_event'])
        if not all(row['correct'] for row in answers):
            raise ValueError('Failed to generate the restored current or historical fact')
    if sha(bootstrap) != packed['bootstrap_sha256']:
        raise ValueError('Durable writes changed the structural bootstrap')
    report = {**initial, 'crash_revisions': revisions, 'memory_answers_after_restart': answers,
              'bootstrap_sha256': packed['bootstrap_sha256'], 'bootstrap_bytes': bootstrap.stat().st_size,
              'database_bytes': database.stat().st_size}
    write(destination, report)
    print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('pack', 'create', 'recover'))
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--component', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'pack' and args.parent is None:
        parser.error('pack requires --parent')
    if args.mode in ('pack', 'create') and args.component is None:
        parser.error('pack and create require --component')
    torch.set_num_threads(2)
    {'pack': pack, 'create': create, 'recover': recover}[args.mode](args)
