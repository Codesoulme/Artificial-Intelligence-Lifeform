"""Audit incremental persistence and process interruption on a migrated individual."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch

from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_memory_answer_distill import _memory_args
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_addressed_read import closed_think_inputs
from spora.durable import TensorStore
from spora.growing import query
from spora.integrated_durable import create_integrated_durable, load_integrated_durable
from spora.internal_memory import observe_sourced
from transformers import AutoTokenizer


KEY, EARLIER, LATEST = 9, 2, 5
EXIT_AFTER_STAGE = 42


def interrupt_write(database, stage):
    with TensorStore(database) as store:
        organism, state, _, revision = store.read()
        candidate, decision = observe_sourced(organism, state, KEY, LATEST, 'direct', 1.)
        if not decision['current_write']:
            raise ValueError('Controlled correction was rejected')

        def interrupt(phase):
            if phase == stage:
                os._exit(EXIT_AFTER_STAGE)

        store.commit(organism, candidate, expected_revision=revision, hook=interrupt)
    raise ValueError('Did not reach interruption point')


def require_metadata(runtime):
    decoder = runtime.decoder
    layers = decoder.base.model.layers
    if len(decoder.compatible_block_indices) != len(layers):
        raise ValueError('Expected a fully migrated compatible model')
    for index in decoder.layer_indices:
        wrapper = layers[index]
        if wrapper.feedforward_fraction != 1. or not wrapper.layer.mlp.read_memory_at_output:
            raise ValueError('A migrated reader was lost on load')


@torch.inference_mode()
def measure(runtime, event):
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    values = []
    for mode, expected, question in (
        ('current', LATEST, '小苏现在最喜欢喝什么饮料？'),
        ('past', EARLIER, '小苏更正前喜欢什么饮品？'),
    ):
        key = torch.tensor([KEY])
        lookup_mode = torch.tensor([int(mode == 'past')])
        time = torch.tensor([event if mode == 'past' else 0])
        evidence = query(runtime.organism, runtime.state, key, lookup_mode, time)
        if int(evidence.argmax(-1)[0]) != expected:
            raise ValueError('Reloaded neural memory did not recover the committed fact')
        inputs = closed_think_inputs(tokenizer, question, 'cuda')
        controls = _memory_args(runtime.organism, runtime.state, {'key': KEY, 'intent': mode, 'event_time': event})
        controls.pop('memory_state')
        tokens = runtime.generate(**inputs, **controls, max_new_tokens=24, do_sample=False,
                                  temperature=None, top_p=None, memory_output_logits=evidence,
                                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(tokens[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        values.append({'mode': mode, 'expected': WORDS[expected], 'answer': text, 'correct': WORDS[expected] in text})
    return values


def create(bootstrap, database, initial):
    if database.exists() or initial.exists():
        raise FileExistsError(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    digest = sha(bootstrap)
    with create_integrated_durable(database, bootstrap, load_full_streamed(), MODEL) as runtime:
        require_metadata(runtime)
        event = runtime.state.cursor
        decision, stats = runtime.observe_sourced(KEY, EARLIER, 'direct', 1.)
        if not decision['current_write'] or stats['revision'] != 1:
            raise ValueError('Controlled initial write did not commit')
        data = {'bootstrap_sha256': digest, 'earlier_event': event, 'first_write': stats}
    initial.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    print(data, flush=True)


def verify(bootstrap, database, initial, output):
    if output.exists():
        raise FileExistsError(output)
    data = json.loads(initial.read_text(encoding='utf-8'))
    crash_revisions = {}
    for stage, expected_revision in (('after_manifest', 1), ('after_commit', 2)):
        result = subprocess.run([sys.executable, '-m', 'scripts.audit_qwen_compatible_durable_20261121',
                                 '--database', str(database), '--interrupt', stage], cwd=ROOT, check=False)
        with TensorStore(database) as store:
            _, _, _, revision = store.read()
            if result.returncode != EXIT_AFTER_STAGE or revision != expected_revision:
                raise ValueError(f'Unexpected durable state after {stage}: exit {result.returncode}, revision {revision}')
            crash_revisions[stage] = revision

    with load_integrated_durable(database, bootstrap, load_full_streamed(), MODEL) as runtime:
        require_metadata(runtime)
        values = measure(runtime, data['earlier_event'])
    if sha(bootstrap) != data['bootstrap_sha256']:
        raise ValueError('Protected structural bootstrap changed')
    report = {**data, 'crash_revisions': crash_revisions, 'reloaded_values': values,
              'database_bytes': database.stat().st_size}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--bootstrap', type=Path)
    parser.add_argument('--initial', type=Path)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--create', action='store_true')
    operation.add_argument('--interrupt', choices=('after_manifest', 'after_commit'))
    operation.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.interrupt is not None:
        interrupt_write(args.database, args.interrupt)
    elif args.bootstrap is None or args.initial is None:
        parser.error('--bootstrap and --initial are required')
    elif args.create:
        create(args.bootstrap, args.database, args.initial)
    else:
        verify(args.bootstrap, args.database, args.initial, args.verify)
