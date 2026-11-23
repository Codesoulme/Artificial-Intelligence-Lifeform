"""Audit the calibrated Qwen structure with incremental neural state on disk."""

import argparse
import json
from pathlib import Path

from scripts.audit_named_memory_interleaved import KEYS, NAMES, append_generation, cases
from scripts.audit_named_memory_qwen_structure_control import (
    CHECKPOINT_SHA, QWEN_SHA, attach_control, load_calibration, memory_inputs,
)
from scripts.deepseek_local_baseline import sha, torch
from scripts.deepseek_spora_full_stream_loader import load_qwen_streamed
from scripts.deepseek_spora_runtime_chat import _prepare_memory_read
from scripts.spora_event_scan_selector import select_event
from scripts.spora_unknown_state_probe import fresh_controlled_state
from scripts.train_qwen_counterfactual_reply_20261122 import WORDS
from spora.durable import TensorStore
from spora.integrated_checkpoint import _digest
from spora.integrated_durable import IntegratedDurableRuntime
from spora.language_training import frozen_digest
from transformers import AutoTokenizer


@torch.inference_mode()
def assess(runtime, tokenizer, eos, generation):
    rows = []
    for name, text, key, mode, expected in cases(generation):
        address = runtime.resolve_confirmed_name(name)['key']
        value, answer, tokens = None, None, []
        if address is not None:
            event = None
            if mode:
                intent = 'effective_previous' if mode == 1 else 'effective_latest'
                selected = select_event(runtime.organism, runtime.state, address, intent,
                                        as_of=runtime.state.cursor - 1)
                if selected['status'] != 'experimental_candidate':
                    raise ValueError(f'Missing controlled version for {name}')
                event = selected['event_time']
            controls, evidence = _prepare_memory_read(runtime.organism, runtime.state, address,
                                                      ('current', 'past', 'later')[mode], event)
            controls.pop('memory_state')
            controls['use_cache'] = True
            index = int(evidence.argmax(-1)[0])
            value = WORDS[index] if index < len(WORDS) else None
            inputs = memory_inputs(tokenizer, text, 'cuda')
            generated = runtime.generate(**inputs, **controls, memory_output_logits=evidence, max_new_tokens=16,
                                         do_sample=False, temperature=None, top_p=None, top_k=None,
                                         eos_token_id=eos, pad_token_id=tokenizer.pad_token_id)
            tokens = generated[0, inputs['input_ids'].shape[1]:].tolist()
            answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        passed = address == key and value == expected and (answer is None if expected is None else
                 bool(tokens and tokens[-1] in eos and answer.rstrip('。！ ') == expected))
        rows.append({'name': name, 'question': text, 'address': address, 'mode': mode, 'expected': expected,
                     'value': value, 'answer': answer, 'tokens': tokens, 'passed': passed})
    return rows


def main(checkpoint, donor, calibration, output, phase):
    report = output / f'{phase}.json'
    if report.exists():
        raise FileExistsError(report)
    torch.set_num_threads(2)
    files = (checkpoint, *(donor / name for name in ('model.safetensors', 'config.json', 'tokenizer.json',
                                                    'tokenizer_config.json', 'generation_config.json')))
    sources = {str(path): sha(path) for path in files}
    if sources[str(checkpoint)] != CHECKPOINT_SHA or sources[str(donor / 'model.safetensors')] != QWEN_SHA:
        raise ValueError('Expected pinned control inputs')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    metadata, tensors = payload['metadata'], payload['tensors']
    if _digest(metadata, tensors) != payload['integrity_sha256']:
        raise ValueError('Checkpoint integrity mismatch')
    tokenizer = AutoTokenizer.from_pretrained(donor, local_files_only=True, trust_remote_code=False)
    base = load_qwen_streamed(donor, QWEN_SHA)
    decoder, organism, state = attach_control(base, metadata, tensors)
    reference = json.loads(calibration.read_text(encoding='utf-8'))
    path = calibration.with_suffix('.reply.pt')
    decoder.output_adapter, adapter_sha = load_calibration(path, reference, sources, decoder, organism)
    seed_sha = frozen_digest(organism, state)
    if seed_sha != reference['individual_sha256']:
        raise ValueError('Unexpected seed individual')
    binding = {'format': 'spora-qwen-structure-persistence-control-v1', 'sources': sources,
               'adapter_sha256': adapter_sha, 'seed_sha256': seed_sha}
    database = output / 'controlled-individual.sqlite'
    if phase == 'seed':
        state = fresh_controlled_state(organism)
        store = TensorStore.create(database, organism, state, binding)
    else:
        store = TensorStore(database)
    with store:
        prior = None
        if phase != 'seed':
            previous = 'seed' if phase == 'append' else 'append'
            prior = json.loads((output / f'{previous}.json').read_text(encoding='utf-8'))
            organism, state, saved_binding, revision = store.read()
            if (saved_binding != binding or revision != prior['revision']
                    or frozen_digest(organism, state) != prior['individual_sha256']):
                raise ValueError('Stored individual differs from the last committed phase')
        runtime = IntegratedDurableRuntime(decoder, organism, state, store, checkpoint)
        eos = base.generation_config.eos_token_id
        if prior is not None:
            rows = assess(runtime, tokenizer, eos, 1 if phase == 'append' else 2)
            if rows != prior['rows']:
                raise ValueError('Independently reloaded memory replies differ')
        if phase == 'seed':
            for name, key in zip(NAMES, KEYS, strict=True):
                runtime.enroll_name(key, name)
            append_generation(runtime, 0)
            append_generation(runtime, 1)
            rows = assess(runtime, tokenizer, eos, 1)
        elif phase == 'append':
            append_generation(runtime, 2)
            rows = assess(runtime, tokenizer, eos, 2)
        result = {'binding': binding, 'revision': store.revision, 'cursor': runtime.state.cursor,
                  'individual_sha256': frozen_digest(runtime.organism, runtime.state), 'reopen_exact': prior is not None,
                  'passed': sum(row['passed'] for row in rows), 'rows': rows}
    with report.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(phase, result['passed'], '/', len(rows), 'revision', result['revision'],
          'reopen exact', result['reopen_exact'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'donor', 'calibration', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--phase', choices=('seed', 'append', 'reopen'), required=True)
    main(**vars(parser.parse_args()))
