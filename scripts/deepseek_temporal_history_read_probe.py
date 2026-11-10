"""Probe temporal address and page-slot reads through integrated DeepSeek."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_integrated_checkpoint_probe import READOUT
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.batched_memory import batch_states
from spora.growing import load, query
from spora.integrated_checkpoint import load_integrated, save_integrated
from spora.integrated_open_base import ScaledOpenBaseMemory
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-temporal-history-read-final-20261110'


def temporal_addresses(organism, key, event_time, device):
    keys = torch.tensor([key], dtype=torch.long, device=device)
    modes = torch.ones_like(keys)
    times = torch.tensor([event_time], dtype=torch.long, device=device)
    _, bound = organism.representation(keys, modes, times)
    return [organism.memory_address(site, bound, keys, modes) for site in organism.memories]


def current_addresses(organism, key, device):
    return [F.normalize(site.current_basis[key:key + 1].to(device).float(), dim=-1)
            for site in organism.memories]


def verify(checkpoint):
    before = {'checkpoint': sha(checkpoint), 'weight': sha(MODEL / 'model.safetensors'), 'original': sha(INDIVIDUAL)}
    if before['weight'] != WEIGHT_SHA or before['original'] != INDIVIDUAL_SHA:
        raise ValueError('Protected source mismatch')
    config_sha = feature_files_sha256(MODEL)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'left'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    model, organism, state = load_integrated(checkpoint, base, MODEL, individual_device='cuda')
    model.eval()
    digest = frozen_digest(organism, state)
    key = KEYS[0]
    modes = [('current', 0, current_addresses(organism, key, 'cuda'),
              f'{NAMES[0]}现在喜欢的饮品是什么？'),
             ('old_event', 1, temporal_addresses(organism, key, 0, 'cuda'),
              f'{NAMES[0]}过去喜欢的饮品是什么？'),
             ('new_event', 1, temporal_addresses(organism, key, 1, 'cuda'),
              f'{NAMES[0]}后来喜欢的饮品是什么？')]
    inputs = [closed_think_inputs(tokenizer, question, 'cuda') for _, _, _, question in modes]
    length = max(item['input_ids'].shape[1] for item in inputs)
    batch = {name: torch.cat([F.pad(item[name], (length - item[name].shape[1], 0),
                                    value=tokenizer.pad_token_id if name == 'input_ids' else 0)
                              for item in inputs], dim=0) for name in ('input_ids', 'attention_mask')}
    batched_state = batch_states([state, state, state])
    memory_addresses = [torch.cat([entry[2][site] for entry in modes], dim=0) for site in range(2)]
    slots = torch.tensor([entry[1] for entry in modes], dtype=torch.long)
    with torch.inference_mode():
        logits = model(**batch, memory_state=batched_state, memory_addresses=memory_addresses,
                       memory_slots=slots, current_only=False, use_cache=False).logits[:, -1].float().cpu()
        generated = model.generate(**batch, memory_state=batched_state, memory_addresses=memory_addresses,
                                   memory_slots=slots, current_only=False, max_new_tokens=24,
                                   do_sample=False, pad_token_id=tokenizer.eos_token_id)
    results = []
    for index, (name, slot, addresses, question) in enumerate(modes):
        tokens = generated[index, length:].tolist()
        answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        k = torch.tensor([key], dtype=torch.long, device=next(organism.parameters()).device)
        if name == 'current':
            native = int(query(organism, state, k, torch.zeros_like(k), torch.zeros_like(k)).argmax(-1))
            expected = 6
        else:
            event_time = 0 if name == 'old_event' else 1
            event_times = torch.tensor([event_time], dtype=torch.long, device=k.device)
            native = int(query(organism, state, k, torch.ones_like(k), event_times).argmax(-1))
            expected = 1 if event_time == 0 else 6
        results.append({'kind': name, 'slot': slot, 'question': question, 'native_value': native,
                        'expected_value': expected, 'native_correct': native == expected,
                        'answer': answer, 'generated_tokens': len(tokens),
                        'ended_with_eos': tokenizer.eos_token_id in tokens,
                        'address_norms': [float(addresses[site].norm()) for site in range(2)]})
    if (sha(checkpoint) != before['checkpoint'] or sha(MODEL / 'model.safetensors') != before['weight']
            or sha(INDIVIDUAL) != before['original'] or feature_files_sha256(MODEL) != config_sha
            or frozen_digest(organism, state) != digest):
        raise ValueError('Protected source changed')
    return {'pid': os.getpid(), 'checkpoint_sha256': before['checkpoint'], 'state_digest': digest,
            'results': results, 'seconds': time.perf_counter() - started,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated()}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = {'weight': MODEL / 'model.safetensors', 'original': INDIVIDUAL,
             'adapter': READOUT / 'adapter.pt', 'adapter_report': READOUT / 'result.json'}
    before = {name: sha(path) for name, path in files.items()}
    if before['weight'] != WEIGHT_SHA or before['original'] != INDIVIDUAL_SHA:
        raise ValueError('Protected source mismatch')
    with files['adapter_report'].open(encoding='utf-8') as stream:
        if json.load(stream)['adapter_sha256'] != before['adapter']:
            raise ValueError('Readout adapter provenance mismatch')
    data = torch.load(files['adapter'], map_location='cpu', weights_only=True)
    if (data.get('format') != 'deepseek-spora-semantic-value-development-v1'
            or data.get('weight_sha256') != WEIGHT_SHA or data.get('individual_sha256') != INDIVIDUAL_SHA
            or data.get('layers') != [5, 17] or data.get('strength') != 0.5):
        raise ValueError('Reader provenance mismatch')
    config_sha = feature_files_sha256(MODEL)
    torch.set_num_threads(2)
    torch.manual_seed(4862)
    organism, source_state, _ = load(INDIVIDUAL, 'cpu')
    original_digest = frozen_digest(organism, source_state)
    # The source individual has the controlled write controller; create an audited fresh state.
    state = fresh_controlled_state(organism)
    key = KEYS[0]
    events = []
    for value in (1, 6):
        state, decision = observe_sourced(organism, state, key, value, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('Controlled temporal fact was not committed')
        events.append({'time': state.cursor - 1, 'value': value, 'action': decision['action']})
    for event_time, expected in ((0, 1), (1, 6)):
        k = torch.tensor([key])
        native = int(query(organism, state, k, torch.ones_like(k), torch.tensor([event_time])).argmax(-1))
        if native != expected:
            raise ValueError(f'Native historical event mismatch at {event_time}: {native}')
    reader = ScaledOpenBaseMemory(data['state']['query.0.weight'].shape[1], organism.config.memory_width, 0.5)
    reader.load_state_dict(data['state'], strict=True)
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT / 'historical-individual.pt'
    save_integrated(checkpoint, organism, state, reader, MODEL, (5, 17))
    checkpoint_sha = sha(checkpoint)
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--verify', str(checkpoint)],
                           check=False, capture_output=True, text=True, encoding='utf-8', timeout=300,
                           env=dict(os.environ, PYTHONIOENCODING='utf-8'))
    if child.returncode:
        raise RuntimeError(f'Independent historical verification failed (exit={child.returncode}).\n'
                           f'stdout:\n{child.stdout[-4000:]}\nstderr:\n{child.stderr[-4000:]}')
    result = json.loads(child.stdout.strip().splitlines()[-1])
    if (sha(checkpoint) != checkpoint_sha or frozen_digest(organism, source_state) != original_digest
            or any(sha(path) != before[name] for name, path in files.items())):
        raise ValueError('Protected input changed')
    report = {'status': 'temporal_address_reader_development_only', 'events': events,
              'independent_process': result, 'checkpoint_sha256': checkpoint_sha,
              'component_sha256': before, 'feature_config_sha256': config_sha,
              'limitations': ['One address and two synthetic direct observations; no independent identity.',
                              'Time is explicitly supplied by the experiment; natural-language time parsing is absent.',
                              'The adapter was trained for current-page readout; historical results are diagnostic only.',
                              'No write API for user-requested history routing, crash recovery or open-domain chat tested.',
                              'Base and tokenizer remain external files.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'events': events, 'results': result['results']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify is None:
        run()
    else:
        print(json.dumps(verify(args.verify), ensure_ascii=False), flush=True)
