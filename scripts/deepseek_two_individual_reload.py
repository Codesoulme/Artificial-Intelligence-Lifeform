"""Save two controlled individuals and check each in a separate process."""
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
from scripts.deepseek_integrated_checkpoint_probe import READOUT, SOURCE
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.deepseek_named_readout_alignment import snapshot
from scripts.deepseek_semantic_value_pilot import WORDS, word_correct
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load, query
from spora.integrated_checkpoint import load_integrated, save_integrated
from spora.integrated_open_base import ScaledOpenBaseMemory
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-two-individual-history-diagnostic-20261110'


def verify(checkpoint, expected):
    if expected not in (1, 6):
        raise ValueError('Invalid controlled value')
    before = {'checkpoint': sha(checkpoint), 'weight': sha(MODEL / 'model.safetensors'), 'original': sha(INDIVIDUAL)}
    if before['weight'] != WEIGHT_SHA or before['original'] != INDIVIDUAL_SHA:
        raise ValueError('Protected source mismatch')
    config_sha = feature_files_sha256(MODEL)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    model, organism, state = load_integrated(checkpoint, base, MODEL)
    model.eval()
    digest = frozen_digest(organism, state)
    if not state.pages or not any(torch.count_nonzero(page[site][0]).item()
                                  for page in state.pages if isinstance(page, list)
                                  for site in range(len(organism.memories))):
        raise ValueError('Checkpoint has no active historical memory')
    key = torch.tensor([KEYS[0]], dtype=torch.long)
    with torch.inference_mode():
        native = int(query(organism, state, key, torch.zeros_like(key), torch.zeros_like(key)).argmax(-1))
    addresses = [F.normalize(site.current_basis.to('cuda').float(), dim=-1)[KEYS[0]:KEYS[0] + 1]
                 for site in organism.memories]
    inputs = closed_think_inputs(tokenizer, f'{NAMES[0]}喜欢的饮品是什么？', 'cuda')
    generations = {}
    with torch.inference_mode():
        for mode, current_only in (('current_only', True), ('with_history', False)):
            output = model.generate(**inputs, memory_state=state, memory_addresses=addresses,
                                    current_only=current_only, max_new_tokens=16, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
            tokens = output[0, inputs['input_ids'].shape[1]:].tolist()
            answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            generations[mode] = {'tokens': tokens, 'answer': answer,
                                 'correct': word_correct(answer, WORDS[expected]),
                                 'ended_with_eos': tokenizer.eos_token_id in tokens}
    if (sha(checkpoint) != before['checkpoint'] or sha(MODEL / 'model.safetensors') != before['weight']
            or sha(INDIVIDUAL) != before['original'] or feature_files_sha256(MODEL) != config_sha
            or frozen_digest(organism, state) != digest):
        raise ValueError('Protected input changed')
    return {'pid': os.getpid(), 'checkpoint_sha256': before['checkpoint'], 'state_digest': digest,
            'history_pages': len(state.pages),
            'native_value': native, 'expected_value': expected, 'generations': generations,
            'native_correct': native == expected,
            'current_only_correct': generations['current_only']['correct'],
            'with_history_correct': generations['with_history']['correct'],
            'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = {'weight': MODEL / 'model.safetensors', 'original': INDIVIDUAL,
             'individual': SOURCE / 'individual.pt', 'individual_report': SOURCE / 'result.json',
             'adapter': READOUT / 'adapter.pt', 'adapter_report': READOUT / 'result.json'}
    before = {name: sha(path) for name, path in files.items()}
    if before['weight'] != WEIGHT_SHA or before['original'] != INDIVIDUAL_SHA:
        raise ValueError('Protected source mismatch')
    for name, field in (('individual', 'saved_individual_sha256'), ('adapter', 'adapter_sha256')):
        with files[name + '_report'].open(encoding='utf-8') as stream:
            if json.load(stream)[field] != before[name]:
                raise ValueError('Source checkpoint mismatch')
    data = torch.load(files['adapter'], map_location='cpu', weights_only=True)
    if (data.get('format') != 'deepseek-spora-semantic-value-development-v1'
            or data.get('weight_sha256') != WEIGHT_SHA or data.get('individual_sha256') != INDIVIDUAL_SHA
            or data.get('layers') != [5, 17] or data.get('strength') != 0.5):
        raise ValueError('Adapter provenance mismatch')
    config_sha = feature_files_sha256(MODEL)
    torch.set_num_threads(2)
    started = time.perf_counter()
    organism, original, _ = load(files['individual'], 'cpu')
    original_digest = frozen_digest(organism, original)
    reader = ScaledOpenBaseMemory(data['state']['query.0.weight'].shape[1], organism.config.memory_width, 0.5)
    reader.load_state_dict(data['state'], strict=True)
    states = [snapshot(organism, KEYS[0], value, distractor=15) for value in (1, 6)]
    if any(not state.pages or not any(torch.count_nonzero(page[site][0]).item()
                                      for page in state.pages if isinstance(page, list)
                                      for site in range(len(organism.memories))) for state in states):
        raise ValueError('Expected nonempty active history in each individual')
    digests = [frozen_digest(organism, state) for state in states]
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoints = [OUTPUT / f'individual-{value}.pt' for value in (1, 6)]
    for path, state in zip(checkpoints, states):
        save_integrated(path, organism, state, reader, MODEL, (5, 17))
    hashes = [sha(path) for path in checkpoints]
    if hashes[0] == hashes[1] or digests[0] == digests[1]:
        raise ValueError('Expected distinct individual states')
    results = []
    for path, value in zip(checkpoints, (1, 6)):
        print(f'verifying saved individual {len(results) + 1}/2', flush=True)
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--verify', str(path), '--value', str(value)],
                               check=False, capture_output=True, text=True, encoding='utf-8', timeout=300,
                               env=dict(os.environ, PYTHONIOENCODING='utf-8'))
        if child.returncode:
            raise RuntimeError(f'Independent verification failed: {child.stderr[-4000:]}')
        results.append(json.loads(child.stdout.strip().splitlines()[-1]))
    if (len({result['pid'] for result in results}) != 2
            or any(not result['native_correct'] or not result['current_only_correct'] for result in results)
            or [result['state_digest'] for result in results] != digests
            or [result['checkpoint_sha256'] for result in results] != hashes
            or [sha(path) for path in checkpoints] != hashes or frozen_digest(organism, original) != original_digest
            or [frozen_digest(organism, state) for state in states] != digests
            or any(sha(path) != before[name] for name, path in files.items())):
        raise ValueError('Reload mismatch or protected source changed')
    report = {'status': 'two_individual_history_independent_reload_development_only', 'results': results,
              'summary': {'individuals': len(results), 'native_current_correct': sum(r['native_correct'] for r in results),
                          'current_only_answer_correct': sum(r['current_only_correct'] for r in results),
                          'current_plus_history_answer_correct': sum(r['with_history_correct'] for r in results)},
              'component_sha256': before, 'feature_config_sha256': config_sha, 'seconds': time.perf_counter() - started,
              'limitations': ['Two controlled states derived from one model and fixed address, not independently trained individuals.',
                              'One fixed prompt, two finite values, and explicitly supplied memory address.',
                              'Frozen DeepSeek and tokenizer remain separate files and are reloaded in each process.',
                              'Full history is loaded, but the experimental decoder reader is trained only for current memory.',
                              'Normal restart is not crash recovery, open-domain memory or identity authentication.',
                              'Forced closed reasoning and short answer do not establish general conversation.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'summary': report['summary'], 'pids': [r['pid'] for r in results]}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--value', type=int)
    args = parser.parse_args()
    if args.verify is None and args.value is None:
        run()
    elif args.verify is not None and args.value is not None:
        print(json.dumps(verify(args.verify, args.value), ensure_ascii=False), flush=True)
    else:
        parser.error('Both --verify and --value are required')
