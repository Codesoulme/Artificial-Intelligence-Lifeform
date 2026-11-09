"""Create and independently reload an integrated research checkpoint."""
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
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_cluener_locator_train import OUTPUT as LOCATOR
from scripts.deepseek_learned_name_span import single_span_mask
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_named_binding_probe import KEYS, NAMES, pooled
from scripts.deepseek_named_span_persistence import CONTRACT, OUTPUT as SOURCE
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_semantic_value_pilot import WORDS, word_correct
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.current_memory_status import current_memory_status
from spora.growing import load
from spora.integrated_checkpoint import load_integrated, save_integrated
from spora.integrated_open_base import ScaledOpenBaseMemory
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-integrated-checkpoint-v2-20261110'
READOUT = ROOT / 'runs/deepseek-semantic-value-balanced-20261110'


def verify(checkpoint):
    config_sha = feature_files_sha256(MODEL)
    locator_path, locator_report = LOCATOR / 'locator.pt', LOCATOR / 'result.json'
    with locator_report.open(encoding='utf-8') as stream:
        prior = json.load(stream)
    locator_data = torch.load(locator_path, map_location='cpu', weights_only=True)
    if (prior['locator_sha256'] != sha(locator_path)
            or locator_data.get('format') != 'deepseek-name-span-locator-development-v1'
            or locator_data.get('weight_sha256') != WEIGHT_SHA or locator_data.get('feature_config_sha256') != config_sha
            or locator_data.get('layers') != [6, 18] or locator_data.get('threshold') != 0.5):
        raise ValueError('Locator provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    integrated, organism, state = load_integrated(checkpoint, base, MODEL)
    integrated.eval()
    digest = frozen_digest(organism, state)
    texts = [f'{name}喜欢的饮品是什么？' for name in NAMES[:2]]
    features, masks = extract(base, tokenizer, texts, batch_size=1)
    vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
    locator = nn.Linear(vectors.shape[-1], 1).eval()
    locator.load_state_dict(locator_data['state'], strict=True)
    with torch.no_grad():
        scores = locator(vectors).squeeze(-1).sigmoid()
    bases = [F.normalize(site.current_basis.to('cuda').float(), dim=-1) for site in organism.memories]
    rows = []
    for i, (text, expected_key, expected_value) in enumerate(zip(texts, KEYS[:2], (6, 1))):
        mask, single = single_span_mask(scores[i], masks[i])
        predicted_key, answer, status = None, None, None
        if single:
            features_at_name = pooled(features[i:i + 1], mask.unsqueeze(0))
            with torch.inference_mode():
                result = organism.named_address_binding(features_at_name, organism, WEIGHT_SHA, config_sha,
                                                         feature_contract=CONTRACT)
            if bool(result['accepted'][0]):
                predicted_key = int(result['indices'][0, 0])
                status = current_memory_status(organism, state, predicted_key)['status']
                if status != 'unknown':
                    addresses = [basis[predicted_key:predicted_key + 1] for basis in bases]
                    inputs = closed_think_inputs(tokenizer, text, 'cuda')
                    with torch.no_grad():
                        generated = integrated.generate(**inputs, memory_state=state, memory_addresses=addresses,
                                                         current_only=True, max_new_tokens=8, do_sample=False,
                                                         pad_token_id=tokenizer.eos_token_id)
                    answer = tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        rows.append({'question': text, 'predicted_key': predicted_key, 'expected_key': expected_key,
                     'status': status, 'answer': answer, 'expected_word': WORDS[expected_value],
                     'correct': predicted_key == expected_key and answer is not None
                     and word_correct(answer, WORDS[expected_value])})
    if frozen_digest(organism, state) != digest or feature_files_sha256(MODEL) != config_sha:
        raise ValueError('Loaded state or feature files changed')
    return {'rows': rows, 'correct': sum(r['correct'] for r in rows), 'state_digest': digest,
            'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = {'weight': MODEL / 'model.safetensors', 'original': INDIVIDUAL,
             'individual': SOURCE / 'individual.pt', 'individual_report': SOURCE / 'result.json',
             'adapter': READOUT / 'adapter.pt', 'adapter_report': READOUT / 'result.json',
             'locator': LOCATOR / 'locator.pt', 'locator_report': LOCATOR / 'result.json'}
    before = {name: sha(path) for name, path in files.items()}
    if before['weight'] != WEIGHT_SHA or before['original'] != INDIVIDUAL_SHA:
        raise ValueError('Protected input mismatch')
    reports = {}
    for name in ('individual', 'adapter', 'locator'):
        with files[name + '_report'].open(encoding='utf-8') as stream:
            reports[name] = json.load(stream)
    if (reports['individual']['saved_individual_sha256'] != before['individual']
            or reports['adapter']['adapter_sha256'] != before['adapter']
            or reports['locator']['locator_sha256'] != before['locator']):
        raise ValueError('Source checkpoint mismatch')
    data = torch.load(files['adapter'], map_location='cpu', weights_only=True)
    if (data.get('format') != 'deepseek-spora-semantic-value-development-v1'
            or data.get('weight_sha256') != WEIGHT_SHA or data.get('individual_sha256') != INDIVIDUAL_SHA
            or data.get('layers') != [5, 17] or data.get('strength') != 0.5):
        raise ValueError('Reader provenance mismatch')
    organism, state, _ = load(files['individual'], 'cpu')
    digest = frozen_digest(organism, state)
    reader = ScaledOpenBaseMemory(data['state']['query.0.weight'].shape[1], organism.config.memory_width, 0.5)
    reader.load_state_dict(data['state'], strict=True)
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT / 'individual.pt'
    save_integrated(checkpoint, organism, state, reader, MODEL, (5, 17))
    checkpoint_sha = sha(checkpoint)
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--verify', str(checkpoint)],
                           check=False, capture_output=True, text=True, encoding='utf-8', timeout=300,
                           env=dict(os.environ, PYTHONIOENCODING='utf-8'))
    if child.returncode:
        raise RuntimeError(f'Independent verification failed: {child.stderr[-4000:]}')
    result = json.loads(child.stdout.strip().splitlines()[-1])
    if (result['state_digest'] != digest or sha(checkpoint) != checkpoint_sha
            or frozen_digest(organism, state) != digest or any(sha(path) != before[name] for name, path in files.items())):
        raise ValueError('Protected input or state changed')
    report = {'status': 'integrated_checkpoint_research_only', 'checkpoint_sha256': checkpoint_sha,
              'component_sha256': before, 'separate_process': result,
              'limitations': ['Base weights, tokenizer and name locator remain separate on disk.',
                              'File fingerprints do not prove an arbitrary caller-supplied in-memory base came from those files.',
                              'Two development examples; fixed vocabulary and explicitly selected read path.',
                              'Checksum detects accidental changes, not malicious tampering.',
                              'Normal restart is not crash-recovery or an open-world safety test.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'correct': result['correct'], 'examples': len(result['rows']),
                      'state_digest': digest, 'checkpoint_sha256': checkpoint_sha}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify is None:
        run()
    else:
        print(json.dumps(verify(args.verify), ensure_ascii=False), flush=True)
