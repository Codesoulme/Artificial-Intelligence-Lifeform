"""Regenerate language answers from a separately reloaded research individual."""
import json
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
from scripts.deepseek_read_strength_probe import ScaledReadout
from scripts.deepseek_semantic_value_pilot import WORDS, word_correct
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.current_memory_status import current_memory_status
from spora.growing import load
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256
from spora.open_base_memory import memory_hooks

OUTPUT = ROOT / 'runs/deepseek-named-span-generation-reload-20261110.json'
READOUT = ROOT / 'runs/deepseek-semantic-value-balanced-20261110'


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = {'weight': MODEL / 'model.safetensors', 'original': INDIVIDUAL,
             'individual': SOURCE / 'individual.pt', 'source_report': SOURCE / 'result.json',
             'locator': LOCATOR / 'locator.pt', 'locator_report': LOCATOR / 'result.json',
             'readout': READOUT / 'adapter.pt', 'readout_report': READOUT / 'result.json'}
    before = {name: sha(path) for name, path in files.items()}
    if before['weight'] != WEIGHT_SHA or before['original'] != INDIVIDUAL_SHA:
        raise ValueError('Protected input mismatch')
    reports = {}
    for name in ('source', 'locator', 'readout'):
        with files[name + '_report'].open(encoding='utf-8') as stream:
            reports[name] = json.load(stream)
    if (reports['source']['saved_individual_sha256'] != before['individual']
            or reports['locator']['locator_sha256'] != before['locator']
            or reports['readout']['adapter_sha256'] != before['readout']):
        raise ValueError('Checkpoint provenance mismatch')
    config_sha = feature_files_sha256(MODEL)
    locator_data = torch.load(files['locator'], map_location='cpu', weights_only=True)
    readout_data = torch.load(files['readout'], map_location='cpu', weights_only=True)
    if (locator_data.get('format') != 'deepseek-name-span-locator-development-v1'
            or locator_data.get('weight_sha256') != WEIGHT_SHA or locator_data.get('feature_config_sha256') != config_sha
            or locator_data.get('layers') != [6, 18] or locator_data.get('threshold') != 0.5
            or readout_data.get('format') != 'deepseek-spora-semantic-value-development-v1'
            or readout_data.get('weight_sha256') != WEIGHT_SHA or readout_data.get('individual_sha256') != INDIVIDUAL_SHA
            or readout_data.get('layers') != [5, 17] or readout_data.get('strength') != 0.5):
        raise ValueError('Component provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, state, _ = load(files['individual'], 'cpu')
    digest = frozen_digest(organism, state)
    if digest != reports['source']['separate_process']['state_digest']:
        raise ValueError('Individual state mismatch')
    texts = [f'{name}喜欢的饮品是什么？' for name in NAMES[:2]]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    features, masks = extract(base, tokenizer, texts, batch_size=1)
    token_vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
    locator = nn.Linear(token_vectors.shape[-1], 1).eval()
    locator.load_state_dict(locator_data['state'], strict=True)
    adapter = ScaledReadout(base.config.hidden_size, organism.config.memory_width, 0.5).to('cuda').eval()
    adapter.load_state_dict(readout_data['state'], strict=True)
    bases = [F.normalize(site.current_basis.to('cuda').float(), dim=-1) for site in organism.memories]
    with torch.no_grad():
        scores = locator(token_vectors).squeeze(-1).sigmoid()
    rows = []
    for i, (text, expected_key, expected_value) in enumerate(zip(texts, KEYS[:2], (6, 1))):
        mask, single = single_span_mask(scores[i], masks[i])
        predicted_key, status, answer = None, None, None
        if single:
            vectors = pooled(features[i:i + 1], mask.unsqueeze(0))
            with torch.inference_mode():
                prediction = organism.named_address_binding(vectors, organism, WEIGHT_SHA, config_sha,
                                                            feature_contract=CONTRACT)
            if bool(prediction['accepted'][0]):
                predicted_key = int(prediction['indices'][0, 0])
                status = current_memory_status(organism, state, predicted_key)['status']
                if status != 'unknown':
                    addresses = [basis[predicted_key:predicted_key + 1] for basis in bases]
                    inputs = closed_think_inputs(tokenizer, text, 'cuda')
                    with torch.no_grad(), memory_hooks([base.model.layers[5], base.model.layers[17]], adapter, state,
                                                       current_only=True, addresses=addresses):
                        generated = base.generate(**inputs, max_new_tokens=8, do_sample=False,
                                                  pad_token_id=tokenizer.eos_token_id)
                    answer = tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        rows.append({'question': text, 'single_span': single, 'predicted_key': predicted_key,
                     'expected_key': expected_key, 'status': status, 'answer': answer,
                     'expected_word': WORDS[expected_value],
                     'correct': predicted_key == expected_key and answer is not None
                     and word_correct(answer, WORDS[expected_value])})
    if (any(sha(path) != before[name] for name, path in files.items())
            or feature_files_sha256(MODEL) != config_sha or frozen_digest(organism, state) != digest):
        raise ValueError('Protected input changed')
    report = {'status': 'reloaded_isolated_individual_research_only', 'rows': rows,
              'correct': sum(r['correct'] for r in rows), 'examples': len(rows),
              'individual_sha256': before['individual'], 'component_sha256': before,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Two development names and questions; not open-domain or independent evaluation.',
                              'The experiment invokes the memory-read path explicitly; no automatic intent routing.',
                              'The encoder, locator and readout remain separately loaded artifacts.',
                              'Ordinary process reload does not prove crash recovery or safe live writing.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'correct': report['correct'], 'examples': report['examples']}), flush=True)


if __name__ == '__main__':
    run()
