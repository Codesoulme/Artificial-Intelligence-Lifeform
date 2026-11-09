"""Compare explicit decoder integration against prior memory hooks on DeepSeek."""
import json
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
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.deepseek_named_readout_alignment import snapshot
from scripts.deepseek_named_span_persistence import OUTPUT as SOURCE
from scripts.deepseek_read_strength_probe import ScaledReadout
from scripts.deepseek_semantic_value_pilot import WORDS, word_correct
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load
from spora.integrated_open_base import IntegratedOpenBase
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256
from spora.open_base_memory import memory_hooks

OUTPUT = ROOT / 'runs/deepseek-integrated-layer-comparison-20261110.json'
READOUT = ROOT / 'runs/deepseek-semantic-value-balanced-20261110'


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = {'weight': MODEL / 'model.safetensors', 'original': INDIVIDUAL,
             'individual': SOURCE / 'individual.pt', 'individual_report': SOURCE / 'result.json',
             'adapter': READOUT / 'adapter.pt', 'adapter_report': READOUT / 'result.json'}
    before = {name: sha(path) for name, path in files.items()}
    if before['weight'] != WEIGHT_SHA or before['original'] != INDIVIDUAL_SHA:
        raise ValueError('Protected input mismatch')
    reports = {}
    for name in ('individual', 'adapter'):
        with files[name + '_report'].open(encoding='utf-8') as stream:
            reports[name] = json.load(stream)
    if (reports['individual']['saved_individual_sha256'] != before['individual']
            or reports['adapter']['adapter_sha256'] != before['adapter']):
        raise ValueError('Checkpoint provenance mismatch')
    checkpoint = torch.load(files['adapter'], map_location='cpu', weights_only=True)
    if (checkpoint.get('format') != 'deepseek-spora-semantic-value-development-v1'
            or checkpoint.get('weight_sha256') != WEIGHT_SHA or checkpoint.get('individual_sha256') != INDIVIDUAL_SHA
            or checkpoint.get('layers') != [5, 17] or checkpoint.get('strength') != 0.5):
        raise ValueError('Adapter provenance mismatch')
    config_sha = feature_files_sha256(MODEL)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, saved, _ = load(files['individual'], 'cpu')
    if organism.named_address_binding.feature_config_sha256 != config_sha:
        raise ValueError('Individual feature fingerprint mismatch')
    states = {'saved': saved, 'coffee': snapshot(organism, KEYS[0], 1, distractor=15),
              'water': snapshot(organism, KEYS[0], 6, distractor=15)}
    state_digests = {name: frozen_digest(organism, state) for name, state in states.items()}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    reader = ScaledReadout(base.config.hidden_size, organism.config.memory_width, 0.5).to('cuda').eval()
    reader.load_state_dict(checkpoint['state'], strict=True)
    bases = [F.normalize(site.current_basis.to('cuda').float(), dim=-1) for site in organism.memories]
    cases = [('saved_lin', 'saved', NAMES[0], KEYS[0], 6), ('saved_ning', 'saved', NAMES[1], KEYS[1], 1),
             ('switch_coffee', 'coffee', NAMES[0], KEYS[0], 1), ('switch_water', 'water', NAMES[0], KEYS[0], 6)]
    inputs = {name: closed_think_inputs(tokenizer, f'{name}喜欢的饮品是什么？', 'cuda')
              for name in (NAMES[0], NAMES[1])}

    def answer(model, case, legacy):
        _, state_name, name, key, _ = case
        current_inputs = inputs[name]
        addresses = [basis[key:key + 1] for basis in bases]
        with torch.no_grad():
            if legacy:
                with memory_hooks([base.model.layers[5], base.model.layers[17]], reader, states[state_name],
                                  current_only=True, addresses=addresses):
                    logits = base(**current_inputs, use_cache=False).logits[0, -1].float().cpu()
                    generated = base.generate(**current_inputs, max_new_tokens=8, do_sample=False,
                                              pad_token_id=tokenizer.eos_token_id)
            else:
                logits = model(**current_inputs, memory_state=states[state_name], memory_addresses=addresses,
                               current_only=True, use_cache=False).logits[0, -1].float().cpu()
                generated = model.generate(**current_inputs, memory_state=states[state_name], memory_addresses=addresses,
                                           current_only=True, max_new_tokens=8, do_sample=False,
                                           pad_token_id=tokenizer.eos_token_id)
        tokens = generated[0, current_inputs['input_ids'].shape[1]:].tolist()
        return logits, tokens, tokenizer.decode(tokens, skip_special_tokens=True).strip()

    with torch.no_grad():
        plain_logits = base(**inputs[NAMES[0]], use_cache=False).logits[0, -1].float().cpu()
    legacy = [answer(base, case, True) for case in cases]
    integrated = IntegratedOpenBase(base, reader, (5, 17)).eval()
    with torch.no_grad():
        plain_after = integrated(**inputs[NAMES[0]], use_cache=False).logits[0, -1].float().cpu()
    rows = []
    for case, previous in zip(cases, legacy):
        current = answer(integrated, case, False)
        rows.append({'case': case[0], 'expected': WORDS[case[4]], 'legacy_answer': previous[2],
                     'integrated_answer': current[2], 'same_tokens': previous[1] == current[1],
                     'max_logit_difference': float((previous[0] - current[0]).abs().max()),
                     'correct': word_correct(current[2], WORDS[case[4]])})
    if (any(sha(path) != before[name] for name, path in files.items()) or feature_files_sha256(MODEL) != config_sha
            or any(frozen_digest(organism, state) != state_digests[name] for name, state in states.items())):
        raise ValueError('Protected input changed')
    report = {'status': 'integrated_layers_development_only', 'rows': rows,
              'base_max_logit_difference': float((plain_logits - plain_after).abs().max()),
              'same_token_outputs': sum(r['same_tokens'] for r in rows),
              'correct': sum(r['correct'] for r in rows), 'component_sha256': before,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Addresses and memory routing are explicitly supplied, not inferred.',
                              'Four development cases on one model and greedy generation, not open-domain evaluation.',
                              'Layer modules wrap the base in memory; existing model files are not rewritten.',
                              'State and adapter remain separately saved; this does not establish automatic persistence or source judgment.',
                              'External key-value caches with active memory are rejected.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'same_token_outputs': report['same_token_outputs'], 'correct': report['correct'],
                      'base_max_logit_difference': report['base_max_logit_difference'],
                      'max_logit_difference': max(r['max_logit_difference'] for r in rows)}), flush=True)


if __name__ == '__main__':
    run()
