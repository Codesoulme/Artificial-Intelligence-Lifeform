"""Check mixed per-request memory reads on the real integrated DeepSeek model."""
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
from scripts.deepseek_integrated_checkpoint_probe import OUTPUT as CHECKPOINT
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.integrated_checkpoint import load_integrated
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-integrated-read-mask-20261110.json'


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = {'weight': MODEL / 'model.safetensors', 'original': INDIVIDUAL,
             'individual': CHECKPOINT / 'individual.pt', 'individual_report': CHECKPOINT / 'result.json'}
    before = {name: sha(path) for name, path in files.items()}
    if before['weight'] != WEIGHT_SHA or before['original'] != INDIVIDUAL_SHA:
        raise ValueError('Protected source mismatch')
    with files['individual_report'].open(encoding='utf-8') as stream:
        if json.load(stream)['checkpoint_sha256'] != before['individual']:
            raise ValueError('Integrated checkpoint provenance mismatch')
    config_sha = feature_files_sha256(MODEL)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    model, organism, state = load_integrated(files['individual'], base, MODEL)
    model.eval()
    digest = frozen_digest(organism, state)
    inputs = closed_think_inputs(tokenizer, f'{NAMES[0]}喜欢的饮品是什么？', 'cuda')
    inputs = {key: value.expand(2, -1) for key, value in inputs.items()}
    addresses = [F.normalize(site.current_basis.to('cuda').float(), dim=-1)[KEYS[0]:KEYS[0] + 1]
                 for site in organism.memories]
    memory_args = {'memory_state': state, 'memory_addresses': addresses, 'current_only': True}
    masks = {'plain': None, 'all': torch.tensor([True, True]), 'mixed': torch.tensor([True, False])}
    logits, generated = {}, {}
    with torch.inference_mode():
        for mode, mask in masks.items():
            options = {} if mode == 'plain' else dict(memory_args, memory_read_mask=mask)
            logits[mode] = model(**inputs, **options, use_cache=False).logits[:, -1].float().cpu()
            output = model.generate(**inputs, **options, max_new_tokens=8, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
            generated[mode] = output[:, inputs['input_ids'].shape[1]:].tolist()
    differences = {'enabled_vs_all': float((logits['mixed'][0] - logits['all'][0]).abs().max()),
                   'disabled_vs_plain': float((logits['mixed'][1] - logits['plain'][1]).abs().max()),
                   'enabled_vs_plain': float((logits['mixed'][0] - logits['plain'][0]).abs().max())}
    def through_eos(tokens):
        return tokens[:tokens.index(tokenizer.eos_token_id) + 1] if tokenizer.eos_token_id in tokens else tokens
    tokens_match = {'enabled_vs_all': through_eos(generated['mixed'][0]) == through_eos(generated['all'][0]),
                    'disabled_vs_plain': through_eos(generated['mixed'][1]) == through_eos(generated['plain'][1])}
    print(json.dumps({'differences': differences, 'tokens_match': tokens_match, 'generated': generated}), flush=True)
    if (not all(torch.isfinite(value).all() for value in logits.values())
            or differences['enabled_vs_all'] != 0 or differences['disabled_vs_plain'] != 0
            or differences['enabled_vs_plain'] == 0 or not all(tokens_match.values())):
        raise ValueError('Mixed mask comparison failed')
    if (any(sha(path) != before[name] for name, path in files.items()) or feature_files_sha256(MODEL) != config_sha
            or frozen_digest(organism, state) != digest):
        raise ValueError('Protected input or state changed')
    report = {'status': 'mixed_read_mask_development_only', 'max_logit_differences': differences,
              'tokens_match': tokens_match, 'generated_tokens': generated,
              'component_sha256': before, 'feature_config_sha256': config_sha,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['One duplicated prompt and manually selected known address, not open-world routing.',
                              'Mask is supplied by the caller; no identity, confidence or unknown detection is inferred.',
                              'No training, memory write, multi-user authentication, concurrent generation or beam search tested.',
                              'Readout remains experimental and does not establish general chat or safe memory.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'differences': differences, 'tokens_match': tokens_match}), flush=True)


if __name__ == '__main__':
    run()
