"""Save a research correction candidate and compare generation after a restart.

Run train and verify as separate processes. This does not change the base model
or the SPORA individual and does not establish reproducible training resumption.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_local_baseline as p
from scripts import deepseek_spora_diverse_kl_correction_audit as experiment
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_online_lora_memory_pilot import OnlineDelta
from spora import research_adapter_checkpoint as checkpoint

OUT = ROOT / 'runs/deepseek-spora-adapter-restart-audit-20261116'
LAYERS = (18, 23)
FEATURE_FILES = ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json')
CODE_FILES = ('scripts/deepseek_spora_full_stream_loader.py',
              'scripts/deepseek_spora_online_lora_memory_pilot.py',
              'scripts/deepseek_spora_ttt_read_adapter_pilot.py',
              'scripts/deepseek_addressed_read.py')


def binding():
    files = {name: p.sha(p.MODEL / name) for name in FEATURE_FILES}
    files.update({name: p.sha(ROOT / name) for name in CODE_FILES})
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest(), files


def train():
    OUT.mkdir(parents=True, exist_ok=False)
    base = p.sha(p.MODEL / 'model.safetensors')
    if base != p.WEIGHT_SHA:
        raise ValueError('Unexpected base weight')
    features, files = binding()
    experiment.OUT = OUT / 'training.json'

    def save_candidate(modules, optimizer, steps):
        checkpoint.save(OUT / 'candidate.pt', modules, optimizer, base_sha256=base,
                        feature_sha256=features, layers=LAYERS, steps=steps)
        metadata = {'base_sha256': base, 'feature_sha256': features, 'files': files,
                    'candidate_sha256': p.sha(OUT / 'candidate.pt')}
        with (OUT / 'binding.json').open('x', encoding='utf-8') as stream:
            json.dump(metadata, stream, indent=2)

    experiment.main(candidate_callback=save_candidate)


def verify():
    started = time.perf_counter()
    metadata = json.loads((OUT / 'binding.json').read_text(encoding='utf-8'))
    training = json.loads((OUT / 'training.json').read_text(encoding='utf-8'))
    features, files = binding()
    base = p.sha(p.MODEL / 'model.safetensors')
    if (base != p.WEIGHT_SHA or base != metadata['base_sha256']
            or features != metadata['feature_sha256'] or files != metadata['files']
            or p.sha(OUT / 'candidate.pt') != metadata['candidate_sha256']):
        raise ValueError('Candidate or base binding mismatch')
    torch.manual_seed(experiment.SEED)
    torch.set_num_threads(2)
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    model.requires_grad_(False)
    modules = []
    for layer in LAYERS:
        mlp = model.model.layers[layer].mlp
        mlp.down_proj = OnlineDelta(mlp.down_proj, 8)
        modules.append(mlp.down_proj)
    params = [parameter for module in modules for part in (module.down, module.up)
              for parameter in part.parameters()]
    optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0, foreach=False)
    restored = checkpoint.restore(OUT / 'candidate.pt', modules, base_sha256=base,
                                  feature_sha256=features, layers=LAYERS, optimizer=optimizer,
                                  allow_legacy_optimizer=True)
    rows = []
    original_rows = training['states'][-1]['rows']
    for original in original_rows:
        answer = pilot.generate(model, tokenizer, original['question'])
        rows.append({'group': original['group'], 'question': original['question'],
                     'original': original['answer'], 'restored': answer,
                     'same_tokens': answer['token_ids'] == original['answer']['token_ids']})
        print(json.dumps({'group': original['group'], 'same_tokens': rows[-1]['same_tokens']},
                         ensure_ascii=False), flush=True)
    if p.sha(p.MODEL / 'model.safetensors') != base:
        raise ValueError('Base weight changed during verification')
    report = {'restored': restored, 'candidate_sha256': metadata['candidate_sha256'],
              'same_tokens': sum(row['same_tokens'] for row in rows), 'total': len(rows),
              'rows': rows, 'seconds': time.perf_counter() - started,
              'limitations': ['Generation capped at 24 tokens, greedy decoding.',
                              'Optimizer loaded but continued training equivalence was not tested.',
                              'Small labeled development experiment; no autonomous memory claim.']}
    with (OUT / 'restart.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'same_tokens': report['same_tokens'], 'total': report['total']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('train', 'verify'))
    args = parser.parse_args()
    train() if args.mode == 'train' else verify()
