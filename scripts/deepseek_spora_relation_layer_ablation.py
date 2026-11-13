"""Ablate each structural replacement on a previously fixed classification set."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_spora_relation_base_comparison import OUTPUT as PRIOR, classify

OUTPUT = ROOT / 'runs/deepseek-spora-relation-layer-ablation-20261114.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    prior_sha = p.sha(PRIOR)
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    if base_sha != p.WEIGHT_SHA or prior['base_sha256'] != base_sha or prior['checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Prior provenance mismatch')
    rows = [{key: r[key] for key in ('set', 'group', 'question', 'label')} for r in prior['rows']]
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='spora-layer-ablation-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        base = p.load_full_streamed().eval()
        originals = {index: base.model.layers[index] for index in (6, 12)}
        chat = p.runtime.load_chat(base_loader=lambda: base)
        source_digest = p.frozen_digest(chat.organism, chat.state)
        replacements = {index: base.model.layers[index] for index in originals}
        for label, replacement_indices in (('original', ()), ('layer_6', (6,)),
                                           ('layer_12', (12,)), ('both', (6, 12))):
            for index in originals:
                base.model.layers[index] = replacements[index] if index in replacement_indices else originals[index]
            classify(base, chat.tokenizer, rows, label)
        for index in originals:
            base.model.layers[index] = replacements[index]
        if p.frozen_digest(chat.organism, chat.state) != source_digest:
            raise ValueError('Protected state changed')
    if p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(p.runtime.CHECKPOINT) != checkpoint_sha or p.sha(PRIOR) != prior_sha:
        raise ValueError('Protected input changed')
    parity = {label: sum(r[label]['top_token_id'] == old[previous]['top_token_id']
                         and r[label]['accepted'] == old[previous]['accepted'] for r, old in zip(rows, prior['rows']))
              for label, previous in (('original', 'original_base'), ('both', 'modified_base'))}
    summary = {}
    for label in ('original', 'layer_6', 'layer_12', 'both'):
        summary[label] = {'cases': len(rows),
                          'correct_choice': sum(r[label]['accepted'] == bool(r['label']) for r in rows),
                          'positive_accepted': sum(r[label]['accepted'] for r in rows if r['label']),
                          'negative_accepted': sum(r[label]['accepted'] for r in rows if not r['label']),
                          'label_top': sum(r[label]['top_token'] in ('是', '否') for r in rows)}
    result = {'summary': summary, 'rows': rows, 'parity_with_prior': parity,
              'base_sha256': base_sha, 'checkpoint_sha256': checkpoint_sha, 'prior_sha256': prior_sha,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Previously inspected authored development examples, not independent test.',
                              'Single forced-choice prompt; logits are not calibrated probabilities.',
                              'No active memory read; this is a language-layer ablation only.',
                              'Mixed one-layer configurations were not trained or validated as deployable models.',
                              'Decoder wrappers at memory sites are present but inactive in every condition.',
                              'Task-specific output changes do not measure broad language quality.']}
    with OUTPUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'summary': summary, 'parity': parity}), flush=True)


if __name__ == '__main__':
    main()
