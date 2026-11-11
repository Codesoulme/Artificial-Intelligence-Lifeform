"""Retrain finite name-to-SPORA-address binding on replaced features."""
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

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_named_binding_probe import (HELD_TEMPLATES, KEYS, NAMES, UNKNOWN, VALUES,
                                                   TRAIN_TEMPLATES, fit_binding, pooled, predict)
from scripts.deepseek_named_incremental_probe import NEW_KEYS
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_observation_transfer_audit import CHECKPOINT
from spora.integrated_checkpoint import load_integrated, replacement_feature_variant_sha256
from spora.structure_transfer import SPORALayerReplacement

OUTPUT = ROOT / 'runs/deepseek-spora-named-binding-retrained-20261111'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = {str(path): sha(path) for path in (MODEL / 'model.safetensors', CHECKPOINT)}
    if source[str(MODEL / 'model.safetensors')] != WEIGHT_SHA:
        raise ValueError('Base provenance mismatch')
    train = [(name, key, template.format(name=name)) for name, key in zip(NAMES, KEYS)
             for template in TRAIN_TEMPLATES]
    held = [(name, key, template.format(name=name)) for name, key in zip(NAMES, KEYS)
            for template in HELD_TEMPLATES]
    unknown = [(name, None, template.format(name=name)) for name in UNKNOWN for template in HELD_TEMPLATES]
    if set(row[2] for row in train) & set(row[2] for row in held + unknown):
        raise ValueError('Training and evaluation text overlap')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    decoder, organism, state = load_integrated(CHECKPOINT, base, MODEL, individual_device='cuda')
    del decoder, state
    blocks = {i: layer for i, layer in enumerate(base.model.layers) if isinstance(layer, SPORALayerReplacement)}
    variant = replacement_feature_variant_sha256(MODEL, blocks)
    bases = torch.stack([F.normalize(site.current_basis.detach().float().cpu(), dim=-1) for site in organism.memories])
    train_features, train_masks = extract(base, tokenizer, [row[2] for row in train], batch_size=1)
    eval_rows = held + unknown
    eval_features, eval_masks = extract(base, tokenizer, [row[2] for row in eval_rows], batch_size=1)
    train_vectors, eval_vectors = pooled(train_features, train_masks), pooled(eval_features, eval_masks)
    targets = torch.stack([bases[:, key] for _, key, _ in train])
    weights = fit_binding(train_vectors, targets)
    train_indices, train_scores = predict(train_vectors, weights, bases)
    eval_indices, eval_scores = predict(eval_vectors, weights, bases)
    train_labels = torch.tensor([key for _, key, _ in train])
    train_correct = (train_indices == train_labels[:, None]).all(-1)
    threshold = float(train_scores.min()) - 1e-6
    rows = []
    for (name, expected, question), indices, scores in zip(eval_rows, eval_indices, eval_scores):
        predicted = indices.tolist()
        similarities = scores.tolist()
        accepted = bool(min(similarities) >= threshold and predicted[0] == predicted[1])
        rows.append({'name': name, 'question': question, 'expected_key': expected, 'predicted': predicted,
                     'similarities': similarities, 'accepted': accepted,
                     'address_correct': expected is not None and accepted and predicted == [expected, expected]})
    summary = {'train_examples': len(train), 'train_correct_both': int(train_correct.sum()),
               'held_examples': len(held), 'held_address_correct': sum(row['address_correct'] for row in rows[:len(held)]),
               'unknown_examples': len(unknown), 'unknown_false_accept': sum(row['accepted'] for row in rows[len(held):])}
    if any(sha(Path(path)) != digest for path, digest in source.items()):
        raise ValueError('Protected input changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT / 'binding.pt'
    torch.save({'format': 'spora-replaced-named-binding-development-v1', 'weights': weights,
                'threshold': threshold, 'base_sha256': WEIGHT_SHA, 'source_checkpoint_sha256': source[str(CHECKPOINT)],
                'feature_variant_sha256': variant, 'hidden_indices': [6, 18]}, checkpoint)
    report = {'status': 'development_only_no_writes', 'summary': summary, 'binding_sha256': sha(checkpoint),
              'source_hashes': source, 'feature_variant_sha256': variant, 'replacement_layers': sorted(blocks),
              'rows': rows, 'seconds': time.perf_counter() - started,
              'limitations': ['Eight supervised names, four unknown names and two held templates per name.',
                              'Address projection is finite and does not create new memory keys.',
                              'Questions are synthetic; no ordinary dialogue or source semantics.',
                              'Threshold is estimated from training examples and no persistent write occurred.']}
    (OUTPUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
