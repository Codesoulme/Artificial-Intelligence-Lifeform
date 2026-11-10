"""Compare whole-dialogue and mention-position probes on identical MMC pairs."""
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

from scripts.audit_mmc_pronoun_pairs import DEV, DEV_SHA, TRAIN, TRAIN_SHA, load
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_mmc_pair_probe import score, select_pairs
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_pooled_address import extract
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load as load_individual
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-mmc-mention-probe-20261110'


def mention_masks(tokenizer, texts):
    formatted = [tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False,
                                               add_generation_prompt=True) for text in texts]
    encoded = tokenizer(formatted, return_tensors='pt', padding=True, add_special_tokens=False,
                        return_offsets_mapping=True)
    masks = [torch.zeros_like(encoded['attention_mask'], dtype=torch.bool) for _ in range(2)]
    for i, (formatted_text, text) in enumerate(zip(formatted, texts)):
        position = formatted_text.find(text)
        if position < 0 or formatted_text.find(text, position + 1) >= 0:
            raise ValueError('Expected one unmodified input')
        previous_end = -1
        for index, tag in enumerate(('m1', 'm2')):
            opening, closing = f'<{tag}>', f'</{tag}>'
            if text.count(opening) != 1 or text.count(closing) != 1:
                raise ValueError('Expected unique mention markers')
            left = text.index(opening) + len(opening)
            right = text.index(closing)
            if not previous_end <= left < right:
                raise ValueError('Invalid mention marker order')
            previous_end = right + len(closing)
            left += position
            right += position
            offsets = encoded['offset_mapping'][i]
            mask = ((offsets[:, 0] < right) & (offsets[:, 1] > left) & encoded['attention_mask'][i].bool())
            if (not bool(mask.any()) or bool(((offsets[:, 0] < left) & mask).any())
                    or bool(((offsets[:, 1] > right) & mask).any())):
                raise ValueError('Mention token crosses boundary or is empty')
            masks[index][i] = mask
    return masks


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sources = {'weight': MODEL / 'model.safetensors', 'individual': INDIVIDUAL, 'train': TRAIN, 'dev': DEV}
    before = {name: sha(path) for name, path in sources.items()}
    if before != {'weight': WEIGHT_SHA, 'individual': INDIVIDUAL_SHA, 'train': TRAIN_SHA, 'dev': DEV_SHA}:
        raise ValueError('Protected source mismatch')
    train, dev = select_pairs(load(TRAIN, TRAIN_SHA), 64), select_pairs(load(DEV, DEV_SHA), 30)
    if {r['scene'] for r in train} & {r['scene'] for r in dev}:
        raise ValueError('Train/development scene overlap')
    config_sha = feature_files_sha256(MODEL)
    torch.manual_seed(4701)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load_individual(INDIVIDUAL, 'cpu')
    digest = frozen_digest(organism, original)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    vectors = {}
    for name, rows in (('train', train), ('dev', dev)):
        texts = [r['text'] for r in rows]
        features, all_mask = extract(base, tokenizer, texts, batch_size=2)
        masks = mention_masks(tokenizer, texts)
        if any(mask.shape != all_mask.shape or bool((mask & ~all_mask).any()) for mask in masks):
            raise ValueError('Mention masks disagree with extracted features')
        vectors[name] = {'whole': pool(features, all_mask),
                         'mentions': torch.cat([pool(features, mask) for mask in masks], dim=-1)}
        print(f'{name} features ready: {len(rows)}', flush=True)
    labels = torch.tensor([r['label'] for r in train])
    results, models = {'heuristic': score(dev, [r['heuristic'] for r in dev])}, {}
    for mode in ('whole', 'mentions'):
        torch.manual_seed(4702)
        classifier = nn.Linear(vectors['train'][mode].shape[-1], 2)
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.0002, weight_decay=0.01)
        for _ in range(600):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(classifier(vectors['train'][mode]), labels)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            train_predictions = classifier(vectors['train'][mode]).argmax(-1).tolist()
            predictions = classifier(vectors['dev'][mode]).argmax(-1).tolist()
        models[mode] = classifier.state_dict()
        results[mode] = {'train': score(train, train_predictions), 'dev': score(dev, predictions),
                         'predictions': [{'sample_sha256': row['sample_sha256'], 'label': row['label'],
                                          'predicted': prediction} for row, prediction in zip(dev, predictions)]}
    if (any(sha(path) != before[name] for name, path in sources.items()) or feature_files_sha256(MODEL) != config_sha
            or frozen_digest(organism, original) != digest or any(p.grad is not None for p in base.parameters())):
        raise ValueError('Protected source changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT / 'heads.pt'
    torch.save({'format': 'deepseek-mmc-mention-research-v1', 'states': models,
                'weight_sha256': WEIGHT_SHA, 'feature_config_sha256': config_sha, 'hidden_indices': [6, 18]}, checkpoint)
    report = {'status': 'mention_pair_development_only', 'results': results, 'heads_sha256': sha(checkpoint),
              'component_sha256': before, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Mention positions and speaker labels are supplied, not predicted or authenticated.',
                              'Same samples, different feature widths and separately optimized heads; not a pure masking ablation.',
                              'Same previously inspected development set; small balanced sample, single seed.',
                              'Different annotated clusters are treated as different; labels may be incomplete or noisy.',
                              'No facts, write authorization, live memory, new domains or chat quality evaluated.',
                              'Data rights, paraphrase overlap and model pretraining overlap remain unknown.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'heuristic': results['heuristic'], **{m: results[m]['dev'] for m in ('whole', 'mentions')}}), flush=True)


if __name__ == '__main__':
    run()
