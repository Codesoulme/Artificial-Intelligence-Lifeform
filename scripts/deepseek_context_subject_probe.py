"""Compare current-turn pooling with whole-context pooling for subject probes."""
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

from scripts.deepseek_dulemon_subject_annotation import LABELS, SOURCE, SOURCE_SHA, compare
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_relative_identity_probe import AMBIGUOUS_TRAIN, HELD, NOVEL, TRAIN, build_cases
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-context-subject-20261110'
PREVIOUS = ('你最近过得怎么样？', '我听明白了。', '我们换个话题吧。', '请继续说。')


def compose(previous_role, previous, role, current):
    if previous_role not in ('Usr', 'Bot') or role not in ('Usr', 'Bot') or previous_role == role:
        raise ValueError('Expected alternating dialogue roles')
    if not previous or not current:
        raise ValueError('Empty dialogue turn')
    prefix = f'前一句（{previous_role}）：{previous}\n当前句（{role}）：'
    return prefix + current, len(prefix)


def current_mask(tokenizer, combined, starts, current_texts):
    formatted = [tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False,
                                               add_generation_prompt=True) for text in combined]
    encoded = tokenizer(formatted, return_tensors='pt', padding=True, add_special_tokens=False,
                        return_offsets_mapping=True)
    mask = torch.zeros_like(encoded['attention_mask'], dtype=torch.bool)
    for i, (text, prefix, current) in enumerate(zip(formatted, starts, current_texts)):
        offset = text.find(combined[i])
        if offset < 0 or text.find(combined[i], offset + 1) >= 0:
            raise ValueError('Expected one formatted conversation')
        left, right = offset + prefix, offset + prefix + len(current)
        offsets = encoded['offset_mapping'][i]
        mask[i] = (offsets[:, 0] < right) & (offsets[:, 1] > left) & encoded['attention_mask'][i].bool()
        if not bool(mask[i].any()) or bool(((offsets[:, 0] < left) & mask[i]).any()):
            raise ValueError('Current turn has no tokens or crosses its boundary')
    return mask


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = {'weight': MODEL / 'model.safetensors', 'individual': INDIVIDUAL, 'audit': SOURCE}
    before = {key: sha(path) for key, path in files.items()}
    if before['weight'] != WEIGHT_SHA or before['individual'] != INDIVIDUAL_SHA or before['audit'] != SOURCE_SHA:
        raise ValueError('Protected source mismatch')
    with SOURCE.open(encoding='utf-8') as stream:
        old = json.load(stream)
    audit = old['results']['single']['baseline']['rows'][:len(LABELS)]
    training = build_cases(TRAIN) + build_cases(AMBIGUOUS_TRAIN)
    held = build_cases(HELD)
    novel = build_cases(NOVEL)
    splits = {'train': training, 'held': held, 'novel': novel, 'audit': audit}
    config_sha = feature_files_sha256(MODEL)
    torch.manual_seed(4645)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    digest = frozen_digest(organism, original)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    vectors = {}
    for split, rows in splits.items():
        pairs = []
        for i, row in enumerate(rows):
            if split == 'audit':
                pairs.append(compose(row['previous_role'], row['previous_text'], row['role'], row['text']))
            else:
                role = 'Usr' if i % 2 == 0 else 'Bot'
                pairs.append(compose('Bot' if role == 'Usr' else 'Usr', PREVIOUS[(i // 2) % len(PREVIOUS)],
                                     role, row['text']))
        combined, starts = zip(*pairs)
        features, all_masks = extract(base, tokenizer, list(combined), batch_size=2)
        masks = current_mask(tokenizer, combined, starts, [r['text'] for r in rows])
        if masks.shape != all_masks.shape or bool((masks & ~all_masks).any()):
            raise ValueError('Current-turn mask mismatch')
        vectors[split] = {'whole': pool(features, all_masks), 'current': pool(features, masks)}
        print(f'{split} features ready: {len(rows)}', flush=True)
    labels = torch.tensor([r['label'] for r in training])
    results, models = {}, {}
    for mode in ('whole', 'current'):
        torch.manual_seed(4646)
        classifier = nn.Linear(vectors['train'][mode].shape[-1], 3)
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.0002, weight_decay=0.01)
        for _ in range(600):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(classifier(vectors['train'][mode]), labels)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            loss.backward()
            optimizer.step()
        classifier.eval()
        models[mode] = classifier.state_dict()
        results[mode] = {}
        with torch.no_grad():
            for split in ('held', 'novel', 'audit'):
                predictions = classifier(vectors[split][mode]).argmax(-1).tolist()
                expected = [item[0] for item in LABELS] if split == 'audit' else [r['label'] for r in splits[split]]
                rows = [{'sample_sha256': row.get('sample_sha256', str(i)), 'predicted': int(prediction)}
                        for i, (row, prediction) in enumerate(zip(splits[split], predictions))]
                results[mode][split] = compare(rows, [(label, '') for label in expected])
    if (any(sha(path) != before[name] for name, path in files.items()) or feature_files_sha256(MODEL) != config_sha
            or frozen_digest(organism, original) != digest or any(p.grad is not None for p in base.parameters())):
        raise ValueError('Protected source changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT / 'heads.pt'
    torch.save({'format': 'deepseek-context-subject-research-v1', 'states': models, 'weight_sha256': WEIGHT_SHA,
                'feature_config_sha256': config_sha, 'hidden_indices': [6, 18]}, checkpoint)
    report = {'status': 'context_pooling_research_only', 'results': results, 'heads_sha256': sha(checkpoint),
              'source_sha256': SOURCE_SHA, 'component_sha256': before, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Synthetic train context has neutral preceding turns; not natural multi-turn training.',
                              'The same 30 agent-annotated development examples have no independent human review.',
                              'Current-token mask uses a manually specified turn boundary, not automated parsing.',
                              'The two pooling modes train separate heads; experiment does not isolate pooling from optimization.',
                              'No writing, identity authentication, multi-session memory or open-domain safety tested.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({mode: {split: {k: v for k, v in stats.items() if k != 'rows'}
                                  for split, stats in value.items()} for mode, value in results.items()}), flush=True)


if __name__ == '__main__':
    run()
