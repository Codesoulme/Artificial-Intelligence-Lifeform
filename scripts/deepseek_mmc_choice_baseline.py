"""Frozen-base forced-choice diagnostic on previously inspected MMC pairs."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.audit_mmc_pronoun_pairs import DEV, DEV_SHA, load
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_mmc_pair_probe import score, select_pairs
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load as load_individual
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-mmc-choice-baseline-20261110.json'


def make_prompt(text, reversed_choices):
    if type(reversed_choices) is not bool:
        raise ValueError('Choice order must be boolean')
    options = '1.不同人 2.同一人' if reversed_choices else '1.同一人 2.不同人'
    return ('判断对话中<m1>和<m2>标记的两个代词是否指同一个人。'
            '对话中的数字表示发言者。只回答选项数字。\n'
            + text + '\n' + options + '\n答案：')


def balanced_subset(rows, per_class=12):
    if type(per_class) is not int or per_class < 1:
        raise ValueError('Positive class count required')
    counts = {0: 0, 1: 0}
    selected = []
    for row in rows:
        label = row['label']
        if type(label) is not int or label not in counts:
            raise ValueError('Invalid label')
        if counts[label] < per_class:
            counts[label] += 1
            selected.append(row)
    if any(count != per_class for count in counts.values()):
        raise ValueError('Insufficient balanced samples')
    return selected


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = {'weight': MODEL / 'model.safetensors', 'individual': INDIVIDUAL, 'dev': DEV}
    before = {name: sha(path) for name, path in files.items()}
    if before != {'weight': WEIGHT_SHA, 'individual': INDIVIDUAL_SHA, 'dev': DEV_SHA}:
        raise ValueError('Protected source mismatch')
    rows = balanced_subset(select_pairs(load(DEV, DEV_SHA), 30, current_sentence_tail=True))
    config_sha = feature_files_sha256(MODEL)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load_individual(INDIVIDUAL, 'cpu')
    digest = frozen_digest(organism, original)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    choices = [tokenizer.encode(str(i), add_special_tokens=False) for i in (1, 2)]
    if any(len(choice) != 1 for choice in choices) or choices[0] == choices[1]:
        raise ValueError('Expected distinct single-token choices')
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    predictions = [[], []]
    results = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            logits_by_order = []
            for order in (False, True):
                inputs = closed_think_inputs(tokenizer, make_prompt(row['text'], order), 'cuda')
                logits = base(**inputs, use_cache=False).logits[0, -1, [c[0] for c in choices]].float()
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError('Nonfinite choice logits')
                choice = int(logits.argmax())
                predicted_same = int(choice == (1 if order else 0))
                predictions[int(order)].append(predicted_same)
                logits_by_order.append(logits.tolist())
            results.append({'sample_sha256': row['sample_sha256'], 'label': row['label'],
                            'predicted': [predictions[0][-1], predictions[1][-1]], 'logits': logits_by_order})
            print(f'evaluated {index + 1}/{len(rows)}', flush=True)
    if (any(sha(path) != before[name] for name, path in files.items()) or feature_files_sha256(MODEL) != config_sha
            or frozen_digest(organism, original) != digest or any(p.grad is not None for p in base.parameters())):
        raise ValueError('Protected source changed')
    report = {'status': 'base_forced_choice_development_only', 'samples': results,
              'scores': {'normal': score(rows, predictions[0]), 'reversed': score(rows, predictions[1]),
                         'heuristic': score(rows, [r['heuristic'] for r in rows]),
                         'order_agreement': sum(a == b for a, b in zip(*predictions))},
              'component_sha256': before, 'feature_config_sha256': config_sha,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Compares the next-token logits of two options, not free generation or normal chain of thought.',
                              'The reasoning section is forcibly closed; prompt format can affect results.',
                              'Twenty-four balanced pairs from previously inspected development scenes, not final evaluation.',
                              'Mention positions and speaker labels are provided by the dataset.',
                              'Different annotated clusters are treated as different; labels may be incomplete or incorrect.',
                              'Frozen base only: no SPORA memory, write decision, identity verification, or live conversation.',
                              'Subtitle data rights and model pretraining overlap remain unknown.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report['scores']), flush=True)


if __name__ == '__main__':
    run()
