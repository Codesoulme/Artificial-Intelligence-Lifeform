"""Compare natural and closed-reasoning generation on a small development sample."""
import json
import re
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
from scripts.deepseek_mmc_choice_baseline import balanced_subset
from scripts.deepseek_mmc_pair_probe import select_pairs
from scripts.deepseek_mmc_sequence_baseline import make_prompt
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load as load_individual
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-mmc-generation-baseline-20261110.json'


def parse_answer(text, mode, ended):
    if type(ended) is not bool or mode not in ('normal', 'closed'):
        raise ValueError('Invalid generation metadata')
    if not ended:
        return None, 'not_terminated'
    if mode == 'normal':
        if text.count('</think>') != 1:
            return None, 'reasoning_not_closed'
        text = text.split('</think>', 1)[1]
    text = text.strip()
    if re.fullmatch(r'是[。.!！]?', text):
        return 1, 'parsed'
    if re.fullmatch(r'否[。.!！]?', text):
        return 0, 'parsed'
    return None, 'unrecognized_answer'


def summarize(rows, mode):
    selected = [row for row in rows if row['mode'] == mode]
    parsed = [row for row in selected if row['prediction'] is not None]
    return {'samples': len(selected), 'terminated': sum(row['ended'] for row in selected),
            'parsed': len(parsed), 'correct': sum(row['prediction'] == row['label'] for row in parsed),
            'wrong_same': sum(row['prediction'] == 1 and row['label'] == 0 for row in parsed),
            'wrong_different': sum(row['prediction'] == 0 and row['label'] == 1 for row in parsed),
            'reasons': {reason: sum(row['reason'] == reason for row in selected)
                        for reason in ('parsed', 'not_terminated', 'reasoning_not_closed', 'unrecognized_answer')}}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = {'weight': MODEL / 'model.safetensors', 'individual': INDIVIDUAL, 'dev': DEV}
    before = {name: sha(path) for name, path in files.items()}
    if before != {'weight': WEIGHT_SHA, 'individual': INDIVIDUAL_SHA, 'dev': DEV_SHA}:
        raise ValueError('Protected source mismatch')
    samples = balanced_subset(select_pairs(load(DEV, DEV_SHA), 30, current_sentence_tail=True), per_class=4)
    config_sha = feature_files_sha256(MODEL)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load_individual(INDIVIDUAL, 'cpu')
    digest = frozen_digest(organism, original)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    results = []
    with torch.inference_mode():
        for index, sample in enumerate(samples):
            prompt = make_prompt(sample['text'], False)
            for mode in ('closed', 'normal'):
                if mode == 'closed':
                    inputs = closed_think_inputs(tokenizer, prompt, 'cuda')
                else:
                    formatted = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                                               add_generation_prompt=True)
                    inputs = tokenizer(formatted, return_tensors='pt', add_special_tokens=False).to('cuda')
                tokens = base.generate(**inputs, do_sample=False, max_new_tokens=128 if mode == 'normal' else 32,
                                       pad_token_id=tokenizer.eos_token_id)[0, inputs['input_ids'].shape[1]:]
                ended = tokenizer.eos_token_id in tokens.tolist()
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                prediction, reason = parse_answer(text, mode, ended)
                results.append({'sample_sha256': sample['sample_sha256'], 'label': sample['label'], 'mode': mode,
                                'text': text, 'generated_tokens': len(tokens), 'ended': ended,
                                'prediction': prediction, 'reason': reason})
                print(f'evaluated {index + 1}/{len(samples)} {mode}', flush=True)
    if (any(sha(path) != before[name] for name, path in files.items()) or feature_files_sha256(MODEL) != config_sha
            or frozen_digest(organism, original) != digest or any(p.grad is not None for p in base.parameters())):
        raise ValueError('Protected source changed')
    report = {'status': 'natural_generation_development_only', 'results': results,
              'summary': {mode: summarize(results, mode) for mode in ('closed', 'normal')},
              'component_sha256': before, 'feature_config_sha256': config_sha,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Eight previously inspected balanced development examples, not final evaluation.',
                              'Greedy generation only; normal mode limited to 128 tokens and closed mode to 32.',
                              'Only exact yes/no responses after a terminated generation are classified.',
                              'Both modes use supplied mention positions and speaker identifiers.',
                              'Different clusters may reflect incomplete or inaccurate annotation.',
                              'Frozen base without SPORA memory; no identity, factual writing or live dialogue tested.',
                              'Raw generated answers remain local; subtitle data rights and pretraining overlap unknown.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    run()
