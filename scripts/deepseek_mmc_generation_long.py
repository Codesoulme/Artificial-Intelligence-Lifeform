"""Revisit four prior truncated generations with a larger reasoning budget."""
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
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_mmc_choice_baseline import balanced_subset
from scripts.deepseek_mmc_generation_baseline import OUTPUT as SHORT, parse_answer, summarize
from scripts.deepseek_mmc_pair_probe import select_pairs
from scripts.deepseek_mmc_sequence_baseline import make_prompt
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load as load_individual
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-mmc-generation-long-20261110.json'


def previous_samples(short_report, samples):
    previous = {r['sample_sha256']: r for r in short_report['results'] if r['mode'] == 'normal'}
    if len(previous) != sum(r['mode'] == 'normal' for r in short_report['results']):
        raise ValueError('Duplicate previous generation')
    for sample in samples:
        old = previous.get(sample['sample_sha256'])
        if old is None or old['label'] != sample['label'] or old['ended'] or old['generated_tokens'] != 128:
            raise ValueError('Expected same previously truncated sample')
    return previous


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = {'weight': MODEL / 'model.safetensors', 'individual': INDIVIDUAL, 'dev': DEV, 'short': SHORT}
    before = {name: sha(path) for name, path in files.items()}
    if before['weight'] != WEIGHT_SHA or before['individual'] != INDIVIDUAL_SHA or before['dev'] != DEV_SHA:
        raise ValueError('Protected source mismatch')
    with SHORT.open(encoding='utf-8') as stream:
        short_report = json.load(stream)
    if (short_report.get('status') != 'natural_generation_development_only'
            or any(short_report['component_sha256'].get(k) != before[k] for k in ('weight', 'individual', 'dev'))):
        raise ValueError('Prior report provenance mismatch')
    samples = balanced_subset(select_pairs(load(DEV, DEV_SHA), 30, current_sentence_tail=True), per_class=2)
    previous = previous_samples(short_report, samples)
    config_sha = feature_files_sha256(MODEL)
    if short_report.get('feature_config_sha256') != config_sha:
        raise ValueError('Prior tokenizer/config mismatch')
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
            formatted = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                                       add_generation_prompt=True)
            inputs = tokenizer(formatted, return_tensors='pt', add_special_tokens=False).to('cuda')
            start = time.perf_counter()
            tokens = base.generate(**inputs, do_sample=False, max_new_tokens=512,
                                   pad_token_id=tokenizer.eos_token_id)[0, inputs['input_ids'].shape[1]:]
            ended = tokenizer.eos_token_id in tokens.tolist()
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            prediction, reason = parse_answer(text, 'normal', ended)
            results.append({'sample_sha256': sample['sample_sha256'], 'label': sample['label'], 'mode': 'normal',
                            'text': text, 'generated_tokens': len(tokens), 'ended': ended,
                            'prediction': prediction, 'reason': reason,
                            'short_text_is_prefix': text.startswith(previous[sample['sample_sha256']]['text']),
                            'seconds': time.perf_counter() - start})
            print(f'evaluated {index + 1}/{len(samples)}: {len(tokens)} tokens, ended={ended}', flush=True)
    if (any(sha(path) != before[name] for name, path in files.items()) or feature_files_sha256(MODEL) != config_sha
            or frozen_digest(organism, original) != digest or any(p.grad is not None for p in base.parameters())):
        raise ValueError('Protected source changed')
    report = {'status': 'extended_generation_development_only', 'results': results,
              'summary': summarize(results, 'normal'), 'component_sha256': before,
              'feature_config_sha256': config_sha, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Four previously inspected balanced development examples; not final evaluation.',
                              'Greedy decoding with 512-token limit; unended or non-exact answers remain unscored.',
                              'Mention positions and speaker identifiers are supplied; annotations may be incomplete or wrong.',
                              'Frozen base only; no SPORA memory, identity authentication or write decision evaluated.',
                              'Generated output remains local; subtitle data rights and pretraining overlap unknown.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    run()
