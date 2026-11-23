"""Measure whether a short dialogue demonstration transfers to held-out turns."""

import argparse
import json
from functools import partial
from pathlib import Path

from scripts.audit_local_named_chat import CASES, FRESH_CASES, run
from scripts.audit_spora_chat_backbones import reply
from scripts.deepseek_local_baseline import MODEL, sha, torch
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_checkpoint import load_integrated_runtime
from spora.language_training import frozen_digest
from transformers import AutoTokenizer


DEMONSTRATION = (
    {'role': 'user', 'content': '预约单上写着周末取相机。用一句话告诉我安排。'},
    {'role': 'assistant', 'content': '周末取相机。'},
    {'role': 'user', 'content': '相机不借了，改为取三脚架。只回器材名。'},
    {'role': 'assistant', 'content': '三脚架'},
)


def main(checkpoint, output):
    if output.exists():
        raise FileExistsError(output)
    digest = sha(checkpoint)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    before = frozen_digest(runtime.organism, runtime.state)
    generate = partial(reply, runtime.decoder, tokenizer, device='cuda', deepseek=True, limit=96,
                       eos=[tokenizer.eos_token_id])
    results = {}
    for name, prefix in (('plain', ()), ('example', DEMONSTRATION)):
        def answer(history, question):
            return generate([*prefix, *history], question)['answer']

        results[name] = {}
        for label, cases in (('original', CASES), ('fresh', FRESH_CASES)):
            result = run(runtime.decoder, tokenizer, cases=cases, reply_with_history=answer)
            results[name][label] = result
            print(name, label, result['cases_passed'], result['turns_passed'], flush=True)
        torch.cuda.empty_cache()
    if sha(checkpoint) != digest or frozen_digest(runtime.organism, runtime.state) != before:
        raise ValueError('Experiment modified the source individual')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump({'checkpoint_sha256': digest, 'individual_sha256': before, 'results': results},
                  stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(**vars(parser.parse_args()))
