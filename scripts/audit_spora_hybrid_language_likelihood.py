"""Score held-out Chinese responses on actual discrete attention layouts."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import torch

from scripts.audit_spora_crosswoz_positive_fields import PINNED
from scripts.benchmark_spora_recurrent_dialogue import (
    MODEL, AutoTokenizer, encode, extend_positions, frozen_digest, load_full_streamed,
    load_integrated_runtime, sha, using,
)
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.research_recurrent_decode import load_recurrent_replacements


def cases(archive, tokenizer, count, *, split='test'):
    if sha(archive) != PINNED[split]:
        raise ValueError(f'Unexpected CrossWOZ official {split} archive')
    with ZipFile(archive) as source:
        records = json.loads(source.read(f'{split}.json'))
    ordered = sorted(records, key=lambda key: sha256(('hybrid-language-v1/' + key).encode()).digest())
    selected = []
    for key in ordered:
        messages = records[key]['messages']
        if len(messages) < 6:
            continue
        turns = messages[:6]
        if [turn['role'] for turn in turns] != ['usr', 'sys'] * 3:
            continue
        history = [{'role': 'user' if row['role'] == 'usr' else 'assistant', 'content': row['content']}
                   for row in turns[:5]]
        prompt = encode(tokenizer, history)
        response = tokenizer.encode(turns[5]['content'], add_special_tokens=False)
        if 10 <= len(response) <= 96 and 80 <= len(prompt) <= 384:
            selected.append(dict(dialogue=key, prompt=prompt, response=response))
        if len(selected) == count:
            return selected
    raise ValueError('Not enough qualifying dialogues')


@torch.inference_mode()
def score(decoder, rows):
    result = []
    for row in rows:
        prompt, response = row['prompt'], row['response']
        ids = torch.tensor([prompt + response[:-1]], device=decoder.base.device)
        output = decoder(input_ids=ids, use_cache=True, logits_to_keep=len(response))
        logprobs = output.logits[0].float().log_softmax(-1)
        if logprobs.shape[0] != len(response):
            raise RuntimeError('Logit slice did not match answer tokens')
        gold = torch.tensor(response, device=ids.device)
        loss = -logprobs.gather(1, gold[:, None]).sum()
        result.append(dict(dialogue=row['dialogue'], prompt_tokens=len(prompt), answer_tokens=len(response),
                           nll_sum=float(loss), correct_tokens=int((logprobs.argmax(-1) == gold).sum())))
    total = sum(row['answer_tokens'] for row in result)
    return dict(rows=result, answer_tokens=total,
                mean_nll=sum(row['nll_sum'] for row in result) / total,
                accuracy=sum(row['correct_tokens'] for row in result) / total)


def main(args):
    started = perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    protected = {path: sha(path) for path in (
        args.checkpoint, args.archive, *(folder / fname for folder in args.layers
                                          for fname in ('attention.pt', 'comparison.json')))}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    selected = cases(args.archive, tokenizer, args.count)
    torch.set_num_threads(2)
    model = load_full_streamed(stage_output_projection=True)
    rotary = model.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, model, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    model.lm_head.to(model.device)
    replacements = load_recurrent_replacements(decoder, args.layers, protected[args.checkpoint])
    if set(replacements) != {19, 21, 22, 24}:
        raise ValueError('Expected four audited migration candidates')
    identity = frozen_digest(runtime.organism, runtime.state)
    results = {}
    for layout in ((), (19, 22), (19, 21, 22), (19, 22, 24)):
        name = '-'.join(map(str, layout)) or 'parent'
        with using({index: replacements[index] for index in layout}, 'recurrent'):
            results[name] = score(decoder, selected)
        print(name, {key: value for key, value in results[name].items() if key != 'rows'}, flush=True)
    if identity != frozen_digest(runtime.organism, runtime.state) or any(
            sha(path) != checksum for path, checksum in protected.items()):
        raise RuntimeError('Source weights or individual changed')
    store_json(args.output, dict(results=results, cases=selected, individual_sha256=identity,
               sources={str(path): checksum for path, checksum in protected.items()}, seconds=perf_counter() - started))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('archive', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('layers', nargs=4, type=Path)
    cli.add_argument('--count', type=int, default=12)
    args = cli.parse_args()
    if args.output.exists() or args.count < 1:
        cli.error('Choose a new output and positive count')
    main(args)
