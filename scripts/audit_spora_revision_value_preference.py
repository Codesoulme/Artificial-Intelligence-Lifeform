"""Test whether moving the correction alone changes new-vs-old value preference."""

import argparse
import json
from pathlib import Path
import re
from time import perf_counter

import torch

from scripts.audit_spora_revision_factors import SOURCE, TRANSFER, paired_prompts
from scripts.audit_spora_chat_backbones import reply
from scripts.benchmark_spora_recurrent_dialogue import (
    MODEL, AutoTokenizer, encode, extend_positions, frozen_digest, grade, load_full_streamed,
    load_integrated_runtime, sha, using,
)
from scripts.train_spora_hybrid_language_recovery import response_logits
from scripts.train_spora_dialogue_composition import settings_for
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.research_recurrent_decode import load_recurrent_replacements


def positions(tokenizer, far):
    history = far['history']
    base, update, filler = history[:2], history[2:4], history[4:]
    message = {'role': 'user', 'content': far['question']}
    histories = {'far': base + update + filler, 'near_update': base + filler + update,
                 'near_both': filler + base + update}
    prompts = {key: encode(tokenizer, [*turns, message]) for key, turns in histories.items()}
    if {len(ids) for ids in prompts.values()} != {far['input_tokens']}:
        raise ValueError('Moving the update changed the input length')
    return prompts, histories['near_update']


def preference(decoder, prompt, response):
    logits = response_logits(decoder, dict(prompt=prompt, response=response), cache=True).float()
    labels = torch.tensor(response, device=logits.device)
    logprob = logits.log_softmax(-1).gather(1, labels[:, None]).squeeze(-1)
    return dict(mean=float(logprob.mean()), tokens=logprob.tolist())


@torch.inference_mode()
def main(args):
    started = perf_counter()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = {path: sha(path) for path in (args.checkpoint, SOURCE, TRANSFER, args.reference,
        *(directory / name for directory in args.layers for name in ('attention.pt', 'comparison.json')))}
    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    if any(reference['sources'].get(str(path)) != digest for path, digest in sources.items()
           if path != args.reference):
        raise ValueError('Preference and generation comparisons must use identical inputs')
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    source['recall'] = json.loads(TRANSFER.read_text(encoding='utf-8'))
    prompts = [(row, positions(tokenizer, row)) for row in paired_prompts(tokenizer, source)
               if row['distance'] == 'far' and row['field'] == 'new']
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    base.lm_head.to(base.device)
    identity = frozen_digest(runtime.organism, runtime.state)
    if identity != reference['individual_sha256']:
        raise ValueError('Preference probe uses a different individual')
    replacements = load_recurrent_replacements(decoder, args.layers, sources[args.checkpoint])
    if set(replacements) != {19, 22}:
        raise ValueError('Expected recurrent layers 19 and 22')
    results = {}
    for label, mode in (('parent', 'parent'), ('19-22', 'recurrent')):
        rows = []
        settings = settings_for(decoder, tokenizer, 48)
        with using(replacements, mode):
            for row, (prompts_by_position, moved_update) in prompts:
                alternatives = {kind: tokenizer.encode(row['values'][kind], add_special_tokens=False)
                                for kind in ('new', 'old')}
                answer = reply(decoder, tokenizer, moved_update, row['question'], **settings)
                if answer['prompt_tokens'] != row['input_tokens']:
                    raise ValueError('Moved update generation received a different prompt length')
                answer['passed'] = grade(answer, [re.escape(row['values']['new'])], [re.escape(row['values']['old'])])
                for position, prompt in prompts_by_position.items():
                    scores = {kind: preference(decoder, prompt, tokens) for kind, tokens in alternatives.items()}
                    result = dict(case=row['case'], variant=row['variant'], position=position, scores=scores,
                                  margin=scores['new']['mean'] - scores['old']['mean'])
                    if position == 'near_update':
                        result['generation'] = answer
                    rows.append(result)
        results[label] = rows
        print(label, {key: sum(row['margin'] > 0 for row in rows if row['position'] == key)
                      for key in ('far', 'near_update', 'near_both')}, flush=True)
    if identity != frozen_digest(runtime.organism, runtime.state) or any(
            sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('A source or the individual changed during the comparison')
    store_json(args.output, dict(results=results, individual_sha256=identity,
               sources={str(path): digest for path, digest in sources.items()}, seconds=perf_counter() - started))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('reference', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('layers', type=Path, nargs=2)
    main(cli.parse_args())
