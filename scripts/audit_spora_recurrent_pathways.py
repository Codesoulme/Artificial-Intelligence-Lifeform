"""Test whether far recurrent reads help when full attention cannot see early turns."""

import argparse
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_spora_recurrent_history_ablation import summarize
from scripts.benchmark_spora_recurrent_dialogue import (
    MODEL, SOURCE, AutoTokenizer, encode, extend_positions, frozen_digest, load_full_streamed,
    load_integrated_runtime, recall_prompts, run_dialogue, sha, torch,
)
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.research_recurrent_decode import attention_block, load_recurrent_replacements, replace


BOUNDARY = 512
POSITIONS = 2048


def segment_mask(start, count, boundary, device):
    end = start + count
    keys = torch.arange(end, device=device)
    queries = keys[start:]
    same_segment = (queries[:, None] >= boundary) == (keys[None, :] >= boundary)
    return (same_segment & (keys[None, :] <= queries[:, None]))[None, None]


@contextmanager
def restrict_full_attention(decoder, boundary, excluded):
    last_shape, allowed = None, None

    def restrict(layer, args, kwargs):
        nonlocal last_shape, allowed
        hidden = kwargs['hidden_states']
        cache = kwargs.get('past_key_values')
        start = cache.get_seq_length(layer.layer_idx) if cache is not None else 0
        count = hidden.shape[1]
        shape = (start, count, hidden.device)
        if shape != last_shape:
            positions = kwargs.get('cache_position')
            if hidden.shape[0] != 1 or positions is None or not torch.equal(
                    positions, torch.arange(start, start + count, device=hidden.device)):
                raise ValueError('Segment probe requires one append-only sequence')
            allowed = segment_mask(start, count, boundary, hidden.device)
            last_shape = shape
        prior = kwargs.get('attention_mask')
        if prior is None:
            mask = allowed
        elif prior.dtype == torch.bool:
            mask = prior[..., :start + count] & allowed
        else:
            mask = prior[..., :start + count].masked_fill(~allowed, torch.finfo(prior.dtype).min)
        return args, {**kwargs, 'attention_mask': mask}

    with ExitStack() as hooks:
        for index in range(len(decoder.base.model.layers)):
            if index not in excluded:
                module = attention_block(decoder, index).self_attn
                hook = module.register_forward_pre_hook(restrict, with_kwargs=True)
                hooks.callback(hook.remove)
        yield


def main(args):
    started = perf_counter()
    paths = [args.checkpoint, SOURCE, *(folder / name for folder in args.candidates
                                       for name in ('attention.pt', 'comparison.json'))]
    protected = {path: sha(path) for path in (*paths, args.reference)}
    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    if (reference['position_table'] != POSITIONS
            or any(reference['sources'].get(str(path)) != protected[path] for path in paths)):
        raise ValueError('The benchmark reference used different inputs or position budget')
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    prompts = [row for row in recall_prompts(tokenizer, source, extended=True) if row['length'] == 'extended']
    original = [row for row in reference['dialogue']['recurrent']['recall'] if row['length'] == 'extended']
    for prompt, saved in zip(prompts, original, strict=True):
        question = {'role': 'user', 'content': prompt['question']}
        evidence_length = len(encode(tokenizer, [*prompt['history'][:4],
                                                  question]))
        tokens = encode(tokenizer, [*prompt['history'], question])
        suffix = tokenizer.decode(tokens[BOUNDARY:], skip_special_tokens=True)
        if (any(saved[key] != value for key, value in prompt.items())
                or not evidence_length + 16 < BOUNDARY < len(tokens)
                or any(value in suffix for value in prompt['values'].values())):
            raise ValueError('The prefix boundary must follow the original events and precede the final question')
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, POSITIONS)
    torch.cuda.empty_cache()
    base.lm_head.to(base.device)
    individual = frozen_digest(runtime.organism, runtime.state)
    if individual != reference['individual_sha256']:
        raise ValueError('The benchmark reference used a different individual')
    candidates = load_recurrent_replacements(decoder, args.candidates, protected[args.checkpoint])
    if tuple(candidates) != tuple(reference['layers']):
        raise ValueError('The benchmark reference used a different layer placement')
    results = {'unrestricted': {'rows': original, 'summary': summarize(original)['extended']}}
    for label, selected, far in (('restricted_parent', {}, False),
                                 ('restricted_with_far', candidates, True),
                                 ('restricted_without_far', candidates, False)):
        with ExitStack() as hooks:
            hooks.enter_context(restrict_full_attention(decoder, BOUNDARY, selected))
            for block, candidate in selected.values():
                hooks.callback(setattr, candidate, 'include_far', candidate.include_far)
                candidate.include_far = far
                hooks.enter_context(replace(block, candidate))
            rows = run_dialogue(decoder, tokenizer, prompts, [])['recall']
        results[label] = {'rows': rows, 'summary': summarize(rows)['extended']}
        print(label, results[label]['summary'], flush=True)
    if (individual != frozen_digest(runtime.organism, runtime.state)
            or any(sha(path) != value for path, value in protected.items())):
        raise ValueError('A probe input or individual changed')
    store_json(args.output, {'conditions': results, 'boundary': BOUNDARY, 'positions': POSITIONS,
                             'individual_sha256': individual, 'layers': list(candidates),
                             'sources': {str(path): value for path, value in protected.items()},
                             'seconds': perf_counter() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('candidates', nargs=2, type=Path)
    options = parser.parse_args()
    if options.output.exists():
        parser.error('Choose a new output path')
    main(options)
