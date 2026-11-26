"""Compare residual-based and recent-only exact retention with equal KV budgets."""

import argparse
from contextlib import contextmanager
import json
from math import log
from pathlib import Path
from time import perf_counter

import torch

from scripts.benchmark_spora_recurrent_dialogue import using
from scripts.train_spora_hybrid_hidden_recovery import (
    HELD, LAYERS, MODEL, SOURCE, TARGET_LAYER, AutoTokenizer, evaluate, extend_positions, frozen_digest,
    load_full_streamed, load_integrated_runtime, load_recurrent_replacements, recall_prompts, records,
    run_dialogue, sha, store_json,
)
from spora.research_recurrent_attention import GatedDeltaRecurrentAttention, SelectiveGatedDeltaAttention


class EvidenceIntervention(SelectiveGatedDeltaAttention):
    """Inject annotated positions in evaluation to separate selection from retrieval."""

    def entry_scores(self, positions, residual, write):
        scores = super().entry_scores(positions, residual, write)
        return scores.masked_fill(torch.isin(positions, self.prioritized), float('inf'))

    def retained_attention_mask(self, local_mask, retained):
        mask = super().retained_attention_mask(local_mask, retained)
        if self.read_multiplier == 1.:
            return mask
        groups = self.q_map.shape[0] // retained.positions.shape[1]
        selected = retained.scores.isposinf().repeat_interleave(groups, dim=1)
        bias = torch.nn.functional.pad(selected.float() * log(self.read_multiplier), (0, local_mask.shape[-1]))
        causal = torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, float('-inf'))
        return causal + bias.unsqueeze(-2)


def annotations(tokenizer, row):
    messages = [*row['history'], {'role': 'user', 'content': row['question']}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + '</think>\n\n'
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    if len(encoded.input_ids) != row['input_tokens']:
        raise ValueError('Answer generation and annotation tokenization differ')
    spans = {}
    for field, turn in (('old', 0), ('new', 2), ('keep', 0)):
        utterance = row['history'][turn]['content']
        value = row['values'][field]
        if prompt.count(utterance) != 1 or utterance.count(value) != 1:
            raise ValueError(f'A {field} field did not identify a unique span of its original turn')
        start = prompt.index(utterance) + utterance.index(value)
        end = start + len(value)
        spans[field] = {index for index, (left, right) in enumerate(encoded.offset_mapping)
                        if left < end and start < right}
        if not spans[field]:
            raise ValueError('Tokenized field has no overlapping tokens')
    return spans


@contextmanager
def observe_prefill(core, count):
    captures = []

    def capture(_module, args, kwargs, result):
        if kwargs.get('state') is not None or len(captures) >= count:
            return
        _, state = result
        slots = state.retained
        captures.append((state.length, [] if slots is None else slots.positions[0].tolist(),
                         0 if slots is None else sum(tensor.untyped_storage().nbytes() for tensor in slots.tensors())))

    hook = core.register_forward_hook(capture, with_kwargs=True)
    try:
        yield captures
    finally:
        hook.remove()


def coverage(prompts, spans, observed):
    rows = []
    for prompt, fields, (length, positions, size) in zip(prompts, spans, observed, strict=True):
        if length != prompt['input_tokens']:
            raise ValueError('Prefill observations and evaluation prompts are misaligned')
        groups = [set(group) for group in positions]
        rows.append(dict(case=prompt['case'], variant=prompt['variant'], length=prompt['length'],
                         slot_bytes=size, slots=positions, fields={
                             field: dict(tokens=sorted(indices), any=any(indices & group for group in groups),
                                         complete=any(indices <= group for group in groups))
                             for field, indices in fields.items()}))
    return rows


def evidence_audit(args, decoder, tokenizer, replacements, prompts, spans, identity, sources):
    reference = json.loads(args.evidence_reference.read_text(encoding='utf-8'))
    if (reference['slots_per_kv_head'] != args.slots or reference['individual_sha256'] != identity
            or reference['sources'] != sources):
        raise ValueError('Evidence interventions require a matched retention run')
    sizes = ('extended',) if args.read_multiplier != 1. else ('long', 'extended')
    targets = [(row, fields) for row, fields in zip(prompts, spans, strict=True) if row['length'] in sizes]
    replacement = replacements[TARGET_LAYER][1]
    original = replacement.core
    heads, head_dim, features = original.k_map.shape
    selected = EvidenceIntervention(heads, head_dim, block=original.block, features=features,
                                     retention_slots=args.slots, policy='residual').to(decoder.base.device).eval()
    selected.read_multiplier = args.read_multiplier
    selected.load_state_dict(original.state_dict(), strict=True)
    replacement.core = selected
    results = {}
    try:
        with using(replacements, 'recurrent'):
            for label in ('evidence', 'sham'):
                found = []
                for row, fields in targets:
                    positions = sorted(fields['new'] | fields['keep'])
                    if label == 'sham':
                        start = max(set.union(*fields.values())) + 64
                        positions = list(range(start, start + len(positions)))
                    if len(positions) > args.slots or max(positions) >= row['input_tokens'] - 2 * selected.block:
                        raise ValueError('Intervention entries do not fit the budget or are still in the recent window')
                    selected.prioritized = torch.tensor(positions, device=decoder.base.device)
                    with observe_prefill(selected, 1) as capture:
                        outcome = run_dialogue(decoder, tokenizer, [row], [])['recall'][0]
                    observed = coverage([row], [fields], capture)[0]
                    if not observed['slots'] or not all(set(positions) <= set(group) for group in observed['slots']):
                        raise RuntimeError('A controlled insertion did not survive to answer generation')
                    found.append(dict(dialogue=outcome, coverage=observed, prioritized=positions))
                scores = {size: sum(item['dialogue']['passed'] for item in found
                                    if item['dialogue']['length'] == size) for size in sizes}
                results[label] = dict(scores=scores, rows=found)
                store_json(args.output / f'{label}.json', results[label])
                print(label, scores, flush=True)
    finally:
        replacement.core = original
    return results


@torch.inference_mode()
def main(args):
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    protected = {path: sha(path) for path in (args.checkpoint, SOURCE, *(
        directory / name for directory in args.candidates for name in ('attention.pt', 'comparison.json')))}
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    prompts = recall_prompts(tokenizer, source, extended=True)
    spans = [annotations(tokenizer, row) for row in prompts]
    base = load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, rotary, 2048)
    base.lm_head.to(base.device)
    identity = frozen_digest(runtime.organism, runtime.state)
    memory = records(runtime.organism, HELD)
    replacements = load_recurrent_replacements(decoder, args.candidates, protected[args.checkpoint])
    if set(replacements) != set(LAYERS):
        raise ValueError('Expected recurrent layers 19, 22 and 24')
    replacement = replacements[TARGET_LAYER][1]
    original = replacement.core
    if type(original) is not GatedDeltaRecurrentAttention:
        raise ValueError('The retained history experiment requires the gated block delta candidate')
    sources = {str(path): digest for path, digest in protected.items()}
    if args.evidence_reference:
        protected[args.evidence_reference] = sha(args.evidence_reference)
        results = evidence_audit(args, decoder, tokenizer, replacements, prompts, spans, identity, sources)
        if (frozen_digest(runtime.organism, runtime.state) != identity
                or any(sha(path) != digest for path, digest in protected.items())):
            raise RuntimeError('A source or the individual changed')
        store_json(args.output / 'result.json', dict(results=results, slots_per_kv_head=args.slots,
                   individual_sha256=identity, sources=sources, reference_sha256=protected[args.evidence_reference],
                   read_multiplier=args.read_multiplier, seconds=perf_counter() - started))
        return
    heads, head_dim, features = original.k_map.shape
    expected = original.state_dict()
    results = {}
    for policy in ('residual', 'recency', 'none'):
        arm_started = perf_counter()
        selected = original
        if policy != 'none':
            selected = SelectiveGatedDeltaAttention(heads, head_dim, block=original.block, features=features,
                                                    retention_slots=args.slots, policy=policy).to(base.device).eval()
            selected.load_state_dict(expected, strict=True)
        replacement.core = selected
        try:
            with using(replacements, 'recurrent'):
                with observe_prefill(selected, len(prompts)) as observed:
                    dialogue = run_dialogue(decoder, tokenizer, prompts, source['open'])
                persistent = evaluate(decoder, tokenizer, runtime.organism, memory)
        finally:
            replacement.core = original
        if frozen_digest(runtime.organism, runtime.state) != identity:
            raise RuntimeError('The individual changed in an inference-only comparison')
        rows = coverage(prompts, spans, observed)
        result = dict(dialogue=dialogue, memory=persistent, coverage=rows, seconds=perf_counter() - arm_started)
        results[policy] = result
        store_json(args.output / f'{policy}.json', result)
        print(policy, dialogue['recall_passed'], dialogue['open_turns_passed'], persistent['summary'], flush=True)
    if any(sha(path) != digest for path, digest in protected.items()):
        raise RuntimeError('An experiment input changed')
    store_json(args.output / 'result.json', dict(results=results, slots_per_kv_head=args.slots,
               individual_sha256=identity, sources=sources,
               seconds=perf_counter() - started))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('candidates', type=Path, nargs=3)
    cli.add_argument('--slots', type=int, default=32)
    cli.add_argument('--evidence-reference', type=Path)
    cli.add_argument('--read-multiplier', type=float, default=1.)
    args = cli.parse_args()
    if not 1 <= args.slots <= 256:
        cli.error('Expected a budget of 1–256 slots per KV head')
    if not 1. <= args.read_multiplier <= 64. or (args.read_multiplier != 1. and not args.evidence_reference):
        cli.error('Read probes require a matched evidence reference and a multiplier from 1 to 64')
    main(args)
