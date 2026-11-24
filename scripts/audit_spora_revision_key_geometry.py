"""Measure whether old and corrected value spans address the same recurrent history."""

import argparse
import json
from pathlib import Path
from statistics import median

from scripts.benchmark_spora_recurrent_dialogue import (
    MODEL, SOURCE, AutoTokenizer, encode, extend_positions, frozen_digest, load_full_streamed,
    load_integrated_runtime, recall_prompts, sha, torch,
)
from scripts.train_spora_recurrent_long_alignment import store_json
from spora.research_recurrent_decode import load_recurrent_replacements, projections
from torch.nn.functional import cosine_similarity
from transformers.models.qwen2.modeling_qwen2 import rotate_half


def span_tokens(offsets, start, end):
    tokens = [index for index, (left, right) in enumerate(offsets) if right > start and left < end]
    if not tokens or max(offsets[index][1] for index in tokens) < end:
        raise ValueError('The tokenizer did not cover the value span')
    return tokens


def probe_prompt(tokenizer, row):
    messages = [*row['history'], {'role': 'user', 'content': row['question']}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + '</think>\n\n'
    batch = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if batch['input_ids'] != encode(tokenizer, messages):
        raise ValueError('Key probe does not match generation input')
    offsets = batch['offset_mapping']
    spans = {}
    for key, user_turn in (('old', 0), ('keep', 0), ('new', 2)):
        message = row['history'][user_turn]['content']
        value = row['values'][key]
        if text.count(message) != 1 or message.count(value) != 1:
            raise ValueError('Expected a uniquely identifiable value span')
        start = text.index(message) + message.index(value)
        spans[key] = span_tokens(offsets, start, start + len(value))
    return batch['input_ids'], spans


@torch.inference_mode()
def measure(core, spans, raw, rotated):
    result = {}
    for name, keys in (('used', rotated), ('without_rotary', raw)):
        groups = core.k_map.shape[0] // keys.shape[1]
        keys = keys.float().repeat_interleave(groups, dim=1)
        fields = {key: core.phi(keys[:, :, positions], core.k_map).mean(dim=2)
                  for key, positions in spans.items()}
        revision = cosine_similarity(fields['new'], fields['old'], dim=-1)[0]
        other = cosine_similarity(fields['new'], fields['keep'], dim=-1)[0]
        result[name] = {'same_field': revision.tolist(), 'different_field': other.tolist(),
                        'heads_favoring_same': int((revision > other).sum())}
    return result


@torch.inference_mode()
def audit(runtime, tokenizer, layers, prompts):
    current, found = {}, {}
    handles = []

    def capture(module, args, kwargs, index):
        hidden = kwargs.get('hidden_states') if 'hidden_states' in kwargs else args[0]
        cos, sin = kwargs['position_embeddings']
        _, actual, _ = projections(module, hidden, (cos, sin), kwargs.get('position_ids'))
        raw = actual.float() * cos[:, None].float() - rotate_half(actual.float()) * sin[:, None].float()
        found[index] = measure(layers[index][1].core, current['spans'], raw, actual)

    rows = []
    try:
        for index, (block, _) in layers.items():
            handles.append(block.self_attn.register_forward_pre_hook(
                lambda module, args, kwargs, layer=index: capture(module, args, kwargs, layer), with_kwargs=True))
        for prompt in prompts:
            ids, current['spans'] = probe_prompt(tokenizer, prompt)
            found.clear()
            tokens = torch.tensor([ids], device=runtime.decoder.base.device)
            runtime.decoder(input_ids=tokens, use_cache=False, logits_to_keep=1)
            if found.keys() != layers.keys():
                raise ValueError('Some recurrent layer keys were not collected')
            rows.append({'case': prompt['case'], 'variant': prompt['variant'], 'positions': current['spans'],
                         'layers': found.copy()})
    finally:
        for handle in handles:
            handle.remove()
    return rows


def summarize(rows, layers):
    result = {}
    for layer in layers:
        result[layer] = {}
        for name in ('used', 'without_rotary'):
            cases = [row['layers'][layer][name] for row in rows]
            positives = [value for case in cases for value in case['same_field']]
            negatives = [value for case in cases for value in case['different_field']]
            result[layer][name] = {'same_field_median': median(positives), 'different_field_median': median(negatives),
                                   'same_field_higher': sum(case['heads_favoring_same'] for case in cases),
                                   'head_case_count': len(positives)}
    return result


def main(args):
    inputs = {path: sha(path) for path in (args.checkpoint, SOURCE)}
    for directory in args.layers:
        inputs.update({directory / name: sha(directory / name) for name in ('attention.pt', 'comparison.json')})
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    prompts = [row for row in recall_prompts(tokenizer, source, extended=True) if row['length'] == 'extended']
    base = load_full_streamed()
    rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    extend_positions(runtime.decoder, rotary, 2048)
    layers = load_recurrent_replacements(runtime.decoder, args.layers, inputs[args.checkpoint])
    rows = audit(runtime, tokenizer, layers, prompts)
    summary = summarize(rows, layers)
    if (individual != frozen_digest(runtime.organism, runtime.state)
            or any(sha(path) != digest for path, digest in inputs.items())):
        raise ValueError('A key probe input changed')
    store_json(args.output, {'summary': summary, 'rows': rows, 'individual_sha256': individual,
                             'inputs': {str(path): digest for path, digest in inputs.items()}})
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('layers', nargs=2, type=Path)
    options = parser.parse_args()
    if options.output.exists():
        parser.error('Choose a new output path')
    main(options)
