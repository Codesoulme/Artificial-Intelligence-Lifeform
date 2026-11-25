"""Test whether migrated layers carry a query that can be separated from its answer."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_spora_recurrent_attention import last_logits
from scripts.benchmark_spora_recurrent_dialogue import (
    SOURCE, MODEL, AutoTokenizer, encode, exchange, extend_positions, frozen_digest,
    load_full_streamed, load_integrated_runtime, sha, torch, using,
)
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.research_recurrent_decode import load_recurrent_replacements


FIELDS = (
    ('展板领取', '房间', '人员'),
    ('校刊寄送', '日期', '地址'),
    ('物品交接', '地点', '时间'),
)
PROBE_LAYERS = (18, 19, 22, 25)
POSITIONS = 2048


def questions(tokenizer, fillers):
    """Cross two queried fields with both possible assignments of the answer labels."""
    suffix = tokenizer.encode('编号：', add_special_tokens=False)
    for topic, left, right in FIELDS:
        prompts = {}
        for mapping in (0, 1):
            history = []
            exchange(history, f'{topic}的字段编号已经确认。{left}的编号是{1 + mapping}，'
                              f'{right}的编号是{2 - mapping}。', '好，我已登记。')
            for filler in fillers:
                exchange(history, filler, '收到。')
            for field, name in enumerate((left, right)):
                question = f'回到{topic}的登记：{name}对应哪个编号？请只填编号。'
                prompts[mapping, field] = encode(
                    tokenizer, [*history, {'role': 'user', 'content': question}],
                ) + suffix
        if len({len(ids) for ids in prompts.values()}) != 1:
            raise ValueError(f'Donor and recipient positions differ for {topic}')
        yield topic, prompts


def snapshot(module_output):
    return module_output if isinstance(module_output, torch.Tensor) else module_output[0]


def capture(decoder, layers, ids):
    positions = {}

    def capture_last(layer):
        def hook(_module, _args, output):
            positions[layer] = snapshot(output)[:, -1, :].clone()
        return hook

    with ExitStack() as hooks:
        for layer in PROBE_LAYERS:
            handle = layers[layer].register_forward_hook(capture_last(layer))
            hooks.callback(handle.remove)
        log_probs = last_logits(decoder, ids)
    if tuple(positions) != PROBE_LAYERS:
        raise RuntimeError('Decoder did not run all probe layers in order')
    return log_probs, positions


def patched(decoder, layer, ids, replacement):
    def hook(_module, _args, output):
        hidden = snapshot(output).clone()
        hidden[:, -1, :] = replacement
        return hidden if isinstance(output, torch.Tensor) else (hidden, *output[1:])

    handle = layer.register_forward_hook(hook)
    try:
        return last_logits(decoder, ids)
    finally:
        handle.remove()


def readout(scores, label_ids, expected):
    choices = scores[label_ids]
    return {'log_odds': float(choices[expected] - choices[1 - expected]),
            'selected': 1 + int(choices.argmax()), 'top_token': int(scores.argmax())}


def evaluate(decoder, tokenizer, fillers, label_ids):
    layers = decoder.base.model.layers
    results = []
    for topic, prompts in questions(tokenizer, fillers):
        captures = {key: capture(decoder, layers, ids) for key, ids in prompts.items()}
        self_key = (0, 0)
        original_scores, vectors = captures[self_key]
        for layer in PROBE_LAYERS:
            repeated = patched(decoder, layers[layer], prompts[self_key], vectors[layer])
            if not torch.equal(repeated, original_scores):
                raise RuntimeError(f'Self-patch changed the logits at layer {layer}')
        for (mapping, field), ids in prompts.items():
            current = mapping ^ field
            expected = 1 - current
            row = {'topic': topic, 'mapping': mapping, 'field': field, 'tokens': len(ids),
                   'baseline': readout(captures[mapping, field][0], label_ids, current),
                   'patches': {}}
            for layer in PROBE_LAYERS:
                row['patches'][layer] = {}
                for label, donor_mapping in (('same_assignment', mapping), ('reversed_assignment', 1 - mapping)):
                    donor = captures[donor_mapping, 1 - field][1][layer]
                    scores = patched(decoder, layers[layer], ids, donor)
                    row['patches'][layer][label] = readout(scores, label_ids, expected)
            results.append(row)
        print(topic, len(prompts[self_key]), flush=True)
    return results


def summarize(rows, label_ids):
    correct = [row for row in rows if row['baseline']['log_odds'] > 0]
    grounded_topics = [
        topic for topic, _, _ in FIELDS if sum(row['topic'] == topic for row in correct) == 4
    ]
    grounded = [row for row in correct if row['topic'] in grounded_topics]
    return {
        'inputs': len(rows), 'baseline_correct': len(correct),
        'baseline_top_is_label': sum(row['baseline']['top_token'] in label_ids for row in rows),
        'fully_grounded_topics': grounded_topics,
        'grounded_inputs': len(grounded),
        'patch_selects_new_field': {
            layer: {
                label: sum(row['patches'][layer][label]['log_odds'] > 0 for row in grounded)
                for label in ('same_assignment', 'reversed_assignment')
            }
            for layer in PROBE_LAYERS
        },
    }


@torch.inference_mode()
def main(args):
    started = perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    protected = {path: sha(path) for path in (
        args.checkpoint, SOURCE, Path(__file__), MODEL / 'tokenizer.json',
        MODEL / 'tokenizer_config.json',
        *(folder / name for folder in args.candidates for name in ('attention.pt', 'comparison.json')),
    )}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    labels = [tokenizer.encode(label, add_special_tokens=False) for label in ('1', '2')]
    if any(len(label) != 1 for label in labels) or labels[0] == labels[1]:
        raise ValueError('Expected two distinct single-token labels')
    label_ids = [label[0] for label in labels]
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    fillers = source['filler'] if args.long else []
    list(questions(tokenizer, fillers))
    base = load_full_streamed(stage_output_projection=True)
    native_rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    extend_positions(decoder, native_rotary, POSITIONS)
    torch.cuda.empty_cache()
    base.lm_head.to(base.device)
    individual = frozen_digest(runtime.organism, runtime.state)
    candidates = load_recurrent_replacements(decoder, args.candidates, protected[args.checkpoint])
    if tuple(candidates) != (19, 22):
        raise ValueError('This probe compares the established two-layer layout with its parent')
    conditions = {}
    for mode in ('parent', 'recurrent'):
        with using(candidates, mode):
            rows = evaluate(decoder, tokenizer, fillers, label_ids)
        summary = summarize(rows, label_ids)
        conditions[mode] = {'summary': summary, 'rows': rows}
        print(mode, summary, flush=True)
    if (individual != frozen_digest(runtime.organism, runtime.state)
            or any(sha(path) != digest for path, digest in protected.items())):
        raise RuntimeError('A source artifact or individual changed')
    store_json(args.output, {'conditions': conditions, 'long': args.long, 'layers': list(PROBE_LAYERS),
                             'label_ids': label_ids,
                             'individual_sha256': individual, 'seconds': perf_counter() - started,
                             'sources': {str(path): digest for path, digest in protected.items()}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('candidates', nargs=2, type=Path)
    parser.add_argument('--long', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    main(args)
