"""Separate the effect of saved attention keys from their associated values."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from scripts import audit_spora_persistent_attention_slots as slots
from spora.research_recurrent_decode import attention_block, load_recurrent_replacements


LAYER = 23
BIASES = (0, 4)


def evaluate(decoder, tokenizer, attention, bank, reference, count):
    rows = []
    suffix = tokenizer.encode('编号：', add_special_tokens=False)
    for index, (topic, left, right, labels) in enumerate(slots.BASE_CASES + slots.FRESH_CASES):
        fresh = index >= len(slots.BASE_CASES)
        versions = [tuple(part[..., -count:, :] for part in version) for version in bank[topic]]
        if versions[0][0].shape != versions[1][0].shape or versions[0][1].shape != versions[1][1].shape:
            raise ValueError('Crossed tensors must have identical shapes')
        for field, name in enumerate((left, right)):
            question = (f'我想核对{topic}留下的{name}编号。能只报数字吗？' if fresh
                        else f'回到{topic}的登记：{name}对应哪个编号？请只填编号。')
            ids = slots.encode(tokenizer, [{'role': 'user', 'content': question}]) + suffix
            prior = reference[topic, name]
            label_ids = prior['label_ids']
            if any(token in ids for token in label_ids):
                raise ValueError('Answer label appeared in the fresh request')
            if [tokenizer.encode(label, add_special_tokens=False) for label in labels] != [[t] for t in label_ids]:
                raise ValueError('Answer tokenization differs from the original experiment')
            conditions = []
            for bias in BIASES:
                for key_mapping, (key, _) in enumerate(versions):
                    previous_mass = None
                    for value_mapping, (_, value) in enumerate(versions):
                        mass = {LAYER: []}
                        scores = slots.read_attention(decoder, ids, ((attention, (key, value)),), bias=bias, mass=mass)
                        expected = value_mapping ^ field
                        readout = slots.query_audit.readout(scores, label_ids, expected)
                        if previous_mass is not None and previous_mass != mass[LAYER]:
                            raise RuntimeError('Changing values modified the attention weights')
                        previous_mass = mass[LAYER]
                        if key_mapping == value_mapping:
                            saved = next(x for x in prior['conditions']
                                         if x['compatible'] and x['bias'] == bias and x['mapping'] == value_mapping)
                            if saved['stored'] != readout or saved['mass'][str(LAYER)] != mass[LAYER]:
                                raise RuntimeError('Matched keys and values did not reproduce the original result')
                        conditions.append({
                            'bias': bias, 'key_mapping': key_mapping, 'value_mapping': value_mapping,
                            'key_target': label_ids[key_mapping ^ field], 'value_target': label_ids[expected],
                            'readout': readout, 'mass': mass[LAYER],
                        })
            rows.append({'topic': topic, 'field': name, 'fresh': fresh, 'label_ids': label_ids,
                         'conditions': conditions})
        print(topic, flush=True)
    return rows


@slots.torch.inference_mode()
def main(args):
    started = perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    slots.torch.set_num_threads(2)
    bank_sha = slots.sha(args.bank)
    reference_sha = slots.sha(args.reference)
    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    bank = slots.torch.load(args.bank, map_location='cpu', weights_only=True)
    if (not reference['sweep'] or not bank['sweep'] or reference['slots_sha256'] != bank_sha
            or reference['sources'] != bank['sources']
            or reference['individual_sha256'] != bank['individual_sha256']):
        raise ValueError('The archive and the prior sweep must describe the same run')
    sources = {Path(path): digest for path, digest in bank['sources'].items()}
    runtime_paths = (
        args.checkpoint, slots.MODEL / 'model.safetensors', slots.MODEL / 'tokenizer.json',
        slots.MODEL / 'tokenizer_config.json', Path(slots.__file__), Path(slots.query_audit.__file__),
        *(folder / name for folder in args.candidates for name in ('attention.pt', 'comparison.json')),
    )
    if set(runtime_paths) != sources.keys() or any(slots.sha(path) != digest for path, digest in sources.items()):
        raise ValueError('The model or source files differ from the written archive')
    sources[Path(__file__)] = slots.sha(Path(__file__))
    tokenizer = slots.AutoTokenizer.from_pretrained(slots.MODEL, local_files_only=True, trust_remote_code=False)
    base = slots.load_full_streamed(stage_output_projection=True)
    rotary = base.model.rotary_emb
    runtime = slots.load_integrated_runtime(args.checkpoint, base, slots.MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    slots.extend_positions(decoder, rotary, slots.query_audit.POSITIONS)
    slots.torch.cuda.empty_cache()
    base.lm_head.to(base.device)
    individual = slots.frozen_digest(runtime.organism, runtime.state)
    if individual != bank['individual_sha256']:
        raise ValueError('The loaded individual differs from the recorded one')
    replacements = load_recurrent_replacements(decoder, args.candidates, sources[args.checkpoint])
    if tuple(replacements) != (19, 22):
        raise ValueError('Expected the established recurrent layout')
    previous_rows = {(x['topic'], x['field']): x for x in reference['rows'] if x['layers'] == [LAYER]}
    if len(previous_rows) != 2 * len(slots.BASE_CASES + slots.FRESH_CASES):
        raise ValueError('Unexpected number of baseline questions')
    attention = attention_block(decoder, LAYER).self_attn
    with slots.using(replacements, 'recurrent'):
        rows = evaluate(decoder, tokenizer, attention, bank['banks'][LAYER],
                        previous_rows, reference['slots_per_layer'])
    if (individual != slots.frozen_digest(runtime.organism, runtime.state)
            or any(slots.sha(path) != digest for path, digest in sources.items())
            or slots.sha(args.bank) != bank_sha or slots.sha(args.reference) != reference_sha):
        raise RuntimeError('An input changed during evaluation')
    slots.store_json(args.output, {
        'layer': LAYER, 'biases': BIASES, 'rows': rows, 'individual_sha256': individual,
        'sources': {str(path): digest for path, digest in sources.items()},
        'bank_sha256': bank_sha, 'reference_sha256': reference_sha, 'seconds': perf_counter() - started,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('bank', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('candidates', nargs=2, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    main(args)
