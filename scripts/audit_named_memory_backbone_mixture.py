"""Measure coherent backbone interpolation with SPORA memory held fixed."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

from scripts.audit_named_memory_qwen_structure_control import (
    CHECKPOINT_SHA, QWEN_SHA, attach_control, dialogue, load_calibration, memory_inputs,
)
from scripts.audit_named_memory_weight_transfer import open_weights, read_vocab_rows, weight_reader
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha, torch
from scripts.deepseek_spora_full_stream_loader import load_qwen_streamed
from scripts.train_qwen_counterfactual_reply_20261122 import HELD, evaluate, records
from spora.integrated_checkpoint import _compatible_blocks, _digest
from spora.language_training import frozen_digest
from transformers import AutoTokenizer


ORDINARY_VOCAB = 151643


def layer_parameters(decoder):
    targets = {'model.norm.weight': decoder.base.model.norm.weight}
    for index, layer in _compatible_blocks(decoder):
        attention = layer.self_attn
        gate, up = layer.mlp.projection.weight.chunk(2, dim=0)
        parts = {'input_layernorm.weight': layer.input_layernorm.weight,
                 'post_attention_layernorm.weight': layer.post_attention_layernorm.weight,
                 'self_attn.o_proj.weight': attention.out.weight,
                 'mlp.gate_proj.weight': gate, 'mlp.up_proj.weight': up,
                 'mlp.down_proj.weight': layer.mlp.down.weight}
        for suffix, fused in (('weight', attention.qkv.weight), ('bias', attention.qkv.bias)):
            for name, part in zip('qkv', fused.split(attention.sizes, dim=0), strict=True):
                parts[f'self_attn.{name}_proj.{suffix}'] = part
        targets.update({f'model.layers.{index}.{name}': tensor for name, tensor in parts.items()})
    return targets


@torch.no_grad()
def apply_mixture(decoder, qwen, deepseek, fraction):
    targets = layer_parameters(decoder)
    with ExitStack() as stack:
        read_qwen = weight_reader(stack, qwen, targets)
        read_deepseek = weight_reader(stack, deepseek, targets)
        for name, parameter in targets.items():
            parameter.copy_(read_qwen(name, parameter.device))
            if fraction:
                parameter.lerp_(read_deepseek(name, parameter.device), fraction)
        qwen_weights = open_weights(stack, qwen)
        deepseek_weights = open_weights(stack, deepseek)
        embedding = decoder.base.get_input_embeddings().weight
        output = decoder.base.get_output_embeddings().weight
        if embedding is output:
            raise ValueError('The mixture requires separate input and output weight matrices')
        for parameter, key in ((embedding, 'model.embed_tokens.weight'), (output, 'lm_head.weight')):
            for start in range(0, ORDINARY_VOCAB, 4096):
                stop = min(start + 4096, ORDINARY_VOCAB)
                part = parameter[start:stop]
                part.copy_(read_vocab_rows(qwen_weights, 'model.embed_tokens.weight', start, stop, parameter))
                if fraction:
                    part.lerp_(read_vocab_rows(deepseek_weights, key, start, stop, parameter), fraction)
    torch.cuda.empty_cache()


def main(checkpoint, donor, calibration, output, fractions):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    qwen = donor / 'model.safetensors'
    deepseek = MODEL / 'model.safetensors'
    reference = json.loads(calibration.read_text(encoding='utf-8'))
    files = (checkpoint, *(donor / name for name in ('model.safetensors', 'config.json', 'tokenizer.json',
                                                    'tokenizer_config.json', 'generation_config.json')))
    sources = {str(path): sha(path) for path in files}
    if (sources[str(checkpoint)] != CHECKPOINT_SHA or sources[str(qwen)] != QWEN_SHA
            or sources != reference['sources'] or sha(deepseek) != WEIGHT_SHA):
        raise ValueError('Backbones and common memory checkpoint must match the calibrated experiment')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    metadata, tensors = payload['metadata'], payload['tensors']
    if _digest(metadata, tensors) != payload['integrity_sha256']:
        raise ValueError('Checkpoint integrity mismatch')
    tokenizer = AutoTokenizer.from_pretrained(donor, local_files_only=True, trust_remote_code=False)
    original_tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    if {key: value for key, value in tokenizer.get_vocab().items() if value < ORDINARY_VOCAB} != {
            key: value for key, value in original_tokenizer.get_vocab().items() if value < ORDINARY_VOCAB}:
        raise ValueError('Ordinary token coordinates differ')
    base = load_qwen_streamed(donor, QWEN_SHA)
    decoder, organism, state = attach_control(base, metadata, tensors)
    adapter_path = calibration.with_suffix('.reply.pt')
    decoder.output_adapter, adapter_sha = load_calibration(adapter_path, reference, sources, decoder, organism)
    base.lm_head.weight = torch.nn.Parameter(base.lm_head.weight.detach().clone(), requires_grad=False)
    base.config.tie_word_embeddings = False
    memory_rows = records(organism, HELD)
    individual_sha = frozen_digest(organism, state)
    if individual_sha != reference['individual_sha256']:
        raise ValueError('Unexpected individual state')
    eos = base.generation_config.eos_token_id
    output.mkdir(parents=True, exist_ok=False)
    for fraction in (0., *fractions, 0.):
        # Sample each coefficient from the source files so rounding never accumulates.
        label = f'deepseek-{fraction:g}'
        if not fraction and (output / f'{label}.json').exists():
            label = 'restored'
        print('Evaluating', label, flush=True)
        apply_mixture(decoder, qwen, deepseek, fraction)
        chat = dialogue(decoder, tokenizer, eos)
        torch.cuda.empty_cache()
        memory = evaluate(decoder, tokenizer, organism, memory_rows, encode=memory_inputs, eos=eos)
        torch.cuda.empty_cache()
        if not fraction and (chat != reference['migrated'] or memory != reference['recalibration']['memory']):
            raise ValueError(f'Qwen control differs at {label}')
        if frozen_digest(organism, state) != individual_sha:
            raise ValueError('The neural individual changed during evaluation')
        result = {'deepseek_fraction': fraction, 'sources': {**sources, str(deepseek): WEIGHT_SHA},
                  'individual_sha256': individual_sha, 'adapter_sha256': adapter_sha,
                  'dialogue': chat, 'memory': memory}
        with (output / f'{label}.json').open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        print(label, [(item['cases_passed'], item['turns_passed']) for item in chat.values()],
              memory['summary'], flush=True)
    if any(sha(Path(path)) != digest for path, digest in sources.items()) or sha(deepseek) != WEIGHT_SHA or sha(adapter_path) != adapter_sha:
        raise ValueError('An experiment input changed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'donor', 'calibration', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--fractions', nargs='+', type=float, default=[.15, .35, .65])
    args = parser.parse_args()
    if not all(0 < value <= 1 for value in args.fractions) or len(set(args.fractions)) != len(args.fractions):
        parser.error('Specify distinct DeepSeek fractions between 0 (exclusive) and 1')
    main(**vars(args))
