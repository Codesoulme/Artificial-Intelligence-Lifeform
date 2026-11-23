"""Screen late-layer weight interpolation on one local SPORA individual."""

import argparse
import json
import struct
from contextlib import ExitStack
from pathlib import Path

from scripts.audit_local_named_chat import CASES, FRESH_CASES, run
from scripts.audit_qwen_packed_reply_20261122 import HELD, evaluate, records
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha, torch
from scripts.deepseek_spora_full_stream_loader import load_full_streamed, stream_fp16_tensor
from spora.integrated_checkpoint import load_integrated_runtime
from spora.language_training import frozen_digest
from spora.pretrained_attention import SPORAPretrainedBlock
from transformers import AutoTokenizer


DONOR_SHA = 'dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee'
ANCESTOR_SHA = 'a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796'
VARIANTS = {
    'original': (28, 0., False),
    'ffn-last-six-half': (22, .5, False),
    'ffn-last-six-full': (22, 1., False),
    'blocks-last-six-half': (22, .5, True),
    'blocks-last-six-full': (22, 1., True),
    'blocks-last-six-low': (22, .15, True),
    'plain-blocks-half': (0, .5, True),
    'plain-blocks-full': (0, 1., True),
    'plain-blocks-and-vocab-full': (0, 1., True),
}
DEFAULT_VARIANTS = ('original', 'ffn-last-six-half', 'ffn-last-six-full',
                    'blocks-last-six-half', 'blocks-last-six-full')


def parameters(decoder, start, full_blocks):
    targets = {}
    for index in range(start, decoder.base.config.num_hidden_layers):
        block = decoder.base.model.layers[index]
        plain = type(block) is SPORAPretrainedBlock and index not in decoder.layer_indices and not block.mlp.read_memory
        if not plain and start == 0:
            continue
        if not plain:
            raise ValueError(f'Expected a plain migrated late layer at {index}')
        gate, up = block.mlp.projection.weight.chunk(2, dim=0)
        parts = {'mlp.gate_proj.weight': gate, 'mlp.up_proj.weight': up,
                 'mlp.down_proj.weight': block.mlp.down.weight}
        if full_blocks:
            attention = block.self_attn
            for suffix, fused in (('weight', attention.qkv.weight), ('bias', attention.qkv.bias)):
                for name, part in zip('qkv', fused.split(attention.sizes, dim=0), strict=True):
                    parts[f'self_attn.{name}_proj.{suffix}'] = part
            parts.update({'self_attn.o_proj.weight': attention.out.weight,
                          'input_layernorm.weight': block.input_layernorm.weight,
                          'post_attention_layernorm.weight': block.post_attention_layernorm.weight})
        targets.update({f'model.layers.{index}.{name}': part for name, part in parts.items()})
    if not targets:
        raise ValueError('No compatible layers selected')
    return targets


def weight_reader(stack, path, targets):
    stream, header, data_start, size = open_weights(stack, path)
    for name, destination in targets.items():
        if header[name]['shape'] != list(destination.shape):
            raise ValueError(f'Weight shape mismatch for {path}: {name}')

    def read(name, device):
        return stream_fp16_tensor(stream, header[name], data_start, size, device)

    return read


def open_weights(stack, path):
    stream = stack.enter_context(path.open('rb'))
    size = path.stat().st_size
    length, = struct.unpack('<Q', stream.read(8))
    if length > min(size - 8, 100_000_000):
        raise ValueError('Invalid source weight header')
    return stream, json.loads(stream.read(length)), 8 + length, size


def read_vocab_rows(weights, key, start, stop, destination):
    stream, header, data_start, size = weights
    item = header[key]
    rows, width = item['shape']
    if rows < stop or width != destination.shape[1]:
        raise ValueError(f'Incompatible vocabulary matrix: {key}')
    offset = item['data_offsets'][0]
    selected = {'shape': [stop - start, width], 'dtype': item['dtype'],
                'data_offsets': [offset + 2 * width * start, offset + 2 * width * stop]}
    return stream_fp16_tensor(stream, selected, data_start, size, destination.device)


@torch.no_grad()
def apply_vocab(decoder, source, ancestor=None):
    embedding = decoder.base.get_input_embeddings().weight
    output = decoder.base.get_output_embeddings().weight
    embedding_key = 'model.embed_tokens.weight'
    with ExitStack() as stack:
        donor = open_weights(stack, source)
        original = open_weights(stack, ancestor) if ancestor is not None else None
        for target, key in ((embedding, embedding_key), (output, 'lm_head.weight')):
            for start in range(0, 151643, 4096):
                stop = min(start + 4096, 151643)
                if original is None:
                    target[start:stop].copy_(read_vocab_rows(donor, key, start, stop, target))
                else:
                    delta = read_vocab_rows(donor, embedding_key, start, stop, target)
                    delta.sub_(read_vocab_rows(original, embedding_key, start, stop, target))
                    target[start:stop].add_(delta)
    torch.cuda.empty_cache()


@torch.no_grad()
def apply_weights(decoder, source, start, fraction, full_blocks, ancestor=None):
    targets = parameters(decoder, start, full_blocks)
    with ExitStack() as stack:
        read_source = weight_reader(stack, source, targets)
        read_ancestor = weight_reader(stack, ancestor, targets) if ancestor is not None else None
        for name, destination in targets.items():
            source_weight = read_source(name, destination.device)
            if read_ancestor is not None:
                source_weight.sub_(read_ancestor(name, destination.device))
                destination.add_(source_weight, alpha=fraction)
            elif fraction == 1.:
                destination.copy_(source_weight)
            else:
                destination.lerp_(source_weight, fraction)
            del source_weight
    torch.cuda.empty_cache()


@torch.no_grad()
def install_chat_markers(stack, decoder, donor_file, tokenizer, chat_tokenizer):
    original_vocab = tokenizer.get_vocab()
    chat_vocab = chat_tokenizer.get_vocab()
    if {text: index for text, index in original_vocab.items() if index < 151643} != {
            text: index for text, index in chat_vocab.items() if index < 151643}:
        raise ValueError('The ordinary token coordinates are incompatible')
    markers = [chat_vocab[name] for name in ('<|endoftext|>', '<|im_start|>', '<|im_end|>')]
    if markers != list(range(151643, 151646)):
        raise ValueError('Unexpected Qwen chat markers')
    limit = decoder.base.config.vocab_size
    slots = list(range(limit - len(markers), limit))
    if max(original_vocab.values()) >= slots[0] or max(chat_vocab.values()) >= slots[0]:
        raise ValueError('Independent chat markers require unused model rows')
    embedding = decoder.base.get_input_embeddings().weight
    output = decoder.base.get_output_embeddings().weight
    weights = open_weights(stack, donor_file)
    header = weights[1]
    item = header['model.embed_tokens.weight']
    vocab, width = item['shape']
    if vocab != limit or width != embedding.shape[1] or 'lm_head.weight' in header:
        raise ValueError('Expected Qwen embeddings tied to its output rows')
    rows = read_vocab_rows(weights, 'model.embed_tokens.weight', markers[0], markers[-1] + 1, embedding)
    active_rows = slice(slots[0], slots[-1] + 1)
    for target in (embedding, output):
        original = target[active_rows].clone()
        stack.callback(target[active_rows].copy_, original)
        target[active_rows].copy_(rows)
    blocked = [[index] for index in sorted(tokenizer.all_special_ids)]

    @torch.inference_mode()
    def reply(history, question):
        messages = [{'role': 'system', 'content': ''}, *history, {'role': 'user', 'content': question}]
        prompt = chat_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = chat_tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(embedding.device)
        ids = inputs['input_ids']
        for marker, slot in zip(markers, slots, strict=True):
            ids.masked_fill_(ids == marker, slot)
        generated = decoder.generate(**inputs, max_new_tokens=96, use_cache=True, do_sample=False,
                                     temperature=None, top_p=None, top_k=None, repetition_penalty=1.,
                                     pad_token_id=slots[0], eos_token_id=[slots[0], slots[-1]],
                                     bad_words_ids=blocked)
        tail = [index for index in generated[0, ids.shape[1]:].tolist() if index < 151643]
        return chat_tokenizer.decode(tail, skip_special_tokens=True).strip()

    return reply


def main(checkpoint, donor, output, variants, ancestor, chat_markers):
    if output.exists():
        raise FileExistsError(output)
    base_file = MODEL / 'model.safetensors'
    donor_file = donor / 'model.safetensors'
    sources = {checkpoint: sha(checkpoint), donor_file: sha(donor_file), base_file: WEIGHT_SHA}
    if sources[donor_file] != DONOR_SHA:
        raise ValueError('Expected the pinned Qwen2.5-1.5B-Instruct donor')
    if ancestor is not None:
        sources[ancestor] = sha(ancestor)
        if sources[ancestor] != ANCESTOR_SHA:
            raise ValueError('Expected the pinned Qwen2.5-1.5B ancestor')
    torch.set_num_threads(2)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    decoder, organism = runtime.decoder, runtime.organism
    before = frozen_digest(organism, runtime.state)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    chat_tokenizer = None
    if chat_markers:
        for name in ('tokenizer.json', 'tokenizer_config.json'):
            sources[donor / name] = sha(donor / name)
        chat_tokenizer = AutoTokenizer.from_pretrained(donor, local_files_only=True, trust_remote_code=False)
    memory_rows = records(organism, HELD)
    output.mkdir(parents=True, exist_ok=False)
    original = None
    trial_order = [*variants, 'restored'] if 'original' in variants and len(variants) > 1 else variants
    for variant in trial_order:
        start, fraction, full_blocks = VARIANTS['original' if variant == 'restored' else variant]
        print('Running', variant, flush=True)
        layers = []
        vocab_delta = variant == 'plain-blocks-and-vocab-full'
        if fraction:
            layers = sorted({int(name.split('.')[2]) for name in parameters(decoder, start, full_blocks)})
            apply_weights(decoder, donor_file, start, fraction, full_blocks, ancestor)
        if vocab_delta:
            apply_vocab(decoder, donor_file, ancestor)
        with ExitStack() as stack, torch.no_grad():
            reply = (install_chat_markers(stack, decoder, donor_file, tokenizer, chat_tokenizer)
                     if chat_tokenizer is not None else None)
            dialogue = run(decoder, tokenizer, cases=CASES, reply_with_history=reply)
            torch.cuda.empty_cache()
            fresh = run(decoder, tokenizer, cases=FRESH_CASES, reply_with_history=reply)
            torch.cuda.empty_cache()
            memory = evaluate(decoder, tokenizer, organism, memory_rows)
        torch.cuda.empty_cache()
        outcomes = (dialogue, fresh, memory)
        if variant == 'original':
            original = outcomes
        elif variant == 'restored' and outcomes != original:
            raise ValueError('Restoring the source weights changed the original outputs')
        if frozen_digest(organism, runtime.state) != before:
            raise ValueError('The individual changed during inference')
        if fraction:
            apply_weights(decoder, base_file, start, 1., full_blocks)
        if vocab_delta:
            apply_vocab(decoder, base_file)
        result = {'variant': variant, 'first_layer': start if fraction else None,
                  'fraction': fraction, 'full_blocks': full_blocks, 'layers': layers,
                  'method': 'task-vector' if ancestor is not None else 'interpolation',
                  'qwen_chat_markers': chat_markers,
                  'vocab_delta': vocab_delta,
                  'sources': {str(path): digest for path, digest in sources.items()},
                  'individual_sha256': before, 'dialogue': dialogue, 'fresh': fresh, 'memory': memory}
        with (output / f'{variant}.json').open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        print(variant, [(part['cases_passed'], part['turns_passed']) for part in (dialogue, fresh)],
              memory['summary'], flush=True)
    if any(sha(path) != digest for path, digest in sources.items()):
        raise ValueError('A protected source changed during the audit')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('checkpoint', 'donor', 'output'):
        parser.add_argument(f'--{argument}', type=Path, required=True)
    parser.add_argument('--ancestor', type=Path, help='Apply the donor minus its base as a task vector')
    parser.add_argument('--chat-markers', action='store_true', help='Use remapped Qwen markers for dialogue only')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=DEFAULT_VARIANTS)
    args = parser.parse_args()
    if len(set(args.variants)) != len(args.variants):
        parser.error('Choose each variant only once')
    if 'plain-blocks-and-vocab-full' in args.variants and (args.ancestor is None or not args.chat_markers):
        parser.error('Vocabulary transfer uses the pinned task vector and chat markers')
    main(**vars(args))
