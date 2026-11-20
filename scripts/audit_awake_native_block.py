"""Assemble pretrained components into one native SPORA decoder block."""

import argparse
import random
import time
from pathlib import Path

from audit_awake_feedforward_memory import HISTORY, history_cases
from audit_awake_multiturn_layers import CEVAL_SUBJECTS, ceval_questions, score_memory
from deepseek_output_aligned_memory import MODEL, ROOT, SOURCE, read, sha, torch, write
from spora.language_training import atomic_save
from spora.pretrained_attention import load_absolute_attention, load_layer_norms
from spora.structure_transfer import SPORALayerReplacement, load_native_block
from train_awake_absolute_attention import replay_requests
from train_awake_dialogue_recovery import evaluate
from train_awake_feedforward_replay import load_student
from train_awake_partial_layer import TRAIN_FILES, corpus
from train_awake_replacement_replay import load_layer_pair


INDEX = 18
ATTENTION = ROOT / 'runs/awake-absolute-attention-memory-v1/attention.pt'
NORMS = ROOT / 'runs/awake-block-components-v1/components.pt'
FEEDFORWARD = ROOT / 'runs/awake-gelu-projection-v1/projected_gelu.pt'


def assemble(base, attention, norms, feedforward, index=INDEX):
    native = SPORALayerReplacement(base.config.hidden_size, base.config.num_attention_heads,
                                   attention.max_position, index).to(attention.qkv.weight.device)
    native.position.load_state_dict(attention.position.state_dict())
    for target, source in ((native.block.norm1, norms[0]), (native.block.norm2, norms[1]),
                           (native.block.qkv, attention.qkv), (native.block.ff, feedforward.network)):
        target.load_state_dict(source.state_dict())
    with torch.no_grad():
        native.block.out.weight.copy_(attention.out.weight)
        native.block.out.bias.zero_()
    return native.requires_grad_(False).eval()


@torch.no_grad()
def capture(decoder, layer, requests):
    rows = []

    def record(_module, args, kwargs, output):
        positions = kwargs['position_ids']
        mask = kwargs.get('attention_mask')
        rows.append((args[0].detach().cpu(), positions.cpu(), None if mask is None else mask.cpu(),
                     output.detach().float().cpu()))

    hook = layer.register_forward_hook(record, with_kwargs=True)
    try:
        for ids, kwargs in requests:
            decoder(input_ids=torch.tensor([ids], device='cuda'), logits_to_keep=1, **dict(kwargs, use_cache=False))
    finally:
        hook.remove()
    return rows


def local_prediction(native, row):
    inputs, positions, mask, expected = row
    actual = native(inputs.cuda(), position_ids=positions.cuda(), attention_mask=None if mask is None else mask.cuda())
    return actual.float(), expected.cuda()


@torch.no_grad()
def local_error(native, rows):
    residual, energy = 0., 0.
    for row in rows:
        actual, expected = local_prediction(native, row)
        inputs = row[0].to(expected)
        residual += (actual - expected).square().sum().item()
        energy += (expected - inputs).square().sum().item()
    return residual / energy


def fit(native, ordinary, memory, steps):
    components = (native.position, native.block.norm1, native.block.norm2)
    parameters = [parameter.requires_grad_(True) for module in components for parameter in module.parameters()]
    parameters.append(native.block.out.bias.requires_grad_(True))
    optimizer = torch.optim.AdamW(parameters, lr=2e-4)
    rng = random.Random(44701)
    for step in range(steps):
        row = rng.choice(memory if step % 3 == 0 else ordinary)
        actual, expected = local_prediction(native, row)
        inputs = row[0].to(expected)
        loss = (actual - expected).square().mean() / (expected - inputs).square().mean().clamp_min(1e-8)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % 16 == 0:
            print('Block alignment', step + 1, 'relative error', round(loss.item(), 5), flush=True)
    native.requires_grad_(False)


def evaluate_with_value_switches(decoder, tokenizer, memory, alternate, questions, limit):
    result = evaluate(decoder, tokenizer, memory, questions, position_limit=limit)
    other = score_memory(decoder, tokenizer, alternate)
    result['summary']['memory_switches'] = sum(
        a['correct'] and b['correct'] and a['expected'] not in b['answer'] and b['expected'] not in a['answer']
        for a, b in zip(result['memory'], other, strict=True)
    )
    result['alternate_memory'] = other
    return result


def main(output, ceval, steps):
    paths = (SOURCE, MODEL / 'model.safetensors', MODEL / 'tokenizer.json', MODEL / 'tokenizer_config.json',
             ATTENTION, NORMS, FEEDFORWARD, *TRAIN_FILES, *(ceval / f'{name}.parquet' for name in CEVAL_SUBJECTS))
    protected = {path: sha(path) for path in paths}
    started = time.perf_counter()
    decoder, tokenizer, _, _, organism, _ = load_layer_pair()
    layers = decoder.base.model.layers
    original = layers[INDEX]
    original_attention, original_mlp = original.self_attn, original.mlp
    original_norms = original.input_layernorm, original.post_attention_layernorm
    attention = load_absolute_attention(ATTENTION, original_attention, protected[SOURCE],
                                        read(ATTENTION.parent / 'result.json')['candidate_sha256'])
    norms = load_layer_norms(NORMS, original, protected[SOURCE], read(NORMS.parent / 'result.json')['candidate_sha256'])
    feedforward = load_student(FEEDFORWARD, INDEX, protected[SOURCE], protected[FEEDFORWARD], original_mlp)
    feedforward.requires_grad_(False).eval()
    native = assemble(decoder.base, attention, norms, feedforward)
    _, memory = history_cases(organism)
    changed = [(key, name, (earlier + 2) % 8, (current + 2) % 8) for key, name, earlier, current in HISTORY]
    _, alternate = history_cases(organism, changed)
    questions = list(ceval_questions(ceval, tokenizer))
    results = {}

    def score(name):
        result = evaluate_with_value_switches(decoder, tokenizer, memory, alternate, questions, native.max_length)
        results[name] = result
        print(name, result['summary'], flush=True)

    score('baseline')
    requests = [(tokenizer.encode(text, add_special_tokens=False)[:192], {}) for text in corpus()]
    samples = capture(decoder, original, requests)
    held, train = samples[:8], samples[8:]
    replay = capture(decoder, original, replay_requests(organism, tokenizer))
    original.input_layernorm, original.post_attention_layernorm = norms
    original.self_attn, original.mlp = attention, feedforward
    score('assembled_components')
    original.input_layernorm, original.post_attention_layernorm = original_norms
    original.self_attn, original.mlp = original_attention, original_mlp
    layers[INDEX] = native
    before = local_error(native, held)
    score('native_initial')
    fit(native, train, replay, steps)
    after = local_error(native, held)
    print('Held block residual error', before, after, flush=True)
    score('native_calibrated')

    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError('A protected source changed during the experiment')
    candidate = output / f'layer-{INDEX}.pt'
    metadata = dict(format='spora-qwen-layer-replacement-candidate-v1', layer=INDEX, width=native.width,
                    heads=native.block.heads, max_length=native.max_length,
                    base_weight_sha256=protected[MODEL / 'model.safetensors'], source_sha256=protected[SOURCE])
    atomic_save(dict(metadata=metadata, state={key: value.cpu() for key, value in native.state_dict().items()}), candidate)
    candidate_digest = sha(candidate)
    layers[INDEX] = load_native_block(candidate, decoder.base, INDEX, checkpoint_digest=protected[SOURCE],
                                     base_digest=protected[MODEL / 'model.safetensors'],
                                     candidate_digest=candidate_digest)
    score('native_reloaded')
    if results['native_reloaded'] != results['native_calibrated']:
        raise ValueError('The reloaded decoder block changed evaluation results')
    write(output / 'result.json', dict(results=results, candidate_sha256=candidate_digest, steps=steps,
                                     held_relative_error=dict(initial=before, calibrated=after),
                                     train_sequences=len(train), memory_sequences=len(replay), held_sequences=len(held),
                                     sources={str(path): digest for path, digest in protected.items()},
                                     seconds=time.perf_counter()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ceval', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=64)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('Expected at least one fitting step')
    args.output.mkdir()
    main(args.output, args.ceval, args.steps)
