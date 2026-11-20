"""Distill a native SPORA feed-forward replacement while retaining Qwen attention."""

import argparse
import copy
import hashlib
from pathlib import Path

from audit_awake_language_layers import QUESTIONS, REFERENCES, load_layer_pair
from deepseek_output_aligned_memory import MODEL, ROOT, SOURCE, read, sha, torch, write
from spora.language_training import atomic_save
from train_awake_replacement_replay import answer_logits, divergence, encode_cases, evaluate, switch, teacher_targets


TRAIN_FILES = (ROOT / 'data/chinese-foundations.json', ROOT / 'data/chinese-pilot/source.json')
SEED = 41562


class SPORAFeedForward(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, inputs):
        return self.network(inputs.to(self.network[0].weight)).to(inputs.dtype)


def corpus():
    dialogues = read(TRAIN_FILES[0])['dialogues']
    rows = ['\n'.join('\n'.join(turn) for turn in item['turns']) for item in dialogues]
    rows += read(TRAIN_FILES[1])['train']
    return sorted(set(rows), key=lambda text: hashlib.sha256(text.encode()).digest())


@torch.no_grad()
def collect(decoder, tokenizer, texts, teacher_mlp):
    inputs, outputs = [], []

    def capture(_module, args, output):
        inputs.append(args[0][0].float().cpu())
        outputs.append(output[0].float().cpu())

    handle = teacher_mlp.register_forward_hook(capture)
    try:
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)[:192]
            decoder(input_ids=torch.tensor([ids], device='cuda'), use_cache=False, logits_to_keep=1)
    finally:
        handle.remove()
    return torch.cat(inputs).to('cuda'), torch.cat(outputs).to('cuda')


@torch.no_grad()
def relative_mse(model, features, labels):
    errors = sum((model(x) - y).square().sum().item()
                 for x, y in zip(features.split(128), labels.split(128), strict=True))
    return errors / labels.square().sum().item()


def refine_output(decoder, tokenizer, student, layer, teacher_layers, replacements, texts, steps):
    original = teacher_layers[layer]
    teacher = original.mlp
    cases = []
    for text in texts[:24]:
        tokens = tokenizer.encode(text, add_special_tokens=False)[:80]
        cut = max(1, len(tokens) - 16)
        if len(tokens) > cut:
            cases.append((text, tokens[:cut], tokens[cut:]))
    switch(decoder, teacher_layers)
    original.mlp = teacher
    targets = teacher_targets(decoder, cases)
    switch(decoder, {**replacements, layer: original})
    original.mlp = student
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-5, weight_decay=.01)
    for index in range(steps):
        row = index % len(cases)
        _, prompt, tails = cases[row]
        logits = answer_logits(decoder, prompt, tails)
        kl = divergence(logits, targets[row])
        nll = torch.nn.functional.cross_entropy(logits, torch.tensor(tails, device='cuda'))
        loss = kl + .2 * nll
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if (index + 1) % 16 == 0:
            print('Output alignment step', index + 1, 'loss', round(loss.item(), 4), flush=True)
    original.mlp = teacher
    switch(decoder, teacher_layers)


def main(output, steps, output_steps, start, layer):
    sources = {path: sha(path) for path in (SOURCE, MODEL/'model.safetensors', *TRAIN_FILES)}
    torch.manual_seed(SEED)
    decoder, tokenizer, originals, replacements, _, _ = load_layer_pair()
    if layer not in replacements:
        raise ValueError(f'Layer {layer} is absent from the integrated replacement checkpoint')
    original = originals[layer]
    teacher = original.mlp
    student = SPORAFeedForward(copy.deepcopy(replacements[layer].block.ff)).requires_grad_(True)
    if start is not None:
        if read(start.parent/'result.json')['candidate_sha256'] != sha(start):
            raise ValueError('Incorrect starting candidate digest')
        payload = torch.load(start, map_location='cpu', weights_only=True)
        if payload['layer'] != layer or payload['source_sha256'] != sources[SOURCE]:
            raise ValueError('Incorrect starting candidate source')
        student.load_state_dict(payload['state'], strict=True)
        sources[start] = sha(start)
    texts = corpus()
    switch(decoder, originals)
    train_x, train_y = collect(decoder, tokenizer, texts[8:], teacher)
    held_x, held_y = collect(decoder, tokenizer, texts[:8], teacher)
    print('Captured activation rows', len(train_x), len(held_x), flush=True)
    before_local = relative_mse(student, held_x, held_y)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=.01)
    for index in range(steps):
        indices = torch.randint(len(train_x), (96,), device='cuda')
        target = train_y[indices]
        loss = (student(train_x[indices]) - target).square().mean() / target.square().mean().clamp_min(1e-8)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if (index + 1) % 32 == 0:
            print('Local distillation step', index + 1, 'loss', round(loss.item(), 4), flush=True)
    if output_steps:
        refine_output(decoder, tokenizer, student, layer, originals, replacements, texts[8:], output_steps)
    after_local = relative_mse(student, held_x, held_y)
    student.requires_grad_(False).eval()
    cases = encode_cases(tokenizer, tuple(zip(QUESTIONS, REFERENCES, strict=True)))
    targets = teacher_targets(decoder, cases)
    switch(decoder, replacements)
    integrated = evaluate(decoder, tokenizer, cases, targets)
    original.mlp = student
    switch(decoder, {layer: original})
    partial = evaluate(decoder, tokenizer, cases, targets)
    original.mlp = teacher
    switch(decoder, {layer: original})
    no_replacement = evaluate(decoder, tokenizer, cases, targets)
    if any(sha(path) != digest for path, digest in sources.items()):
        raise ValueError('Source data or checkpoint changed')
    atomic_save(dict(layer=layer, source_sha256=sources[SOURCE], state=student.state_dict()), output/'partial-layer.pt')
    rows = {'integrated': integrated, 'partial': partial, f'original_at_{layer}': no_replacement}
    summary = {name: dict(nll=sum(row['nll'] for row in group) / len(group),
                          kl=sum(row['kl'] for row in group) / len(group), eos=sum(row['eos'] for row in group))
               for name, group in rows.items()}
    local = dict(train_tokens=len(train_x), held_tokens=len(held_x), before=before_local, after=after_local)
    write(output/'result.json', dict(rows=rows, summary=summary, local=local, layer=layer, steps=steps,
                                     output_steps=output_steps,
                                     sources={str(path): digest for path, digest in sources.items()},
                                     candidate_sha256=sha(output/'partial-layer.pt')))
    print(local, summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--output-steps', type=int, default=0)
    parser.add_argument('--start', type=Path)
    parser.add_argument('--layer', type=int, default=6)
    args = parser.parse_args()
    if args.steps < 0 or args.output_steps < 0 or args.steps + args.output_steps < 1:
        parser.error('At least one training step is required')
    args.output.mkdir()
    main(args.output, args.steps, args.output_steps, args.start, args.layer)
