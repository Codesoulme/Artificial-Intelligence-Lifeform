"""Explicit held-out evaluation and bounded raw-byte generation after a pilot."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
import torch
from .growing import load
from .language_bridge import LanguageBridge, causal_loss, encode, BOS, EOS, PAD
from .language_training import load_interface
from .text_data import read_manifest, windows


@torch.no_grad()
def generate(model, state, prompt, max_new_bytes=48):
    if type(max_new_bytes) is not int or not 1 <= max_new_bytes <= 128:
        raise ValueError('Finite generation budget required')
    prefix = encode(prompt)[:-1]
    if len(prefix)+max_new_bytes > model.organism.config.max_length:
        raise ValueError('Prompt and generation budget exceed context')
    tokens = torch.tensor([prefix], device=next(model.parameters()).device)
    produced = []
    ended = False
    for _ in range(max_new_bytes):
        logits = model(tokens, state)[0, -1].clone()
        if not bool(torch.isfinite(logits).all()):
            raise ValueError('Nonfinite generation logits')
        logits[BOS] = logits[PAD] = -torch.inf
        token = int(logits.argmax())
        if token == EOS:
            ended = True
            break
        produced.append(token)
        tokens = torch.cat([tokens, tokens.new_tensor([[token]])], 1)
    raw = bytes(produced)
    try:
        text = raw.decode('utf-8')
        valid = True
    except UnicodeDecodeError:
        text = raw.decode('utf-8', errors='replace')
        valid = False
    return {'text': text, 'generated_byte_values': produced, 'utf8_valid': valid,
        'ended_with_eos': ended, 'budget_exhausted': not ended,
        'decoding': 'greedy raw bytes; BOS/PAD masked; no UTF-8 constraint or repair'}


@torch.no_grad()
def evaluate(model, state, examples):
    total, targets = 0., 0
    for row in examples:
        tokens = row[None].to(next(model.parameters()).device)
        count = int((tokens[:, 1:] != PAD).sum())
        total += float(causal_loss(model(tokens, state), tokens))*count
        targets += count
    if targets == 0:
        raise ValueError('Empty evaluation targets')
    return {'negative_log_likelihood': total/targets, 'bits_per_target': total/targets/math.log(2),
        'target_count': targets, 'target_unit': 'UTF-8 byte or EOS; not Chinese character',
        'uniform_259_token_nll': math.log(259)}


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser()
    parser.add_argument('--individual', required=True)
    parser.add_argument('--interface', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--split', choices=['development', 'test'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--length', type=int, default=128)
    parser.add_argument('--prompt', action='append', default=[])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():raise FileExistsError(output)
    digest = hashlib.sha256(Path(args.individual).read_bytes()).hexdigest()
    documents = read_manifest(args.manifest)
    torch.set_num_threads(2)
    organism, state, _ = load(args.individual, args.device)
    payload = torch.load(args.interface, map_location='cpu', weights_only=True)
    model = LanguageBridge(organism, context_layers=payload.get('context_layers', 0)).to(args.device).eval()
    load_interface(model, payload, digest)
    if not 2 <= args.length <= organism.config.max_length:
        raise ValueError('Invalid evaluation context')
    if payload.get('format') == 'spora-byte-training-v2':
        if payload['provenance']['manifest_sha256'] != hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest():
            raise ValueError('Evaluation must use the bound held-out manifest')
    result = evaluate(model, state, windows(documents, args.split, args.length))
    result.update(split=args.split, parent_sha256=digest,
        interface_sha256=hashlib.sha256(Path(args.interface).read_bytes()).hexdigest(),
        trained_tokens=payload.get('trained_tokens', 0),
        generation=[dict(prompt=p, **generate(model, state, p)) for p in args.prompt],
        limitations='Loss and selected completions do not establish comprehension or memory alignment')
    if hashlib.sha256(Path(args.individual).read_bytes()).hexdigest() != digest:
        raise AssertionError('Parent file changed')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':main()
