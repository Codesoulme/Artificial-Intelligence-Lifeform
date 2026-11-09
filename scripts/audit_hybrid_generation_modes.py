"""Compare greedy and seeded nucleus generation as a diagnostic, not an acceptance test."""
import argparse
import json
from pathlib import Path

import torch

from spora.growing import load
from spora.hybrid_tokenizer import HybridLanguageBridge, HybridTokenizer
from spora.language_bridge import BOS, EOS, PAD
from spora.language_training import frozen_digest, load_interface
from spora.train_hybrid import sha

PROMPTS = ('从前,有一个小女孩', '有一天,小明去公园', '你好', '我想学习中文')


def nucleus_choice(logits, generator, p=0.9):
    if not 0 < p <= 1:
        raise ValueError('p must be in (0, 1]')
    values, indices = torch.sort(logits.detach().cpu(), descending=True)
    probabilities = torch.softmax(values, dim=-1)
    keep = probabilities.cumsum(dim=-1) - probabilities < p
    probabilities = probabilities * keep
    probabilities /= probabilities.sum()
    return int(indices[torch.multinomial(probabilities, 1, generator=generator)].item())


@torch.no_grad()
def generate(model, state, tokenizer, prompt, mode, seed, max_new):
    tokens = tokenizer.encode(prompt)[:-1]
    if len(tokens) + max_new > model.organism.config.max_length:
        raise ValueError('Context budget exceeds individual limit')
    device = next(model.parameters()).device
    rng = torch.Generator().manual_seed(seed)
    output, ended = [], False
    for _ in range(max_new):
        logits = model(torch.tensor([tokens], device=device), state)[0, -1].clone()
        if not bool(torch.isfinite(logits).all()):
            raise ValueError('Nonfinite logits')
        logits[BOS] = logits[PAD] = -torch.inf
        token = int(logits.argmax()) if mode == 'greedy' else nucleus_choice(logits, rng)
        if token == EOS:
            ended = True
            break
        tokens.append(token)
        output.append(token)
    try:
        text, valid = tokenizer.decode(output), True
    except UnicodeDecodeError:
        text, valid = None, False
    repeated = sum(a == b for a, b in zip(output, output[1:]))
    return {'prompt': prompt, 'mode': mode, 'seed': seed, 'text': text, 'tokens': output,
            'utf8_valid': valid, 'ended': ended, 'adjacent_equal_tokens': repeated}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('individual', 'checkpoint', 'tokenizer', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    keys = ('individual', 'checkpoint', 'tokenizer')
    hashes = {key: sha(getattr(args, key)) for key in keys}
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if payload.get('format') != 'spora-hybrid-training-v1' or payload['provenance']['vocab_sha256'] != hashes['tokenizer']:
        raise ValueError('Checkpoint or tokenizer mismatch')
    tokenizer = HybridTokenizer.from_file(args.tokenizer)
    torch.set_num_threads(2)
    organism, state, _ = load(args.individual, args.device)
    before = frozen_digest(organism, state)
    model = HybridLanguageBridge(organism, tokenizer, hashes['tokenizer'], payload['context_layers']).to(args.device).eval()
    if payload['provenance'].get('train_backbone_copy') is True:
        model.enable_language_block_copies()
        model.eval()
    load_interface(model, payload, hashes['individual'])
    records = []
    for prompt in PROMPTS:
        records.append(generate(model, state, tokenizer, prompt, 'greedy', 0, 48))
        for seed in (20401, 20402):
            records.append(generate(model, state, tokenizer, prompt, 'nucleus', seed, 48))
    if before != frozen_digest(organism, state) or any(sha(getattr(args, key)) != hashes[key] for key in keys):
        raise AssertionError('Protected inputs changed')
    result = {'input_sha256': hashes, 'records': records, 'nucleus_p': 0.9, 'temperature': 1.0,
              'test_split_read': False, 'parent_and_inputs_unchanged': True,
              'limitations': ['Reused development prompts, not a blind or independent evaluation.',
                              'Sampling can change surface diversity and is not evidence of understanding or chat.',
                              'No repair or UTF-8 constraints are applied.']}
    with output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(records, ensure_ascii=False))


if __name__ == '__main__':
    main()
