"""Compare the same fixed development targets with and without memory reads."""
import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F

from spora.conversation import restore_conversation
from spora.conversation_data import DialogueRows
from spora.growing import load
from spora.language_training import frozen_digest
from scripts.prepare_conversation_v2 import sha
from scripts.train_conversation_v10 import padded
from tokenizers import Tokenizer


@torch.no_grad()
def evaluate(model, state, stream, rows, pad, starts, count, device):
    text_sum = 0.0
    text_targets = 0
    for start in starts:
        seq = torch.as_tensor(np.array(stream[start:start + 513], dtype=np.int64), device=device).unsqueeze(0)
        text_sum += float(F.cross_entropy(model(seq[:, :-1], state).flatten(0, 1), seq[:, 1:].flatten(), reduction='sum'))
        text_targets += 512
    dialogue_sum = 0.0
    dialogue_targets = 0
    for index in range(min(count, len(rows))):
        row = rows[index]
        length = len(row['tokens']) - 1
        x, y = padded([row], length, pad, device)
        valid = int((y != -100).sum())
        dialogue_sum += float(F.cross_entropy(model(x, state).flatten(0, 1), y.flatten(), ignore_index=-100, reduction='sum'))
        dialogue_targets += valid
    return dict(text_nll=text_sum / text_targets, text_targets=text_targets,
                dialogue_nll=dialogue_sum / dialogue_targets, dialogue_targets=dialogue_targets,
                dialogue_rows=min(count, len(rows)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--windows', type=int, default=32)
    parser.add_argument('--rows', type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.windows <= 256 or not 1 <= args.rows <= 512:
        raise ValueError('Evaluation bounds exceeded')
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads((ROOT / args.plan).read_text('utf-8'))
    data = ROOT / 'runs/conversation-data-v5'
    manifest = json.loads((data / 'manifest.json').read_text('utf-8'))
    for name in ('tokenizer.json', 'pretrain-development.bin', 'dialogue-development-tokens.jsonl'):
        if sha(data / name) != manifest['files'][name]:
            raise ValueError('Development input changed: ' + name)
    parent = ROOT / plan['parent']
    checkpoint = ROOT / args.checkpoint
    checkpoint_sha = sha(checkpoint)
    if sha(parent) != plan['parent_sha256']:
        raise ValueError('Individual changed')
    torch.set_num_threads(2)
    device = 'cuda'
    organism, state, _ = load(parent, device)
    original = frozen_digest(organism, state)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = restore_conversation(organism, payload, plan['parent_sha256'], manifest['tokenizer_sha256'], device)
    tokenizer = Tokenizer.from_file(str(data / 'tokenizer.json'))
    rows = DialogueRows(data / 'dialogue-development-tokens.jsonl', 512)
    stream = np.memmap(data / 'pretrain-development.bin', dtype=np.uint16, mode='r')
    rng = random.Random(381110)
    starts = [rng.randrange(len(stream) - 512) for _ in range(args.windows)]
    normal = evaluate(model, state, stream, rows, tokenizer.token_to_id('<|pad|>'), starts, args.rows, device)
    hooks = [layer.register_forward_hook(lambda module, inputs, output: torch.zeros_like(output))
             for layer in model.memory_out]
    try:
        ablated = evaluate(model, state, stream, rows, tokenizer.token_to_id('<|pad|>'), starts, args.rows, device)
    finally:
        for hook in hooks:
            hook.remove()
        rows.close()
    if original != frozen_digest(organism, state) or checkpoint_sha != sha(checkpoint):
        raise AssertionError('Protected individual or checkpoint changed')
    result = dict(checkpoint_sha256=checkpoint_sha, source_step=payload['steps'],
                  individual_sha256=sha(parent), data_manifest_sha256=sha(data / 'manifest.json'),
                  source_code_sha256=sha(Path(__file__)), seed=381110, starts=starts,
                  normal=normal, memory_outputs_zeroed=ablated,
                  old_individual_exact=True, checkpoint_unchanged=True,
                  scope='Same targets under forward-hook ablation; NLL is not chat quality. No model weights or memories changed.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream_out:
        json.dump(result, stream_out, ensure_ascii=False, indent=2)
        stream_out.write('\n')
    print(json.dumps(dict(normal=normal, memory_outputs_zeroed=ablated)), flush=True)


if __name__ == '__main__':
    main()
