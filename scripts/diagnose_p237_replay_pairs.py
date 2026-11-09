"""Compare P223 and P234 on the replay candidate's matched targets; no training."""
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from scripts.train_p111_chunk_coverage import padded
from spora.conversation import restore_conversation
from spora.dialogue_candidate import validate_candidate
from spora.dialogue_replay_coverage import DialogueReplayCoverage
from spora.growing import load
from spora.language_training import frozen_digest
from spora.paired_order import PairedOrder

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    'individual': ('checkpoints/individual-prechinese-v3.pt', '6269d7749148c95ed4653928f61642c8d93098defed649baab3f807c71494db9'),
    'tokenizer': ('runs/conversation-data-v5/tokenizer.json', 'b3629001784896437c4aa9f279f6b4cdc5a534be191b537009e7293f971fda20'),
    'parent': ('runs/p223-reviewed-mixture-pilot-20261109/checkpoint.pt', 'e5df3ba8c17c8fd2c5e0d50d687cab69a72136bb9fe0ca5bf7bda973298d9cd8'),
    'replay': ('runs/p234-dialogue-replay-diagnostic-20261109/checkpoint.pt', '861ab1c1635d9cddfaae11cad474aea1c2cade0ee4a20ef95dab9142a1fd224c'),
}
MANIFEST = '050f7f8e057320e73f642f51b771695261679e16e76f33c1a0bc22df849e5e1e'
DATA = ROOT/'runs/p108-reviewed-pairs-v1'
OUTPUT = ROOT/'runs/p237-replay-pair-diagnostic-20261109.json'


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def paired_rows(directory):
    result = []
    for name in ('context.jsonl', 'last-user.jsonl'):
        with (directory/name).open(encoding='utf-8') as stream:
            result.append([json.loads(line) for line in stream])
    PairedOrder(*result, seed=0, epochs=1, length=512)
    return result


@torch.no_grad()
def score(model, state, row, pad, device):
    length = len(row['tokens'])-1
    x, y = padded([row], length, pad, device)
    logits = model(x, state).flatten(0, 1)
    targets = y.flatten()
    count = int((targets != -100).sum())
    if count < 1:
        raise ValueError('Empty target')
    loss = F.cross_entropy(logits.float(), targets, ignore_index=-100, reduction='sum')
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite loss')
    return float(loss), count


def aggregate(rows):
    groups = {}
    for category in ('all', 'seen', 'unseen'):
        selected = [r for r in rows if category == 'all' or (r['presentations'] > 0) == (category == 'seen')]
        tokens = sum(r['targets'] for r in selected)
        if not tokens:
            groups[category] = {'pairs': 0, 'targets': 0}
            continue
        groups[category] = {'pairs': len(selected), 'targets': tokens,
            **{f'{model}_{arm}_nll': sum(r[model][arm] for r in selected)/tokens
               for model in ('parent', 'replay') for arm in ('context', 'last_user')}}
    return groups


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    paths = {name: ROOT/path for name, (path, _) in INPUTS.items()}
    if any(file_hash(paths[name]) != expected for name, (_, expected) in INPUTS.items()):
        raise ValueError('Input changed')
    candidate = validate_candidate(DATA, MANIFEST, INPUTS['tokenizer'][1], 512, 8192)
    context, last = paired_rows(DATA)
    last = {row['pair_id']: row for row in last}
    source = candidate['files']['context.jsonl']
    records = [dict(source=source, pair_id=row['pair_id'], mask=row['mask']) for row in context]
    torch.set_num_threads(2)
    organism, state, _ = load(paths['individual'], 'cuda')
    before = frozen_digest(organism, state)
    replay_payload = torch.load(paths['replay'], map_location='cpu', weights_only=True)
    coverage = DialogueReplayCoverage.from_dict(records, replay_payload['dialogue_coverage']).summaries()
    models = {}
    for name in ('parent', 'replay'):
        payload = replay_payload if name == 'replay' else torch.load(paths[name], map_location='cpu', weights_only=True)
        models[name] = restore_conversation(organism, payload, INPUTS['individual'][1], INPUTS['tokenizer'][1], 'cuda')
    pad = Tokenizer.from_file(str(paths['tokenizer'])).token_to_id('<|pad|>')
    rows = []
    for row in context:
        item = {'pair_id': row['pair_id'], 'presentations': coverage[(source, row['pair_id'])]['presentations']}
        for name, model in models.items():
            context_loss, count = score(model, state, row, pad, 'cuda')
            last_loss, other_count = score(model, state, last[row['pair_id']], pad, 'cuda')
            if count != other_count or ('targets' in item and item['targets'] != count):
                raise ValueError('Targets differ')
            item['targets'] = count
            item[name] = {'context': context_loss, 'last_user': last_loss}
        rows.append(item)
    if (any(file_hash(paths[name]) != expected for name, (_, expected) in INPUTS.items())
            or frozen_digest(organism, state) != before
            or validate_candidate(DATA, MANIFEST, INPUTS['tokenizer'][1], 512, 8192) != candidate):
        raise ValueError('Protected input changed')
    result = {'status': 'training_candidate_diagnostic_not_generalization',
              'inputs': {name: value[1] for name, value in INPUTS.items()},
              'candidate_manifest_sha256': MANIFEST, 'script_sha256': file_hash(Path(__file__)),
              'summary': aggregate(rows), 'rows': rows,
              'limitations': ['P234 sampled this same pool during training; P223 may also have encountered it.',
                              'Unseen means not sampled in P234 only, not independent of ancestors.',
                              'Both contexts have identical targets; no-context targets may still be predictable.',
                              'Teacher forcing and NLL do not measure free-generation quality.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
