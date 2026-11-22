"""Measure a typed value-copy gate without changing the formal writer."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoTokenizer

from audit_spora_dialogue_working_state import FEATURE_DEPTH, FrozenTurnFeatures, SemanticLocator
from audit_spora_writer_transfer import write_metrics
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_dialogue_working_state import WRITER_WIDTH
from spora.integrated_durable import load_integrated_durable
from spora.language_training import frozen_digest
from spora.research_dialogue_working_state import FIELDS, TurnValueWriter
from spora.research_update_copy_gate import TypedValueCopyGate


HOLDOUT = ROOT / 'data/dialogue-read-plan-holdout.json'
TRANSFER = ROOT / 'data/dialogue-writer-transfer.json'
FORMAT = 'spora-working-turn-experiment-v1'


def evaluate(locator, tokenizer, holdout):
    rows = []
    for row in holdout:
        if hasattr(locator, 'reset'):
            locator.reset()
        current = row['fields'] | row['changes']
        case = {
            'turns': [row['first'], row['update'], row['question']],
            'initial': row['fields'], 'changes': row['changes'], 'current': current,
        }
        _, measures = write_metrics(locator, tokenizer, case)
        rows.append(measures)
    return rows


def score(rows):
    return {
        'state_exact': sum(row['state_exact'] for row in rows),
        'missing_cases': sum(bool(row['missing']) for row in rows),
        'wrong_cases': sum(bool(row['wrong']) for row in rows),
        'stale': sum(row['stale'] for row in rows),
    }


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    sources = {path: sha(path) for path in (HOLDOUT, TRANSFER, args.checkpoint, args.database, args.writer)}
    holdout = json.loads(HOLDOUT.read_text(encoding='utf-8'))
    transfer = json.loads(TRANSFER.read_text(encoding='utf-8'))
    payload = torch.load(args.writer, map_location='cpu', weights_only=True)
    if (payload['format'], payload['parent_sha256'], payload['fields'],
        payload['feature_depth'], payload['writer_width']) != (
            FORMAT, sources[args.checkpoint], FIELDS, FEATURE_DEPTH, WRITER_WIDTH):
        raise ValueError('Writer checkpoint is incompatible with the parent')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    with load_integrated_durable(args.database, args.checkpoint, load_full_streamed(), MODEL) as runtime:
        decoder = runtime.decoder.eval().requires_grad_(False)
        device = decoder.base.device
        digest = frozen_digest(runtime.organism, runtime.state)
        features = FrozenTurnFeatures(decoder, tokenizer)
        writer = TurnValueWriter(WRITER_WIDTH, feature_width=payload['hidden']).to(device)
        writer.load_state_dict(payload['writer'])
        fallback = SemanticLocator(writer.eval(), features)
        gated = TypedValueCopyGate(fallback)
        split_rows = {}
        for name, cases in (('read_plan', holdout), ('writer_transfer', transfer['held'])):
            baseline_rows = evaluate(fallback, tokenizer, cases)
            gated_rows = evaluate(gated, tokenizer, cases)
            split_rows[name] = {
                'baseline': score(baseline_rows), 'gated': score(gated_rows),
                'cases': [{'baseline': base, 'gated': gate}
                          for base, gate in zip(baseline_rows, gated_rows, strict=True)],
            }
        if digest != frozen_digest(runtime.organism, runtime.state):
            raise ValueError('Copy gate changed the durable individual')
    if any(sha(path) != value for path, value in sources.items()):
        raise ValueError('Copy gate input changed during audit')
    result = {
        'format': 'spora-typed-update-copy-gate-v1',
        'scores': {name: {key: value for key, value in split.items() if key != 'cases'}
                   for name, split in split_rows.items()},
        'cases': {name: split['cases'] for name, split in split_rows.items()},
        'digest': digest, 'inputs': {str(path): value for path, value in sources.items()},
        'seconds': time.perf_counter() - started,
        'peak_bytes': torch.cuda.max_memory_allocated(),
    }
    (args.output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(result['scores'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'database', 'writer', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
