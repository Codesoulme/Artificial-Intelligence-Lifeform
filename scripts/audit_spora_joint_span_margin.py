"""Measure the joint span writer's write threshold without retraining it."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoTokenizer

from audit_spora_dialogue_working_state import FEATURE_DEPTH, FrozenTurnFeatures
from audit_spora_writer_transfer import write_metrics
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_durable import load_integrated_durable
from spora.research_joint_span_writer import JointSpanWriter


DATA = ROOT / 'data/dialogue-read-plan-holdout.json'


class Locator:
    def __init__(self, writer, features, device, margin):
        self.writer, self.features, self.device, self.margin = writer, features, device, margin

    def __call__(self, text):
        return self.writer.write(text, self.margin, features=self.features([text]).to(self.device))


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    holdout = json.loads(DATA.read_text(encoding='utf-8'))
    payload = torch.load(args.writer, map_location='cpu', weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    args.output.mkdir(parents=True, exist_ok=False)
    with load_integrated_durable(args.database, args.checkpoint, load_full_streamed(), MODEL) as runtime:
        decoder = runtime.decoder.eval().requires_grad_(False)
        device = decoder.base.device
        features = FrozenTurnFeatures(decoder, tokenizer)
        writer = JointSpanWriter(payload['writer_width'], feature_width=payload['hidden']).to(device)
        writer.load_state_dict(payload['writer'])
        writer.eval()
        scores = {}
        cases = {}
        for margin in args.margins:
            locator = Locator(writer, features, device, margin)
            rows = []
            for row in holdout:
                fields = row['fields'] | row['changes']
                case = {
                    'turns': [row['first'], row['update'], row['question']],
                    'initial': row['fields'], 'changes': row['changes'], 'current': fields,
                }
                _, measures = write_metrics(locator, tokenizer, case)
                rows.append(measures)
            scores[str(margin)] = {
                'state_exact': sum(row['state_exact'] for row in rows),
                'missing': sum(bool(row['missing']) for row in rows),
                'wrong': sum(bool(row['wrong']) for row in rows),
                'stale': sum(row['stale'] for row in rows),
            }
            cases[str(margin)] = rows
        result = {'scores': scores, 'cases': cases,
                  'sources': {str(path): sha(path) for path in (DATA, args.checkpoint, args.database, args.writer)}}
        (args.output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(scores, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'database', 'writer', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--margins', type=float, nargs='+', default=(0., .5, 1., 1.5, 2.))
    main(parser.parse_args())
