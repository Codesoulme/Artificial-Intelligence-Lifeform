"""Compare paired answers on relation-sensitive development cases, read-only."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tokenizers import Tokenizer
from scripts.converse_research import prefix_tokens
from scripts.evaluate_context_contrast import score
from scripts.prepare_conversation_v2 import sha
from spora.conversation import restore_conversation
from spora.growing import load
from spora.language_training import frozen_digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--plan', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    cases_path = ROOT / 'data/context-shortcut-diagnostic.json'
    dataset = json.loads(cases_path.read_text(encoding='utf8'))
    plan = json.loads((ROOT / args.plan).read_text(encoding='utf8'))
    parent, checkpoint = ROOT / plan['parent'], ROOT / args.checkpoint
    vocab = ROOT / plan['data'] / 'tokenizer.json'
    if sha(parent) != plan['parent_sha256']:
        raise ValueError('Individual changed')
    before = sha(checkpoint)
    torch.set_num_threads(2)
    organism, state, _ = load(parent, 'cuda')
    frozen = frozen_digest(organism, state)
    model = restore_conversation(organism, torch.load(checkpoint, map_location='cpu', weights_only=True),
                                 plan['parent_sha256'], sha(vocab), 'cuda')
    tokenizer = Tokenizer.from_file(str(vocab))
    rows = []
    for case in dataset['cases']:
        if len(case['contexts']) != 2 or len(case['answers']) != 2 or case['answers'][0] == case['answers'][1]:
            raise ValueError('Expected two counterfactual contexts and two distinct answers')
        pair = []
        for context, expected in zip(case['contexts'], case['answers']):
            prefix = prefix_tokens(tokenizer, [('user', context)])
            scores = [score(model, state, tokenizer, prefix, a) for a in case['answers']]
            winner = case['answers'][int(scores[1] > scores[0])]
            pair.append(dict(context=context, expected=expected, scores=scores,
                             winner=winner, correct=winner == expected, tie=scores[0] == scores[1]))
        rows.append(dict(name=case['name'], answers=case['answers'], pair=pair,
                         both_correct=all(r['correct'] and not r['tie'] for r in pair)))
    if sha(checkpoint) != before or frozen_digest(organism, state) != frozen:
        raise AssertionError('Protected state changed')
    report = dict(scope=dataset['scope'], checkpoint_sha256=before, cases_sha256=sha(cases_path),
                  evaluator_sha256=sha(Path(__file__)), scoring_source_sha256=sha(ROOT / 'scripts/evaluate_context_contrast.py'),
                  both_correct=sum(r['both_correct'] for r in rows), total=len(rows), rows=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({'both_correct': report['both_correct'], 'total': report['total']}))


if __name__ == '__main__':
    main()
