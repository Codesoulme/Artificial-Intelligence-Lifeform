"""Calibrate a question about an address without authorizing a tensor read."""

import argparse
from collections import Counter
import json
from pathlib import Path
from shutil import copyfile
from types import SimpleNamespace

import torch

from scripts import audit_spora_spoken_query_plan as plan
from scripts.audit_spora_natural_lookup_cli import digest
from scripts.audit_spora_spoken_query_data import calibration_data
from scripts.spora_version_chat import FrozenTurnFeatures, base, local_model, spoken_lookup_planner
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.research_version_dialogue_scope import eligible_spoken_lookup
from spora.research_version_store import MemoryAddress


def measure(cutoff, rows):
    counts = Counter()
    for question, gold, candidates, scored in rows:
        candidate = plan.proposed_address(scored, candidates, cutoff) if eligible_spoken_lookup(question) else None
        expected = (MemoryAddress(candidates[gold[1]], plan.FIELDS_ORDER[gold[2]]) if gold[0] == plan.READ else None)
        counts.update(correct=candidate is not None and candidate == expected,
                      wrong_address=candidate is not None and expected is not None and candidate != expected,
                      unnecessary=candidate is not None and expected is None)
    return counts


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    path = args.previous / 'result.json'
    candidate = args.previous / 'public.pt'
    report = json.loads(path.read_text(encoding='utf-8'))
    incoming = (digest(path), digest(candidate))
    names, calibration = calibration_data()
    texts = [text for text, _ in calibration]
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    with local_model(args.recurrent) as model:
        spoken_lookup_planner(args.previous, args.recurrent, model, tokenizer)
        head = plan.QueryPlanProbe(model.config.hidden_size).eval()
        head.load_state_dict(torch.load(candidate, map_location='cpu', weights_only=True))
        extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=plan.DEPTH)
        input_states, owners = plan.vectors(extract, texts, names, 'user-span')
        rows = [(text, gold, candidates, score) for (text, gold), candidates, score in
                zip(calibration, owners, plan.predict(head, input_states), strict=True)]
    old = report['results']['public']['threshold']
    choices = sorted({0., .5, .6, .7, .8, .85, .9, .95, .99, 1.000001, old})
    totals = {cutoff: measure(cutoff, rows) for cutoff in choices}

    def quality(cutoff):
        row = totals[cutoff]
        errors = row['wrong_address'] + row['unnecessary']
        return row['correct'] - 2 * errors, -errors, cutoff

    chosen = max(choices, key=quality)
    if (digest(path), digest(candidate)) != incoming:
        raise ValueError('Inputs changed during calibration')
    copyfile(candidate, args.output / 'public.pt')
    report['address_proposal'] = {'threshold': chosen, 'previous': old, 'calibration_count': len(rows),
                                  'metrics': totals, 'input_report_sha256': incoming[0]}
    store_json(args.output / 'result.json', report)
    print('old', old, totals[old], 'chosen', chosen, totals[chosen], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--recurrent', type=Path, nargs=3, required=True)
    main(parser.parse_args())
