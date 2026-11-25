"""Score held conversation turns with the trained pending and intent heads."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_spora_dialogue_evidence import load_spora_for_eval, store_json
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.deepseek_local_baseline import sha, torch
from scripts.spora_pending_replay_data import REPLIES
from scripts.train_spora_pending_events import APPROVE, classifier, features


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    metadata = json.loads((args.weights.parent / 'result.json').read_text(encoding='utf-8'))
    if sha(args.weights) != metadata['weights_sha256']:
        raise ValueError('Expected the verified pending-event checkpoint')
    started = perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chat = load_spora_for_eval(args.output.parent)
    if {k: v for k, v in chat.awake_individual.items() if k != 'memory_path'} != {
            k: v for k, v in metadata['runtime'].items() if k != 'memory_path'}:
        raise ValueError('The pending-event heads use a different backbone')
    chat.tokenizer.padding_side = 'right'
    extractor = FrozenTurnFeatures(chat.decoder, chat.tokenizer)
    rows = [{'passages': value[:3]} for value in REPLIES.values()]
    context, state = features(extractor, rows, True)
    reply, _ = features(extractor, rows, False)
    weights = torch.load(args.weights, map_location='cpu', weights_only=True)
    predictions = {}
    with torch.no_grad():
        for key, data in (('state', state), ('context', context), ('reply', reply)):
            model = classifier(data.shape[-1], 2 if key == 'state' else 3).eval()
            model.load_state_dict(weights[key])
            predictions[key] = model(data).argmax(1).tolist()
    outcomes = {}
    for index, (key, value) in enumerate(REPLIES.items()):
        pending = bool(predictions['state'][index])
        outcomes[key] = dict(expected=value[3], pending=pending,
                             context=pending and predictions['context'][index] == APPROVE,
                             reply=pending and predictions['reply'][index] == APPROVE)
    store_json(args.output, dict(outcomes=outcomes, weights_sha256=sha(args.weights),
                                 runtime=chat.awake_individual, seconds=perf_counter() - started))
    print(outcomes, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('weights', type=Path)
    parser.add_argument('output', type=Path)
    main(parser.parse_args())
