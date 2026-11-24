"""Compare pooled and sequential clause representations for multistep memory intent."""

import argparse
import json
from pathlib import Path

from scripts.audit_spora_context_write_intent import (
    CASES, CONTRAST_INTENT, MODEL, PAIRS, RECENCY_CASES, RECENCY_PAIRS, RECENCY_VALIDATION, SEED, TRAIN_INTENT,
    TRAIN_NAMES, TRAIN_VALUES, VALIDATION, AutoTokenizer, FrozenTurnFeatures, atomic_save, contexts, corpus,
    dialogue_text, fit_intent, frozen_digest, intent_margin, load_full_streamed, load_integrated_runtime,
    sha, store_json, torch,
)
from spora.research_clause_intent import ClauseIntent, clause_ends


FORMAT = 'spora-clause-intent-v1'
DEPTH = 20


def clause_features(featurize, rows, width):
    offsets = [clause_ends(row['text']) for row in rows]
    lengths = torch.tensor([len(bounds) for bounds in offsets])
    length = int(lengths.max())
    result = torch.zeros(len(rows), length, width, dtype=torch.float16)
    for first in range(0, len(rows), 64):
        batch = rows[first:first + 64]
        hidden = featurize([row['text'] for row in batch])
        for index, bounds in enumerate(offsets[first:first + len(batch)]):
            result[first + index, :len(bounds)].copy_(hidden[index, bounds])
    mask = torch.arange(length)[None, :] < lengths[:, None]
    return result, mask


def case_rows(cases):
    return [{'name': name, 'label': label, 'text': dialogue_text(earlier, final)}
            for name, positive, negative, final in cases for label, earlier in ((True, positive), (False, negative))]


def evaluate(rows, margins, floor):
    decisions = [{**row, 'score': score, 'accepted': score > floor} for row, score in zip(rows, margins, strict=True)]
    pairs = list(zip(decisions[::2], decisions[1::2], strict=True))
    return {'correct_write': sum(row['label'] and row['accepted'] for row in decisions),
            'false_write': sum(not row['label'] and row['accepted'] for row in decisions),
            'correct_pairs': sum(pos['accepted'] and not neg['accepted'] for pos, neg in pairs),
            'correct_order': sum(pos['score'] > neg['score'] for pos, neg in pairs), 'rows': decisions}


def main(checkpoint, output, verify):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    original = sha(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    source = {'checkpoint': original, 'individual': individual, 'depth': DEPTH}
    saved = torch.load(verify / 'heads.pt', weights_only=True, map_location='cpu') if verify else None
    if saved is not None and (saved['format'] != FORMAT or saved['source'] != source):
        raise ValueError('Clause heads belong to a different model')
    featurize = FrozenTurnFeatures(runtime.decoder, tokenizer, depth=DEPTH)
    width = runtime.decoder.base.config.hidden_size
    training = contexts(TRAIN_NAMES[:12], PAIRS + RECENCY_PAIRS)
    single = corpus(TRAIN_NAMES[12:], TRAIN_INTENT + CONTRAST_INTENT, TRAIN_VALUES)
    generator = torch.Generator().manual_seed(SEED)
    training += [single[i] for i in torch.randperm(len(single), generator=generator)[:len(training)].tolist()]
    train = clause_features(featurize, training, width) if saved is None else None
    calibration = contexts(('舒朝扬', '杜英河', '罗晓月', '顾树明'), VALIDATION + RECENCY_VALIDATION)
    recent = case_rows(RECENCY_CASES)
    held = case_rows(CASES)
    calibration_features = clause_features(featurize, calibration, width)
    evaluation = {'dialogue': (held, clause_features(featurize, held, width)),
                  'recency': (recent, clause_features(featurize, recent, width))}
    report, states = {}, {}
    for name, recurrent in (('pooled', False), ('recurrent', True)):
        torch.manual_seed(SEED)
        head = ClauseIntent(width, recurrent=recurrent)
        if saved is None:
            fit_intent(head, *train, training, epochs=10, learning_rate=.002)
        else:
            head.load_state_dict(saved['states'][name], strict=True)
            head.eval().requires_grad_(False)
        cal = intent_margin(head(*calibration_features)).tolist()
        floor = max(score for row, score in zip(calibration, cal, strict=True) if not row['label'])
        report[name] = {'floor': floor, 'calibration': evaluate(calibration, cal, floor),
                        'parameters': sum(parameter.numel() for parameter in head.parameters())}
        for key, (rows, inputs) in evaluation.items():
            margins = intent_margin(head(*inputs)).tolist()
            comparison = evaluate(rows, margins, floor)
            report[name][key] = comparison
            print(name, key, floor, {key: item for key, item in comparison.items() if key != 'rows'}, flush=True)
        states[name] = head.state_dict()
    if sha(checkpoint) != original or frozen_digest(runtime.organism, runtime.state) != individual:
        raise ValueError('Original model or individual changed')
    output.mkdir(parents=True, exist_ok=False)
    result = {'format': FORMAT, 'source': source, 'training': len(training), 'arms': report}
    if saved is None:
        atomic_save({'format': FORMAT, 'source': source, 'states': states}, output / 'heads.pt')
    else:
        reference = json.loads((verify / 'result.json').read_text(encoding='utf-8'))
        if result != reference:
            raise ValueError('Reloaded clause head differs from original audit')
        result = {'identical': True, 'source': source}
    store_json(output / 'result.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--verify', type=Path)
    main(**vars(parser.parse_args()))
