"""Train a decoder-based decision to read or bypass session working state."""

import argparse
import json
from pathlib import Path

from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_ordered_copy import (
    DATA, FORMAT, HELD, LAYER, MODEL, SEED, UNRELATED, AutoTokenizer, dialogue_data, evaluate,
    frozen_digest, load_full_streamed, load_integrated_runtime, measure, records, settings_for, sha, state_from, torch,
)
from scripts.audit_spora_dialogue_working_state import score_case
from spora.language_training import atomic_save
from spora.research_dialogue_working_state import WorkingTurnReader, WorkingTurnState


SOURCE = DATA.with_name('dialogue-working-state-read-gate-v1.json')
THRESHOLD = .5


def encode_questions(features, texts):
    hidden = features(texts).float()
    lengths = torch.tensor(list(map(len, texts)))
    valid = torch.arange(hidden.shape[1])[None, :] < lengths[:, None]
    mean = (hidden * valid[..., None]).sum(dim=1) / lengths[:, None]
    last = hidden[torch.arange(len(texts)), lengths - 1]
    return torch.cat((torch.nn.functional.normalize(mean), torch.nn.functional.normalize(last)), dim=-1)


def labelled_questions(source, gate_data):
    labels = {}
    for case in dialogue_data(source, held=False, count=128):
        labels[case['question']] = True
    for groups in gate_data['train'].values():
        for name, expected in (('read', True), ('skip', False)):
            for text in groups[name]:
                if text in labels and labels[text] != expected:
                    raise ValueError('Conflicting supervision for the same question')
                labels[text] = expected
    return labels


def main(args):
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    reader_path = args.copy / 'standard.pt'
    comparison_path = args.copy / 'comparison.json'
    reload_path = args.copy / 'standard-reload.json'
    protected = {path: sha(path) for path in (args.checkpoint, reader_path, comparison_path, reload_path, DATA, SOURCE)}
    copy_result = json.loads(comparison_path.read_text(encoding='utf-8'))
    replay = json.loads(reload_path.read_text(encoding='utf-8'))
    if not replay['identical'] or replay['sha256'] != protected[reader_path]:
        raise ValueError('Reader weights differ from the reproduced candidate')
    saved = torch.load(reader_path, map_location='cpu', weights_only=True)
    if saved['format'] != FORMAT or saved['mode'] != 'standard' or saved['checkpoint_sha256'] != protected[args.checkpoint]:
        raise ValueError('Reader is incompatible with the decoder checkpoint')
    gate_data = json.loads(SOURCE.read_text(encoding='utf-8'))
    source = json.loads(DATA.read_text(encoding='utf-8'))
    training = labelled_questions(source, gate_data)
    cases = copy_result['arms']['standard']['splits']['new']['rows']
    previous = copy_result['arms']['standard']['splits']['previous']['rows']
    batch = [(row, True, row['case']['question'], 'previous') for row in previous]
    batch += [(row, True, row['case']['question'], 'new') for row in cases]
    batch += [(row, False, text, 'change_topic') for row, text in zip(cases, UNRELATED)]
    batch += [(cases[index], read, text, 'hard') for index, read, text in gate_data['held']]
    texts = [text for _, _, text, _ in batch]
    if training.keys() & set(texts):
        raise ValueError('Read decision evaluation overlaps training')

    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(args.checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    digest = frozen_digest(runtime.organism, runtime.state)
    if digest != copy_result['individual_sha256']:
        raise ValueError('Individual differs from the reader comparison')
    reader = WorkingTurnReader(decoder.base.config.hidden_size, layer_index=LAYER)
    reader.load_state_dict(saved['reader'], strict=True)
    reader.to(decoder.base.device).eval().requires_grad_(False)
    features = FrozenTurnFeatures(decoder, tokenizer)
    if args.verify:
        result = json.loads((args.output / 'comparison.json').read_text(encoding='utf-8'))
        path = args.output / 'gate.pt'
        if (result['sources'] != {str(name): value for name, value in protected.items()}
                or result['individual_sha256'] != digest or result['gate_sha256'] != sha(path)):
            raise ValueError('Read decision inputs changed before reload')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if (checkpoint['format'] != 'spora-working-state-read-gate-v1'
                or checkpoint['reader_sha256'] != protected[reader_path]
                or checkpoint['checkpoint_sha256'] != protected[args.checkpoint]
                or checkpoint['threshold'] != THRESHOLD):
            raise ValueError('Read decision checkpoint is incompatible')
        held_x = encode_questions(features, texts)
        gate = torch.nn.Linear(held_x.shape[1], 1)
        gate.load_state_dict(checkpoint['gate'], strict=True)
        gate.eval().requires_grad_(False)
        probabilities = gate(held_x).squeeze(-1).sigmoid().tolist()
        if texts != [row['question'] for row in result['rows']]:
            raise ValueError('Read decision questions changed')
        for probability, expected in zip(probabilities, result['rows'], strict=True):
            if abs(probability - expected['probability']) > 1e-6 or (probability >= THRESHOLD) != expected['read']:
                raise ValueError('Read decision changed after reload')
        with (args.output / 'reload.json').open('x', encoding='utf-8') as stream:
            json.dump({'read_decisions_identical': len(probabilities), 'gate_sha256': sha(path)}, stream, indent=2)
            stream.write('\n')
        print('Reloaded read decisions:', len(probabilities), flush=True)
        return
    train_x = encode_questions(features, list(training))
    train_y = torch.tensor(list(training.values()), dtype=torch.float32)
    gate = torch.nn.Linear(train_x.shape[1], 1)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=.015, weight_decay=.05)
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(gate(train_x).squeeze(-1), train_y)
        loss.backward()
        optimizer.step()
    gate.eval().requires_grad_(False)
    probabilities = gate(encode_questions(features, texts)).squeeze(-1).sigmoid().tolist()
    settings = settings_for(decoder, tokenizer, 48)
    rows = []
    for (reference, expected, question, split), probability in zip(batch, probabilities, strict=True):
        case = reference['case']
        history = reference['history']
        state = state_from(case['current'], tokenizer)
        empty = WorkingTurnState()
        off = measure(decoder, reader, tokenizer, settings, history, question, empty)
        on = measure(decoder, reader, tokenizer, settings, history, question, state)
        read = probability >= THRESHOLD
        chosen = measure(decoder, reader, tokenizer, settings, history, question, state if read else empty)
        for answer in (off, on, chosen):
            answer.pop('seconds')
        if chosen != (on if read else off):
            raise ValueError('Read decision did not reproduce its chosen route')
        scores = ({'off': score_case(case, off)['complete'], 'on': score_case(case, on)['complete'],
                   'gated': score_case(case, chosen)['complete']} if expected else {})
        rows.append({'split': split, 'question': question, 'expected': expected, 'probability': probability,
                     'read': read, 'off': off, 'on': on, 'gated': chosen, 'scores': scores})
        print(split, f'{probability:.3f}', expected, read, scores, flush=True)
    memories = records(runtime.organism, HELD)
    baseline = evaluate(decoder, tokenizer, runtime.organism, memories)
    with reader.active(decoder, state):
        retained = evaluate(decoder, tokenizer, runtime.organism, memories)
    if (baseline != retained or frozen_digest(runtime.organism, runtime.state) != digest
            or any(sha(path) != value for path, value in protected.items())):
        raise ValueError('Read decision experiment changed a protected input or persistent response')
    path = args.output / 'gate.pt'
    atomic_save({'format': 'spora-working-state-read-gate-v1', 'gate': gate.state_dict(),
                 'reader_sha256': protected[reader_path], 'threshold': THRESHOLD,
                 'checkpoint_sha256': protected[args.checkpoint]}, path)
    result = {'sources': {str(path): value for path, value in protected.items()}, 'reader_sha256': protected[reader_path],
              'individual_sha256': digest, 'gate_sha256': sha(path), 'train_count': len(training),
              'train_positive': int(train_y.sum()), 'threshold': THRESHOLD, 'memory': retained['summary'],
              'peak_bytes': torch.cuda.max_memory_allocated(), 'rows': rows}
    with (args.output / 'comparison.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('copy', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
