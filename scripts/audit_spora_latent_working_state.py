"""Compare internal working-state reads at different layers of the migrated decoder."""

import argparse
import json
from pathlib import Path

from scripts.audit_spora_dialogue_working_state import (DATA, FEATURE_DEPTH, SEED, WRITER_WIDTH,
                                                        FrozenTurnFeatures, SemanticLocator, dialogue_data, encode,
                                                        fit_reader, fit_writer, measure, rollout_states, score_case,
                                                        state_from)
from scripts.audit_spora_chat_backbones import reply
from scripts.deepseek_local_baseline import MODEL, sha, torch
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_qwen_counterfactual_reply_20261122 import HELD, evaluate, records
from scripts.train_spora_dialogue_composition import settings_for
from spora.integrated_checkpoint import load_integrated_runtime
from spora.language_training import atomic_save, frozen_digest
from spora.research_dialogue_working_state import TurnValueWriter, WorkingTurnReader
from transformers import AutoTokenizer


FORMAT = 'spora-latent-working-state-reader-v1'


def baselines(decoder, tokenizer, cases, settings):
    rows = []
    for case in cases:
        history = []
        for question, _ in (case['first'], case['update']):
            answer = reply(decoder, tokenizer, history, question, **settings)['answer']
            history.extend(({'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}))
        generated = reply(decoder, tokenizer, history, case['question'], **settings)
        rows.append({'case': case, 'history': history, 'off': {'generated': generated, **score_case(case, generated)}})
    return rows


def initial_version(case):
    return {**case, 'initial': case['current'], 'current': case['initial'],
            'changes': {field: case['initial'][field] for field in case['changes']}}


def evaluate_reader(decoder, reader, tokenizer, rows, settings):
    scored = []
    for index, row in enumerate(rows):
        case, history = row['case'], row['history']
        initial = state_from(case['initial'], tokenizer)
        current = state_from(case['current'], tokenizer)
        previous = measure(decoder, reader, tokenizer, settings, history, case['question'], initial)
        updated = measure(decoder, reader, tokenizer, settings, history, case['question'], current)
        before = score_case(initial_version(case), previous)
        after = score_case(case, updated)
        entry = {**row, 'initial': {'generated': previous, **before}, 'current': {'generated': updated, **after}}
        if index % 4 == 0:
            state_only = measure(decoder, reader, tokenizer, settings, [], case['question'], current)
            entry['state_only'] = {'generated': state_only, **score_case(case, state_only)}
        scored.append(entry)
    return scored


def summarize(rows):
    return {
        'off': sum(row['off']['complete'] for row in rows),
        'initial': sum(row['initial']['complete'] for row in rows),
        'current': sum(row['current']['complete'] for row in rows),
        'both_states': sum(row['initial']['complete'] and row['current']['complete']
                           and row['initial']['generated']['answer'] != row['current']['generated']['answer']
                           for row in rows),
        'count': len(rows),
    }


@torch.inference_mode()
def likelihood_probe(decoder, reader, tokenizer, rows):
    results = []
    for row in rows:
        case = row['case']
        states = {name: state_from(case[name], tokenizer) for name in ('initial', 'current')}
        answers = {'initial': case['initial_answer'], 'current': case['answer']}
        query = {'role': 'user', 'content': case['question']}
        history = row['history']
        result = {'question': case['question']}
        for context, messages in (('visible_history', [*history, query]), ('state_only', [query])):
            prompt = encode(tokenizer, messages)
            bonuses = {}
            for requested, answer in answers.items():
                target = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
                tokens = torch.tensor([prompt + target[:-1]], device=decoder.base.device)
                labels = torch.tensor(target, device=tokens.device)
                scores = {}
                for state_name, state in states.items():
                    with reader.active(decoder, state):
                        logits = decoder(input_ids=tokens, use_cache=False, logits_to_keep=len(target)).logits[0].float()
                    scores[state_name] = float(logits.log_softmax(dim=-1).gather(-1, labels[:, None]).mean())
                alternate = 'current' if requested == 'initial' else 'initial'
                bonuses[requested] = scores[requested] - scores[alternate]
            result[context] = bonuses
        results.append(result)
    return results


def load_reader(args, index, decoder, digest, protected):
    path = args.output / f'layer-{index}.pt'
    reference = json.loads((args.output / f'layer-{index}.json').read_text(encoding='utf-8'))
    reader_sha = sha(path)
    if (reference['reader_sha256'] != reader_sha or reference['individual_sha256'] != digest
            or reference['sources'] != {str(name): value for name, value in protected.items()}):
        raise ValueError('Working-state reader or source changed before reload')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if (saved['format'] != FORMAT or saved['layer'] != index
            or saved['checkpoint_sha256'] != protected[args.checkpoint]):
        raise ValueError('Working-state reader belongs to a different checkpoint')
    reader = WorkingTurnReader(decoder.base.config.hidden_size, layer_index=index,
                               copy_logits=saved.get('copy_logits', False))
    reader.load_state_dict(saved['reader'], strict=True)
    reader.to(decoder.base.device).eval().requires_grad_(False)
    return reader, reference, path, reader_sha


def verify_reader(args, index, runtime, tokenizer, settings, digest, protected):
    destination = args.output / f'layer-{index}-reload.json'
    if destination.exists():
        raise FileExistsError(destination)
    decoder = runtime.decoder
    reader, reference, path, reader_sha = load_reader(args, index, decoder, digest, protected)
    rows = [{key: row[key] for key in ('case', 'history', 'off')} for row in reference['rows']]
    observed = evaluate_reader(decoder, reader, tokenizer, rows, settings)
    likelihood = likelihood_probe(decoder, reader, tokenizer, rows)
    for rows in (observed, reference['rows']):
        for row in rows:
            for arm in ('off', 'initial', 'current', 'state_only'):
                if arm in row:
                    row[arm]['generated'].pop('seconds', None)
    if (observed != reference['rows'] or sha(path) != reader_sha
            or frozen_digest(runtime.organism, runtime.state) != digest
            or any(sha(name) != value for name, value in protected.items())):
        raise ValueError('Working-state reader reload changed an input or answer')
    with destination.open('x', encoding='utf-8') as stream:
        json.dump({'identical': True, 'scores': summarize(observed), 'reader_sha256': reader_sha,
                   'likelihood': likelihood}, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('Reload', index, summarize(observed), flush=True)


def audit_writer(args, index, runtime, tokenizer, settings, digest, protected):
    destination = args.output / f'layer-{index}-writer.json'
    writer_path = args.output / f'layer-{index}-writer.pt'
    if destination.exists() or writer_path.exists():
        raise FileExistsError(destination)
    decoder = runtime.decoder
    reader, reference, reader_path, reader_sha = load_reader(args, index, decoder, digest, protected)
    payload = json.loads(DATA.read_text(encoding='utf-8'))
    train = dialogue_data(payload, held=False, count=128)
    features = FrozenTurnFeatures(decoder, tokenizer)
    torch.manual_seed(SEED)
    hidden = decoder.base.config.hidden_size
    writer = TurnValueWriter(WRITER_WIDTH, feature_width=hidden).to(decoder.base.device)
    trace = fit_writer(writer, train, payload['no_write'], args.writer_epochs, decoder.base.device, features=features)
    writer.eval().requires_grad_(False)
    locator = SemanticLocator(writer, features)
    rows = []
    for row in reference['rows']:
        case = row['case']
        turns = [case['first'][0], case['update'][0], case['question']]
        state, writes = rollout_states(locator, tokenizer, turns)
        written = writes[0] | writes[1] | writes[2]
        initial = state_from(writes[0], tokenizer)
        earlier = measure(decoder, reader, tokenizer, settings, row['history'], case['question'], initial)
        updated = measure(decoder, reader, tokenizer, settings, row['history'], case['question'], state)
        rows.append({'case': case, 'writes': writes, 'written': written, 'state_correct': written == case['current'],
                     'initial': {'generated': earlier, **score_case(initial_version(case), earlier)},
                     'current': {'generated': updated, **score_case(case, updated)}})
    if (sha(reader_path) != reader_sha or frozen_digest(runtime.organism, runtime.state) != digest
            or any(sha(name) != value for name, value in protected.items())):
        raise ValueError('Working-state writer audit changed an input')
    atomic_save({'format': FORMAT, 'checkpoint_sha256': protected[args.checkpoint], 'reader_sha256': reader_sha,
                 'feature_depth': FEATURE_DEPTH, 'writer': writer.cpu().state_dict()}, writer_path)
    summary = {'state_exact': sum(row['state_correct'] for row in rows),
               'answer_complete': sum(row['current']['complete'] for row in rows),
               'both_states': sum(row['initial']['complete'] and row['current']['complete']
                                  and row['initial']['generated']['answer'] != row['current']['generated']['answer']
                                  for row in rows),
               'oracle_complete': reference['scores']['current'], 'count': len(rows)}
    with destination.open('x', encoding='utf-8') as stream:
        json.dump({'summary': summary, 'rows': rows, 'trace': trace, 'writer_sha256': sha(writer_path),
                   'reader_sha256': reader_sha}, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('Automatic writer', index, summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--layers', nargs='+', type=int, default=(19, 27))
    parser.add_argument('--steps', type=int, default=384)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-layer', type=int, nargs='+')
    mode.add_argument('--audit-writer-layer', type=int)
    parser.add_argument('--writer-epochs', type=int, default=6)
    parser.add_argument('--copy-logits', action='store_true', help='Train the optional learned copy-score path')
    args = parser.parse_args()
    layers = args.verify_layer or args.layers
    if args.audit_writer_layer is not None:
        layers = (args.audit_writer_layer,)
    if min(args.steps, args.writer_epochs) < 1 or len(set(layers)) != len(layers) or min(layers) < 0:
        parser.error('Use positive step counts and distinct nonnegative layer indices')
    if args.verify_layer is None and args.audit_writer_layer is None:
        args.output.mkdir(parents=True, exist_ok=False)
    protected = {path: sha(path) for path in (args.checkpoint, DATA)}
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(args.checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    if max(layers) >= len(decoder.base.model.layers):
        raise ValueError('Selected working-state layer is outside the decoder')
    digest = frozen_digest(runtime.organism, runtime.state)
    settings = settings_for(decoder, tokenizer, 48)
    if args.verify_layer is not None:
        for index in layers:
            verify_reader(args, index, runtime, tokenizer, settings, digest, protected)
        return
    if args.audit_writer_layer is not None:
        audit_writer(args, args.audit_writer_layer, runtime, tokenizer, settings, digest, protected)
        return
    payload = json.loads(DATA.read_text(encoding='utf-8'))
    train = dialogue_data(payload, held=False, count=128)
    held = [case for index, case in enumerate(dialogue_data(payload, held=True, count=12)) if index % 12 >= 8]
    rows = baselines(decoder, tokenizer, held, settings)
    torch.cuda.empty_cache()
    memory_cases = records(runtime.organism, HELD)
    memory = evaluate(decoder, tokenizer, runtime.organism, memory_cases)
    expected = sum(row['off']['complete'] for row in rows)
    print('Baseline', expected, 'of', len(rows), 'memory', memory['summary'], flush=True)
    for index in args.layers:
        torch.manual_seed(SEED)
        reader = WorkingTurnReader(decoder.base.config.hidden_size, layer_index=index,
                                   copy_logits=args.copy_logits).to(decoder.base.device)
        trace = fit_reader(decoder, reader, tokenizer, train, args.steps, contrast=True)
        reader.requires_grad_(False)
        scored = evaluate_reader(decoder, reader, tokenizer, rows, settings)
        with reader.active(decoder, state_from(held[0]['current'], tokenizer)):
            retained = evaluate(decoder, tokenizer, runtime.organism, memory_cases)
        if retained != memory or frozen_digest(runtime.organism, runtime.state) != digest:
            raise ValueError('Working-state reader changed the persistent memory path')
        candidate = args.output / f'layer-{index}.pt'
        atomic_save({'format': FORMAT, 'checkpoint_sha256': protected[args.checkpoint], 'layer': index,
                     'copy_logits': args.copy_logits, 'reader': reader.cpu().state_dict()}, candidate)
        result = {'layer': index, 'steps': args.steps, 'copy_logits': args.copy_logits,
                  'scores': summarize(scored), 'trace': trace, 'rows': scored,
                  'memory': retained['summary'], 'memory_rows_identical': True, 'individual_sha256': digest,
                  'reader_sha256': sha(candidate), 'sources': {str(path): value for path, value in protected.items()}}
        if any(sha(path) != value for path, value in protected.items()):
            raise ValueError('A protected working-state experiment input changed')
        with (args.output / f'layer-{index}.json').open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        print(index, result['scores'], flush=True)
        del reader
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
