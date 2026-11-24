"""Compare learned reply continuation against the existing decoder copy scores."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.audit_spora_latent_working_state import (
    DATA, HELD, MODEL, SEED, AutoTokenizer, dialogue_data, evaluate, evaluate_reader, fit_reader, frozen_digest,
    load_full_streamed, load_integrated_runtime, load_reader, measure, records, settings_for, sha, state_from,
    summarize, torch,
)
from spora.language_training import atomic_save
from spora.research_dialogue_working_state import WorkingTurnReader, WorkingTurnState


LAYER = 19
FORMAT = 'spora-ordered-copy-v1'
UNRELATED = (
    '先不管刚才的安排。用一句话解释为什么开水瓶要盖上塞子。',
    '换个话题，太阳从哪边升起？只答方位。',
    '上一件事暂停。我手里有五本书，借出两本，还剩几本？',
    '不用谈计划。用一句话告诉我纸箱湿了该怎么处理。',
)


def summary(rows):
    result = summarize(rows)
    result['current_fields'] = sum(sum(row['current']['present'].values()) for row in rows)
    result['field_count'] = sum(len(row['current']['present']) for row in rows)
    result['state_only'] = sum(row['state_only']['complete'] for row in rows if 'state_only' in row)
    result['state_only_count'] = sum('state_only' in row for row in rows)
    return result


def unrelated_outputs(decoder, reader, tokenizer, settings, snapshots):
    return [measure(decoder, reader, tokenizer, settings, [], question, state)['answer']
            for question, state in zip(UNRELATED, snapshots, strict=True)]


def main(args):
    if args.verify is None:
        args.output.mkdir(parents=True, exist_ok=False)
    original_sources = {path: sha(path) for path in (args.checkpoint, DATA)}
    reference_path = args.reader / f'layer-{LAYER}.json'
    protected = original_sources | {path: sha(path) for path in (reference_path, args.revision)}
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(args.checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    digest = frozen_digest(runtime.organism, runtime.state)
    parent, old, reader_path, reader_sha = load_reader(SimpleNamespace(output=args.reader, checkpoint=args.checkpoint),
                                                       LAYER, decoder, digest, original_sources)
    protected[reader_path] = reader_sha
    revision = json.loads(args.revision.read_text(encoding='utf-8'))
    if revision['sources'][str(reader_path)] != reader_sha or revision['individual_sha256'] != digest:
        raise ValueError('The fresh scenarios were evaluated on a different model')
    splits = {'previous': old['rows'], 'new': revision['oracle']}
    settings = settings_for(decoder, tokenizer, 48)
    snapshots = [state_from(row['case']['current'], tokenizer) for row in splits['new'][:len(UNRELATED)]]
    no_state = unrelated_outputs(decoder, parent, tokenizer, settings, [WorkingTurnState()] * len(UNRELATED))
    memory_cases = records(runtime.organism, HELD)
    memory = evaluate(decoder, tokenizer, runtime.organism, memory_cases)

    if args.verify is None:
        source = json.loads(DATA.read_text(encoding='utf-8'))
        train = dialogue_data(source, held=False, count=128)
        result = {'parent_scores': {name: summary(rows) for name, rows in splits.items()},
                  'unrelated': {'questions': UNRELATED, 'parent_off': no_state,
                                'parent_on': unrelated_outputs(decoder, parent, tokenizer, settings, snapshots)},
                  'steps': args.steps, 'training_cases': len(train), 'arms': {}}
        modes = ('standard', 'ordered')
    else:
        result = json.loads((args.output / 'comparison.json').read_text(encoding='utf-8'))
        if result['sources'] != {str(path): value for path, value in protected.items()} or result['individual_sha256'] != digest:
            raise ValueError('Working-state comparison sources changed')
        modes = (args.verify,)

    for name in modes:
        path = args.output / f'{name}.pt'
        torch.manual_seed(SEED)
        reader = WorkingTurnReader(decoder.base.config.hidden_size, layer_index=LAYER, ordered=name == 'ordered')
        if args.verify is None:
            missing, unexpected = reader.load_state_dict(parent.state_dict(), strict=False)
            if missing != (['order_scale'] if name == 'ordered' else []) or unexpected:
                raise ValueError('Candidate failed to inherit the reader weights')
            reader.to(decoder.base.device)
            trace = fit_reader(decoder, reader, tokenizer, train, args.steps, contrast=True)
        else:
            expected = result['arms'][name]
            if sha(path) != expected['sha256']:
                raise ValueError('Candidate reader changed before reload')
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if (saved['format'] != FORMAT or saved['mode'] != name or saved['parent_sha256'] != reader_sha
                    or saved['checkpoint_sha256'] != protected[args.checkpoint]):
                raise ValueError('Candidate reader metadata changed')
            reader.load_state_dict(saved['reader'], strict=True)
            reader.to(decoder.base.device)
        reader.eval().requires_grad_(False)
        scored = {}
        for split, references in splits.items():
            rows = evaluate_reader(decoder, reader, tokenizer, references, settings)
            for row in rows:
                for key in ('initial', 'current', 'state_only'):
                    if key in row:
                        row[key]['generated'].pop('seconds', None)
            scored[split] = {'summary': summary(rows), 'rows': rows}
            print(name, split, scored[split]['summary'], flush=True)
        ordinary = unrelated_outputs(decoder, reader, tokenizer, settings, snapshots)
        with reader.active(decoder, snapshots[0]):
            retained = evaluate(decoder, tokenizer, runtime.organism, memory_cases)
        if (retained != memory or frozen_digest(runtime.organism, runtime.state) != digest
                or any(sha(source) != value for source, value in protected.items())):
            raise ValueError('Working-state reader changed persistent memory or an experiment input')
        if args.verify is not None:
            if expected['splits'] != scored or expected['unrelated'] != ordinary or expected['memory'] != retained['summary']:
                raise ValueError('Reader reload did not reproduce the original results')
            with (args.output / f'{name}-reload.json').open('x', encoding='utf-8') as stream:
                json.dump({'identical': True, 'sha256': sha(path), 'memory': retained['summary']}, stream, indent=2)
            return
        atomic_save({'format': FORMAT, 'mode': name, 'parent_sha256': reader_sha,
                     'checkpoint_sha256': protected[args.checkpoint], 'reader': reader.cpu().state_dict()}, path)
        result['arms'][name] = {'splits': scored, 'unrelated': ordinary, 'memory': retained['summary'],
                                'trace': trace, 'sha256': sha(path)}
        del reader
    result['sources'] = {str(path): value for path, value in protected.items()}
    result['individual_sha256'] = digest
    result['peak_bytes'] = torch.cuda.max_memory_allocated()
    with (args.output / 'comparison.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('reader', type=Path)
    parser.add_argument('revision', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--steps', type=int, default=384)
    parser.add_argument('--verify', choices=('standard', 'ordered'))
    options = parser.parse_args()
    if options.steps < 1:
        parser.error('--steps must be positive')
    main(options)
