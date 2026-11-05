"""Atomic storage lifecycle. Existing .pt individuals remain independent."""
import argparse
import json
import torch
from .durable import TensorStore, adopt
from .growing import observe, grow, query, scalars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--individual', required=True)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('adopt'); create.add_argument('--source', required=True)
    learn = commands.add_parser('learn')
    learn.add_argument('key', type=int); learn.add_argument('value', type=int)
    learn.add_argument('--trusted', type=int, choices=[0, 1], default=1)
    expand = commands.add_parser('grow'); expand.add_argument('--pages', type=int, default=1)
    ask = commands.add_parser('ask'); ask.add_argument('key', type=int); ask.add_argument('--time', type=int)
    backup = commands.add_parser('backup'); backup.add_argument('--output', required=True)
    sleep = commands.add_parser('sleep'); sleep.add_argument('--page', type=int, required=True)
    sleep.add_argument('--rank', type=int, default=16); sleep.add_argument('--steps', type=int, default=600)
    commands.add_parser('info'); commands.add_parser('checkpoint'); commands.add_parser('compact')
    args = parser.parse_args(); torch.set_num_threads(2)
    if args.command == 'adopt':
        with adopt(args.source, args.individual) as store:
            result = {'revision': store.revision, 'adopted': str(store.path)}
    else:
        with TensorStore(args.individual) as store:
            model, state, meta, revision = store.read()
            if args.command == 'learn':
                updated = observe(model, state, args.key, args.value, args.trusted)
                result = store.commit(model, updated, revision)
            elif args.command == 'grow':
                result = store.commit(model, grow(model, state, args.pages), revision)
            elif args.command == 'backup':
                store.backup(args.output); result = {'backup': args.output, 'revision': revision}
            elif args.command == 'checkpoint':
                result = {'checkpoint': store.checkpoint()}
            elif args.command == 'compact':
                result = {'compaction': store.compact()}
            elif args.command == 'sleep':
                from .sleep_memory import consolidate_store
                result = consolidate_store(store, model, state, revision, args.page, args.rank, args.steps)
            elif args.command == 'info':
                result = {'revision': revision, 'observations': state.cursor, 'scalars': scalars(state), 'metadata': meta}
            else:
                with torch.no_grad():
                    logits = query(model, state, torch.tensor([args.key]),
                                   torch.tensor([int(args.time is not None)]), torch.tensor([args.time or 0]))
                value = int(logits.argmax(-1))
                result = {'value': None if value == model.config.unknown else value, 'revision': revision}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
