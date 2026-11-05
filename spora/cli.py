"""Interact with a persistent symbolic individual without history injection."""
import argparse
import hashlib
from pathlib import Path
import torch
from .checkpoint import load_checkpoint, save_checkpoint


def main():
    parser=argparse.ArgumentParser(description='SPORA symbolic learner; not a language chatbot')
    parser.add_argument('--individual',default='checkpoints/individual.pt')
    sub=parser.add_subparsers(dest='command',required=True)
    create=sub.add_parser('create')
    create.add_argument('--foundation',required=True)
    learn=sub.add_parser('learn')
    learn.add_argument('key',type=int)
    learn.add_argument('value',type=int)
    ask=sub.add_parser('ask')
    ask.add_argument('key',type=int)
    sub.add_parser('info')
    args=parser.parse_args()
    torch.set_num_threads(2)
    path=Path(args.individual)
    if args.command=='create':
        if path.exists():
            raise FileExistsError('Individual exists; choose another path to create a branch')
        model,_,_,_=load_checkpoint(args.foundation)
        fast,velocity=model.initial_state(1)
        parent_hash=hashlib.sha256(Path(args.foundation).read_bytes()).hexdigest()
        save_checkpoint(path,model,fast,velocity,cursor=0,
                        metadata={'stage':'P1 individual','parent_sha256':parent_hash})
        print('Created',path.resolve())
        return
    model,fast,velocity,data=load_checkpoint(path)
    if args.command=='learn':
        if path.with_suffix('.lock').exists():
            raise RuntimeError('Another writer holds this individual')
        lock=path.with_suffix('.lock')
        handle=lock.open('x')
        try:
            with handle:
                # Reload after acquiring lock so concurrent writers cannot overwrite updates.
                model,fast,velocity,data=load_checkpoint(path)
                fast,velocity=model.write_event(fast,velocity,torch.tensor([args.key]),torch.tensor([args.value]))
                save_checkpoint(path,model,fast,velocity,cursor=data['cursor']+1,metadata=data['metadata'])
        finally:
            lock.unlink()
        print('Stored in neural parameters. Experience cursor:',data['cursor']+1)
    elif args.command=='ask':
        with torch.no_grad():
            probabilities=model(torch.tensor([args.key]),fast).softmax(-1)[0]
        value=int(probabilities.argmax())
        print(f'value={value}, probability={probabilities[value].item():.4f}; no write performed')
    else:
        print(f'parameters={model.parameter_count()}, experiences={data["cursor"]}, stage={data["metadata"].get("stage")}')


if __name__=='__main__':
    main()
