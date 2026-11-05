"""Explicit bounded access to optional internal concept/action capabilities."""
import argparse
import json
import torch
from .durable import TensorStore
from .semantic_memory import semantic_query, consolidate_semantic_store


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--individual',required=True,help='Transactional SQLite individual')
    commands=parser.add_subparsers(dest='command',required=True)
    infer=commands.add_parser('infer')
    infer.add_argument('key',type=int)
    consolidate=commands.add_parser('consolidate')
    consolidate.add_argument('--steps',type=int,default=800)
    consolidate.add_argument('--seed',type=int,default=2510000)
    act=commands.add_parser('act')
    act.add_argument('--position',type=float,nargs=2,required=True)
    act.add_argument('--goal',type=float,nargs=2,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    with TensorStore(args.individual) as store:
        model,state,_,revision=store.read()
        if args.command=='infer':
            if not 0<=args.key<64:raise ValueError('Finite symbol outside 0..63')
            result=semantic_query(model,state,torch.tensor([args.key]))
            observed=bool(result['observed'][0]);inferred=bool(result['inferred'][0])
            result={'key':args.key,'value':int(result['values'][0]) if observed or inferred else None,
                'kind':'observed' if observed else ('inferred' if inferred else 'unknown'),
                'observation_cursor':state.cursor}
        elif args.command=='consolidate':
            result=consolidate_semantic_store(store,model,state,revision,steps=args.steps,seed=args.seed)
        else:
            if not hasattr(model,'skill_policy'):raise ValueError('No learned action policy')
            action=int(model.skill_policy.act(torch.tensor([args.position]),torch.tensor([args.goal])))
            result={'action':action,'displacement':[[0,0],[.2,0],[-.2,0],[0,.2],[0,-.2]][action],
                'domain':'numeric 2D reaching only','environment_action_executed':False}
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
