"""Explicit numeric observations and bounded maintenance for controlled individuals."""
import argparse,json
import torch
from .durable import TensorStore
from .internal_memory import observe_sourced,read_provenance,consolidation_request,SOURCES
from .learned_read import prepare_router
from .sleep_memory import consolidate_store


def main():
    p=argparse.ArgumentParser();p.add_argument('--individual',required=True)
    sub=p.add_subparsers(dest='command',required=True)
    learn=sub.add_parser('learn');learn.add_argument('key',type=int);learn.add_argument('value',type=int)
    learn.add_argument('--source',choices=SOURCES,required=True);learn.add_argument('--quality',type=float,required=True)
    learn.add_argument('--valid-at',type=int)
    source=sub.add_parser('source');source.add_argument('key',type=int);source.add_argument('--time',type=int,required=True)
    sub.add_parser('info');sub.add_parser('prepare-read')
    sleep=sub.add_parser('consolidate');sleep.add_argument('--min-information',type=int,default=16)
    a=p.parse_args();torch.set_num_threads(2)
    with TensorStore(a.individual) as store:
        model,state,_,revision=store.read()
        if state.internal is None:raise ValueError('Individual does not have internal controls')
        if a.command=='learn':
            updated,result=observe_sourced(model,state,a.key,a.value,a.source,a.quality,a.valid_at)
            result['commit']=store.commit(model,updated,revision)
        elif a.command=='source':result=read_provenance(model,state,a.key,a.time)
        elif a.command=='prepare-read':
            updated,training=prepare_router(model,state)
            result={'training':training,'commit':store.commit(model,updated,revision),'note':'Optional learned mode; default reading remains exact.'}
        elif a.command=='consolidate':
            result=consolidation_request(model,state,a.min_information)
            if result['pages']:
                result['sleep']=consolidate_store(store,model,state,revision,result['pages'][0])
        else:result={'cursor':state.cursor,'revision':revision,'internal':state.internal.metadata(),
                     'consolidation':consolidation_request(model,state),'sources_are_declared':True}
        print(json.dumps(result))


if __name__=='__main__':main()
