"""Post-selection pressure probe within the existing 64-key vocabulary."""
import argparse
import json
from pathlib import Path
import torch
from spora.checkpoint import load_checkpoint
from spora.tasks import mappings
from spora.p2 import Event,write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    output=Path(f'runs/p3-capacity-seed{args.seed}.json')
    if output.exists():raise FileExistsError('Preserve pressure evidence')
    base=Path(f'runs/p3-audit-seed{args.seed}')
    audit=json.loads((base/'audit.json').read_text(encoding='utf-8'))
    config=load_checkpoint(audit['checkpoint'])[0].config
    audit_generator=torch.Generator().manual_seed(910000)
    extra_generator=torch.Generator().manual_seed(920000)
    records=[]
    for world in range(4):
        keys,values=mappings(audit_generator,1,24,config,'cpu')
        used=set(keys[0].tolist())
        remaining=torch.tensor([k for k in range(config.key_count) if k not in used])
        extra_keys=remaining[torch.randperm(len(remaining),generator=extra_generator)][None,:]
        extra_values=torch.randint(config.value_count,extra_keys.shape,generator=extra_generator)
        stages=audit['worlds'][world]['stages']
        if len(stages)!=3 or not all(s['gate'] for s in stages):
            records.append({'world':world,'status':'no complete accepted chain'});continue
        model,fast,velocity,_=load_checkpoint(base/f'world{world}-stage3-candidate.pt')
        for count in (0,8,24,40):
            begin={0:0,8:0,24:8,40:24}[count]
            for i in range(begin,count):
                event=Event(extra_keys[:,i],extra_values[:,i],torch.ones(1),'new')
                fast,velocity,_=write(model,fast,velocity,event,'one')
            with torch.no_grad():
                old=model.predict_many(keys,fast).argmax(-1)
                fresh=model.predict_many(extra_keys[:,:count],fast).argmax(-1) if count else None
            records.append({'world':world,'total_facts':24+count,
                            'consolidated24':float((old==values).float().mean()),
                            'new_facts':float((fresh==extra_values[:,:count]).float().mean()) if count else None})
    result={'foundation_seed':args.seed,'worlds':4,'results':records,
            'limits':['Post-selection pressure probe; not used to select thresholds',
                      'Same 64-key vocabulary, not an expanded model or proven new capacity',
                      'No extra consolidation during pressure; default unchanged']}
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
