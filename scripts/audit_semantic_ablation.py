"""Remove only learned head weights, keeping counters/inputs fixed."""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spora.growing import load
from spora.semantic_memory import attributes


def main():
    torch.set_num_threads(2)
    directory=ROOT/'runs/p14-p15-audit-v2'
    rows=json.loads((directory/'audit.json').read_text())['concepts']
    results=[]
    with torch.no_grad():
        for row in rows:
            model,state,_=load(directory/f"concept-seed{row['seed']}-world{row['world']}.pt")
            keys=torch.arange(64);bits=attributes(keys).long()
            held=keys[bits.sum(-1)%2!=0]
            target=((bits[held][:,row['rule_axes']]^torch.tensor(row['rule_complements']))<<torch.arange(4)).sum(-1)
            module=model.semantic_memory
            probabilities=module(held).sigmoid()
            decoded=((probabilities>=.5).long()<<torch.arange(4)).sum(-1)
            snapshot=copy.deepcopy(module.state_dict())
            module.head.weight.zero_();module.head.bias.zero_()
            ablated=module(held).sigmoid()
            counters_exact=all(torch.equal(snapshot[k],module.state_dict()[k]) for k in ('trained_cursor','evidence_count'))
            module.load_state_dict(snapshot)
            restored=torch.equal(probabilities,module(held).sigmoid())
            record={'seed':row['seed'],'world':row['world'],
                'trained_head_accuracy':float((decoded==target).float().mean()),
                'zero_head_all_bits_half':bool((ablated==.5).all()),'counters_unchanged':counters_exact,
                'restored_exact':restored,'note':'Direct head probe; the saved corrected individual deliberately keeps this old rule stale'}
            record['passed']=record['trained_head_accuracy']>=.95 and record['zero_head_all_bits_half'] and counters_exact and restored
            results.append(record)
    result={'rows':results,'passed':all(r['passed'] for r in results)}
    destination=ROOT/'runs/p14-head-ablation.json'
    with destination.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
