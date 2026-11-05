"""Short CPU cost comparison, including actual incremental storage."""
import json,statistics,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spora.growing import load,observe
from spora.internal_memory import observe_sourced,read_provenance
from spora.durable import TensorStore


def main():
    torch.set_num_threads(2)
    output=ROOT/'runs/p12-cost'
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    old,old_state,_=load(ROOT/'checkpoints/individual-time-candidate.pt')
    model,state,_=load(ROOT/'checkpoints/individual-internal-v2-candidate.pt')
    times={'old_trusted_write':[],'sourced_write':[],'source_read':[]}
    for _ in range(3):
        changed,_=observe_sourced(model,state,12,8,'direct',1.)
        observe(old,old_state,12,8,1)
    for _ in range(21):
        start=time.perf_counter();observe(old,old_state,12,8,1);times['old_trusted_write'].append((time.perf_counter()-start)*1000)
        start=time.perf_counter();changed,_=observe_sourced(model,state,12,8,'direct',1.);times['sourced_write'].append((time.perf_counter()-start)*1000)
        start=time.perf_counter();read_provenance(model,changed,12,72);times['source_read'].append((time.perf_counter()-start)*1000)
    with TensorStore.create(output/'commit.sqlite',model,state) as store:
        start=time.perf_counter();stats=store.commit(model,changed,0);stats['ms']=(time.perf_counter()-start)*1000
        store.checkpoint()
    report={'median_ms':{k:statistics.median(v) for k,v in times.items()},'commit':stats,
            'repetitions':21,'threads':2,'scope':'CPU warm in-process timing; no peak RAM/energy benchmark; one commit sample',
            'repeat_note':'Repeated measurements start from the same snapshot; do not count as distinct personal events'}
    (output/'profile.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
