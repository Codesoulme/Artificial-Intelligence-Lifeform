"""Zero-initialized nonlinear low-rank residual on shared research KV writer."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
from torch import nn
import torch.nn.functional as F
from scripts.deepseek_local_baseline import sha
import scripts.deepseek_shared_kv_writer_20261127 as experiment

OUT=ROOT/'runs/research/residual-kv-writer-20261127-v1'
PARENT=ROOT/'runs/research/shared-kv-writer-20261127-v1/result.json'
RANK=16

class ResidualWriter(experiment.Writer):
    def __init__(self,layers):
        super().__init__(layers)
        self.key_in=nn.Parameter(torch.randn(layers,2,128,RANK)*0.02)
        self.key_out=nn.Parameter(torch.zeros(layers,2,RANK,128))
        self.value_in=nn.Parameter(torch.randn(layers,2,128,RANK)*0.02)
        self.value_out=nn.Parameter(torch.zeros(layers,2,RANK,128))
    def forward(self,source):
        pooled=super().forward(source);result=[]
        for layer,(key,value) in enumerate(pooled):
            def delta(x,a,b):
                hidden=torch.tanh(torch.einsum('bhsd,hdr->bhsr',F.layer_norm(x,(128,)),a))
                return torch.einsum('bhsr,hrd->bhsd',hidden,b)
            result.append((key+delta(key,self.key_in[layer],self.key_out[layer]),
                           value+delta(value,self.value_in[layer],self.value_out[layer])))
        return result

def main():
    if not PARENT.exists():raise FileNotFoundError(PARENT)
    source=Path(experiment.__file__).resolve();inputs={'baseline_result':sha(PARENT),'base_experiment_code':sha(source)}
    experiment.Writer=ResidualWriter;experiment.OUT=OUT
    experiment.main()
    after={'baseline_result':sha(PARENT),'base_experiment_code':sha(source)}
    report=json.loads((OUT/'result.json').read_text(encoding='utf8'))
    parent=json.loads(PARENT.read_text(encoding='utf8'))
    def key(row):return (row['phase'],row.get('split'),tuple(row.get('case',())),row['mode'])
    old={key(r):r for r in parent['rows'] if r['phase']=='before'}
    new={key(r):r for r in report['rows'] if r['phase']=='before'}
    comparison={'same_cases':set(old)==set(new),'compared':len(new),
                'same_token_sequences':sum(old[k]['tokens']==r['tokens'] for k,r in new.items() if k in old)}
    with (OUT/'comparison.json').open('x',encoding='utf8') as f:
        json.dump({'inputs':inputs,'inputs_after':after,'initial_comparison':comparison,'rank':RANK,
                   'candidate_sha256':sha(OUT/'candidate.pt')},f,ensure_ascii=False,indent=2)
    print(json.dumps({'initial_comparison':comparison,'protected_inputs_unchanged':inputs==after}),flush=True)

if __name__=='__main__':main()
