"""Supplemental standardized restart check; preserve original one-shot audit."""
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spora.checkpoint import load_checkpoint
from spora.audit_p4 import isolated_query


def main():
    destination=ROOT/'runs'/'p4-restart-standardized.json'
    if destination.exists():raise FileExistsError('Preserve previous verification')
    results=[]
    for kind in ('split','shared'):
        for seed in (7,11,23):
            directory=ROOT/'runs'/f'p4-{kind}-seed{seed}-audit'
            audit=json.loads((directory/'audit.json').read_text(encoding='utf-8'))
            checkpoint=directory/'restart-individual.pt'
            before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            model,fast,_,_=load_checkpoint(checkpoint)
            q=audit['restart']['queries']
            tensors=[torch.tensor(q[name]).reshape(1,-1) for name in ('keys','modes','times')]
            with torch.no_grad():
                torch.set_num_threads(4)
                four=model.predict_many(tensors[0],fast,tensors[1],tensors[2]).softmax(-1)
                torch.set_num_threads(2)
                two=model.predict_many(tensors[0],fast,tensors[1],tensors[2]).softmax(-1)
            cold=isolated_query(checkpoint,**q)
            remote=torch.tensor(cold['probabilities']).unsqueeze(0)
            after=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            results.append({'kind':kind,'seed':seed,'original_audit_exact':audit['restart']['identical_probabilities'],
                'same_threads_exact':torch.equal(two,remote),'same_threads_prediction_match':two.argmax(-1).flatten().tolist()==cold['predictions'],
                'threads4_vs2_max_probability_difference':float((four-two).abs().max()),
                'threads4_vs2_prediction_match':torch.equal(four.argmax(-1),two.argmax(-1)),
                'file_unchanged':before==after,'sha256':after})
    result={'reason':'First audit loaded before the same-thread fix; original results preserved',
            'settings':'CPU, two threads locally and in fresh worker','models':results}
    destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
