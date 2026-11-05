"""Fixed illustrative timeline, stored only as an audit artifact, not a release."""
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spora.temporal_cli import create,learn
from spora.audit_p4 import isolated_query


def main():
    torch.set_num_threads(2)
    directory=ROOT/'runs'/'p4-fixed-demo'
    if directory.exists():raise FileExistsError('Preserve demonstration evidence')
    directory.mkdir(parents=True)
    individual=directory/'audit-only-individual.pt'
    create(individual,ROOT/'checkpoints'/'p4-split-seed7.pt')
    for value,trust in ((5,1),(9,1),(3,0)):learn(individual,12,value,trust)
    before=hashlib.sha256(individual.read_bytes()).hexdigest()
    request={'keys':[12,12,12,12,12,13],'modes':[0,1,1,1,1,1],'times':[0,0,1,2,3,0]}
    answers=isolated_query(individual,**request)
    reset=isolated_query(individual,**request,reset=True)
    after=hashlib.sha256(individual.read_bytes()).hexdigest()
    result={'purpose':'fixed illustrative audit, not a published candidate','request':request,
            'expected':[9,5,9,3,16,16],'prediction':answers['predictions'],
            'reset_prediction':reset['predictions'],'read_only_unchanged':before==after,
            'sha256':after,'unknown_class':16}
    (directory/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
