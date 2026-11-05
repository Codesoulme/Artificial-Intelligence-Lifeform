"""Kill a process while replacing dense history with a trained sealed page."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spora.growing import load,observe
from spora.durable import TensorStore,adopt
from audit_durable import fingerprint,size,worker_line


def child(path,replacement,stage):
    torch.set_num_threads(2)
    with TensorStore(path) as store:
        model,state,_,revision=store.read()
        next_model,next_state,_=load(replacement)
        # Keep the resident backbone objects for incremental hashing.
        if not all(torch.equal(v,next_model.state_dict()[k]) for k,v in model.state_dict().items()):
            raise AssertionError('Sleep must not replace backbone')
        def hook(current):
            if current==stage:
                print(json.dumps({'paused':stage}),flush=True);sys.stdin.readline()
        store.commit(model,next_state,revision,hook=hook)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output')
    parser.add_argument('--child');parser.add_argument('--replacement');parser.add_argument('--stage')
    a=parser.parse_args()
    if a.child:return child(a.child,a.replacement,a.stage)
    output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);torch.set_num_threads(2)
    source=ROOT/'runs/p10-sleep-audit/seed7-world0/before-stage0.pt'
    replacement=ROOT/'runs/p10-sleep-audit/seed7-world0/after-stage0.pt'
    model,old,_=load(source);_,new,_=load(replacement)
    report={'cases':[],'limitation':'Forced process termination, not physical power loss'}
    stages=json.loads((ROOT/'configs/p9-p10-plan.json').read_text())['storage']['crash_stages']
    for stage in stages:
        path=output/f'{stage}.sqlite'
        with adopt(source,path) as initial:initial.checkpoint()
        process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--child',str(path.resolve()),
            '--replacement',str(replacement),'--stage',stage],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            line=worker_line(process)
            if not line or json.loads(line).get('paused')!=stage:raise RuntimeError('Worker did not reach sleep commit boundary')
            expected=new if stage=='after_commit' else old
            with TensorStore(path) as reader:
                rm,rs,_,_=reader.read()
                assert fingerprint(rm,rs)==fingerprint(model,expected)
            process.kill();process.wait(timeout=10)
            with TensorStore(path) as recovered:
                rm,rs,_,revision=recovered.read()
                exact=fingerprint(rm,rs)==fingerprint(model,expected)
                next_a=observe(model,expected,12,7,1);next_b=observe(rm,rs,12,7,1)
                continuation=fingerprint(model,next_a)==fingerprint(rm,next_b)
                recovered.commit(rm,next_b,revision)
                before=size(path);free_before=recovered.connection.execute('PRAGMA freelist_count').fetchone()[0]
                started=time.perf_counter();recovered.compact();cost=(time.perf_counter()-started)*1000
                cm,cs,_,_=recovered.read()
                compact_exact=fingerprint(cm,cs)==fingerprint(rm,next_b)
                row={'stage':stage,'exact':exact,'next_write_exact':continuation,'reader_exact':True,
                     'compaction_exact':compact_exact,'bytes_before_compaction':before,'bytes_after_compaction':size(path),
                     'free_pages_before':free_before,'compaction_ms':cost}
            report['cases'].append(row);print(json.dumps(row),flush=True)
        finally:
            if process.poll() is None:process.kill();process.wait(timeout=10)
            for stream in (process.stdin,process.stdout,process.stderr):stream.close()
    report['passed']=all(r['exact'] and r['next_write_exact'] and r['compaction_exact'] for r in report['cases'])
    (output/'audit.json').write_text(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
