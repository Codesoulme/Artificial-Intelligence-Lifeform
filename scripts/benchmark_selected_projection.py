"""Compare dense and selected vocabulary projection on actual training rows."""
import gc
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from tokenizers import Tokenizer
from spora.growing import load
from spora.conversation import restore_conversation
from spora.conversation_optimization import backward_supervised
from spora.conversation_supervision import backward_selected
from spora.language_training import frozen_digest
from scripts.train_conversation_v6 import load_rows, padded, text_batch
from scripts.prepare_conversation_v2 import ROOT, sha


def main():
    out=ROOT/'runs/p23-selected-projection-speed.json'
    if out.exists(): raise FileExistsError(out)
    torch.set_num_threads(2);results=[]
    for variant in ('small','wide'):
        plan=json.loads((ROOT/f'configs/p23f-{variant}.json').read_text('utf-8'))
        path=ROOT/f'runs/p23f-{variant}/step000500.pt';digest=sha(path)
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if sha(ROOT/plan['parent'])!=plan['parent_sha256']:raise ValueError('Individual changed')
        organism,state,_=load(ROOT/plan['parent'],'cuda');before=frozen_digest(organism,state)
        tokenizer=Tokenizer.from_file(str(ROOT/plan['data']/'tokenizer.json'))
        teacher=load_rows(ROOT/plan['curriculum']/'teacher-tokens.jsonl',512)
        stream=np.memmap(ROOT/plan['data']/'pretrain-train.bin',dtype=np.uint16,mode='r')
        for kind in ('teacher','pretrain'):
            rng=random.Random(231607)
            if kind=='teacher':
                rows=[rng.choice(teacher) for _ in range(16)]
                length=32*math.ceil((max(len(r['tokens']) for r in rows)-1)/32)
                x,y=padded(rows,length,tokenizer.token_to_id('<|pad|>'),'cuda')
            else:x,y=text_batch(stream,rng,16,512,'cuda')
            reference=None;first_loss=None
            choices=[('dense',4),('selected',4),('selected',16 if variant=='small' else 8)]
            for mode,micro in choices:
                torch.cuda.empty_cache()
                model=restore_conversation(organism,payload,plan['parent_sha256'],payload['tokenizer_sha256'],'cuda')
                opt=torch.optim.AdamW(model.trainable_parameters(),lr=.0001,weight_decay=.01,betas=(.9,.95))
                backward=backward_supervised if mode=='dense' else backward_selected
                timings=[];torch.cuda.reset_peak_memory_stats()
                for step in range(8):
                    torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True)
                    loss=backward(model,state,x,y,micro)
                    if step==0:
                        first_grad=[p.grad.detach().cpu().clone() for p in model.trainable_parameters()]
                        if reference is None:reference=first_grad;first_loss=float(loss)
                        difference=max(float((a-b).abs().max()) for a,b in zip(first_grad,reference))
                        close=all(torch.allclose(a,b,atol=1e-5,rtol=1e-4) for a,b in zip(first_grad,reference))
                        loss_difference=abs(float(loss)-first_loss)
                        del first_grad
                    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),1.,error_if_nonfinite=True)
                    opt.step();torch.cuda.synchronize()
                    if step>=2:timings.append(time.perf_counter()-start)
                row=dict(variant=variant,kind=kind,mode=mode,microbatch=micro,
                    length=x.shape[1],supervised_tokens=int((y!=-100).sum()),input_positions=x.numel(),
                    seconds_median=statistics.median(timings),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    gradient_close=close,max_gradient_difference=difference,initial_loss_difference=loss_difference)
                results.append(row);print(json.dumps(row),flush=True)
                del model,opt;gc.collect();torch.cuda.empty_cache()
            del x,y,reference
        if sha(path)!=digest or frozen_digest(organism,state)!=before:raise AssertionError('Protected input changed')
        del organism,state,payload;teacher.close();gc.collect();torch.cuda.empty_cache()
    passed=all(r['gradient_close'] and r['initial_loss_difference']<1e-5 for r in results)
    out.write_text(json.dumps(dict(passed=passed,results=results,
        scope='Disposable trained-model copies; no updates saved. Real sampled teaching and pretraining batches. Two warmup and six timing iterations each. Short benchmark, not whole-run speed or chat quality.'),indent=2),encoding='utf-8')
    if not passed:raise SystemExit('Gradient check failed')


if __name__=='__main__':main()
