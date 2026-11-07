"""Measure microbatch choices on disposable copies of trained candidates."""
import gc
import json
import random
import statistics
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from spora.growing import load
from spora.conversation import restore_conversation
from spora.conversation_optimization import backward_supervised
from spora.language_training import interface_state, frozen_digest
from scripts.prepare_conversation_v2 import ROOT, sha
from scripts.train_conversation_v6 import text_batch


def main():
    out = ROOT / 'runs/p23-microbatch-speed.json'
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(2)
    results = []
    for variant, choices in [('small', (4, 8, 16)), ('wide', (4, 8))]:
        plan = json.loads((ROOT/f'configs/p23f-{variant}.json').read_text('utf-8'))
        path = ROOT/f'runs/p23f-{variant}/step000500.pt'
        original = sha(path)
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if sha(ROOT/plan['parent']) != plan['parent_sha256']:
            raise ValueError('Individual changed')
        organism, state, _ = load(ROOT/plan['parent'], 'cuda')
        before = frozen_digest(organism, state)
        stream = np.memmap(ROOT/plan['data']/'pretrain-train.bin', dtype=np.uint16, mode='r')
        x, y = text_batch(stream, random.Random(231601), 16, 512, 'cuda')
        reference = None
        for micro in choices:
            torch.cuda.empty_cache()
            model = restore_conversation(organism, payload, plan['parent_sha256'], payload['tokenizer_sha256'], 'cuda')
            opt = torch.optim.AdamW(model.trainable_parameters(), lr=.0001, weight_decay=.01, betas=(.9,.95))
            timings = []; first = None
            torch.cuda.reset_peak_memory_stats()
            for step in range(8):
                torch.cuda.synchronize(); start = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                loss = backward_supervised(model, state, x, y, micro)
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1., error_if_nonfinite=True)
                opt.step(); torch.cuda.synchronize()
                if step >= 2: timings.append(time.perf_counter()-start)
                if step == 0: first = interface_state(model)
            if reference is None: reference = first
            close = all(torch.allclose(v, reference[k], atol=2e-6, rtol=2e-5) for k,v in first.items())
            row = dict(variant=variant, microbatch=micro, effective_batch=16, length=512,
                       seconds_median=statistics.median(timings), seconds_mean=statistics.mean(timings),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(), loss=float(loss),
                       first_update_close_to_micro4=close,
                       max_first_update_difference=max(float((v-reference[k]).abs().max()) for k,v in first.items()))
            results.append(row); print(json.dumps(row), flush=True)
            del model, opt, first; gc.collect(); torch.cuda.empty_cache()
        if sha(path) != original or frozen_digest(organism,state) != before:
            raise AssertionError('Protected input changed')
        del x,y,organism,state,payload,reference; gc.collect(); torch.cuda.empty_cache()
    out.write_text(json.dumps(dict(results=results,
        scope='Disposable updates, no weights saved. Fixed actual pretraining batch, length512, float32; two warmup plus six timing updates. A microbenchmark, not end-to-end speed or chat quality.'), indent=2), encoding='utf-8')


if __name__ == '__main__': main()
