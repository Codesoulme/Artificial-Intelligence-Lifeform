"""Finite meta-training on random association worlds, from random weights."""
import argparse
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from .model import SPORA, ModelConfig
from .tasks import mappings, absorb
from .checkpoint import save_checkpoint, load_checkpoint


def validate(model, batches=8, batch_size=8, pairs=8, seed=100000):
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    totals = {'written': 0, 'no_write': 0, 'after_new_old': 0, 'after_new_new': 0}
    count = 0
    for _ in range(batches):
        keys, values = mappings(generator, batch_size, 2*pairs, model.config, device)
        initial, _ = model.initial_state(batch_size)
        fast, velocity = absorb(model, keys[:, :pairs], values[:, :pairs])
        with torch.no_grad():
            totals['written'] += (model.predict_many(keys[:, :pairs], fast).argmax(-1)==values[:, :pairs]).sum().item()
            totals['no_write'] += (model.predict_many(keys[:, :pairs], initial).argmax(-1)==values[:, :pairs]).sum().item()
        fast, velocity = absorb(model, keys[:, pairs:], values[:, pairs:], fast, velocity)
        with torch.no_grad():
            totals['after_new_old'] += (model.predict_many(keys[:, :pairs], fast).argmax(-1)==values[:, :pairs]).sum().item()
            totals['after_new_new'] += (model.predict_many(keys[:, pairs:], fast).argmax(-1)==values[:, pairs:]).sum().item()
        count += batch_size*pairs
    model.train()
    return {k: v/count for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--pairs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--run-dir', default='runs/p1-seed7')
    parser.add_argument('--checkpoint', default='checkpoints/p1-seed7.pt')
    parser.add_argument('--resume')
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1:
        parser.error('Steps, batch size and evaluation interval must be positive')
    if not args.resume and (Path(args.checkpoint).exists() or (Path(args.run_dir)/'metrics.jsonl').exists()):
        raise FileExistsError('Run already exists. Resume explicitly or choose new output paths.')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    generator = torch.Generator().manual_seed(args.seed+1000)
    start_step = 0
    if args.resume:
        model, _, _, saved = load_checkpoint(args.resume, device)
        state = saved['training_state']
        if state['seed'] != args.seed or state['pairs'] != args.pairs or state['batch_size'] != args.batch_size:
            raise ValueError('Resume must preserve seed and task configuration')
        start_step = state['step']
        generator.set_state(state['generator'])
        torch.set_rng_state(state['rng'])
    else:
        model = SPORA().to(device)
    if 2*args.pairs > model.config.key_count:
        raise ValueError('Too many pairs for disjoint retention test')
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    if args.resume:
        optimizer.load_state_dict(saved['outer_optimizer'])
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if device=='cuda':
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    info = {'model':'SPORA-T-P1','parameters':model.parameter_count(),'device':device,
            'torch':torch.__version__,'config':model.config_dict(),'arguments':vars(args),
            'initialization':'random, no pretrained tensors','exact_inner_gradient':True}
    (run_dir/'config.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(info),flush=True)
    model.train()
    with (run_dir/'metrics.jsonl').open('a',encoding='utf-8') as log:
        for step in range(start_step+1, start_step+args.steps+1):
            keys, values = mappings(generator,args.batch_size,args.pairs,model.config,device)
            optimizer.zero_grad(set_to_none=True)
            fast, velocity = absorb(model,keys,values,differentiable=True)
            logits = model.predict_many(keys,fast)
            task_loss = F.cross_entropy(logits.flatten(0,1), values.flatten())
            auxiliary = model.auxiliary_loss()
            loss = task_loss+.1*auxiliary
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite loss; previous checkpoint retained')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            if step==start_step+1 or step % 20==0:
                if device=='cuda': torch.cuda.synchronize()
                row={'step':step,'loss':loss.item(),'task_loss':task_loss.item(),
                     'accuracy':(logits.argmax(-1)==values).float().mean().item(),
                     'grad_norm':float(grad),'elapsed_seconds':time.perf_counter()-start}
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            if step % args.eval_every==0 or step==start_step+args.steps:
                metrics=validate(model,batches=4,batch_size=args.batch_size,pairs=args.pairs)
                row={'step':step,'development':metrics,'elapsed_seconds':time.perf_counter()-start}
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                save_checkpoint(args.checkpoint,model,optimizer=optimizer,
                    metadata={'stage':'P1 foundation','seed':args.seed,'step':step},
                    training_state={'step':step,'seed':args.seed,'pairs':args.pairs,
                                    'batch_size':args.batch_size,'generator':generator.get_state(),
                                    'rng':torch.get_rng_state()})
    summary={'completed_steps':step,'elapsed_seconds':time.perf_counter()-start,
             'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0,
             'development':metrics,'checkpoint':str(Path(args.checkpoint).resolve())}
    (run_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
