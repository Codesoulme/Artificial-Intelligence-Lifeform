"""Controlled reconstruction ablation using the previous probe's exact token split."""
import json
import time
from pathlib import Path

from scripts import deepseek_spora_long_memory_generate_parity as p
from safetensors import safe_open

ROOT = p.ROOT
PREV = ROOT / 'runs/continuous-embedding-bridge-pilot-20261112'
OUT = ROOT / 'runs/continuous-embedding-bridge-comparison-20261112.json'
SEEDS = (761, 762)
WIDTHS = (22, 90)
STEPS = 400


def score(pred, targets, offsets):
    """Candidate rank with the target score taken from the same matrix product."""
    torch = p.torch
    logits = torch.nn.functional.normalize(pred.float(), dim=-1) @ torch.nn.functional.normalize(targets.float(), dim=-1).T
    indices = torch.arange(pred.shape[0], device=pred.device)
    target_logits = logits[indices, offsets]
    ranks = (logits > target_logits[:, None]).sum(-1) + 1
    tied = (logits == target_logits[:, None]).sum(-1)
    cosine = torch.nn.functional.cosine_similarity(pred.float(), targets[offsets].float(), dim=-1)
    return {'top1_strict_unique': int(((ranks == 1) & (tied == 1)).sum()), 'count': len(ranks),
            'top1_optimistic': int((ranks == 1).sum()), 'ties': int((tied > 1).sum()),
            'mean_rank': float(ranks.float().mean()), 'cosine_mean': float(cosine.mean())}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch = p.torch
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    previous_sha = p.sha(PREV / 'bridge.pt')
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Unexpected base weights')
    previous = json.loads((PREV / 'result.json').read_text(encoding='utf-8'))
    if previous['candidate_sha256'] != previous_sha or previous['base_sha256'] != base_sha:
        raise ValueError('Previous candidate mismatch')
    ids = previous['train_ids'] + previous['held_ids']
    if len(set(ids)) != len(ids) or len(ids) != 320:
        raise ValueError('Unexpected token split')
    with safe_open(p.MODEL / 'model.safetensors', framework='pt', device='cpu') as f:
        embedding = f.get_tensor('model.embed_tokens.weight').index_select(0, torch.tensor(ids)).float().cuda()
    train = embedding[:256]
    offsets_train = torch.arange(256, device='cuda')
    offsets_held = torch.arange(256, 320, device='cuda')
    saved = torch.load(PREV / 'bridge.pt', map_location='cpu', weights_only=True)
    old_proj = saved['projection'].cuda()
    old_layer = torch.nn.Linear(90, embedding.shape[1], bias=False, device='cuda')
    old_layer.load_state_dict(saved['decoder'])
    with torch.no_grad():
        old_codes = torch.nn.functional.normalize(embedding @ old_proj, dim=-1)
        old_input = torch.nn.functional.pad(old_codes, (0, 90 - old_codes.shape[-1]))
        old_pred = old_layer(old_input)
        rescored = {'train': score(old_pred[:256], embedding, offsets_train),
                    'held': score(old_pred[256:], embedding, offsets_held)}
    del old_layer, old_pred, old_codes, old_input, saved, old_proj
    rows = []
    for seed in SEEDS:
        for width in WIDTHS:
            for train_projection in (False, True):
                for objective in ('cosine', 'contrast'):
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    gen = torch.Generator(device='cpu').manual_seed(seed)
                    projection = torch.nn.Parameter(torch.randn((embedding.shape[1], width), generator=gen).cuda(),
                                                    requires_grad=train_projection)
                    decoder = torch.nn.Linear(width, embedding.shape[1], bias=False, device='cuda')
                    torch.nn.init.normal_(decoder.weight, std=0.02)
                    params = list(decoder.parameters()) + ([projection] if train_projection else [])
                    optim = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-5, foreach=False)
                    loss_val = None
                    for step in range(STEPS):
                        optim.zero_grad(set_to_none=True)
                        code = torch.nn.functional.normalize(train @ projection, dim=-1)
                        predicted = decoder(code)
                        norm_pred = torch.nn.functional.normalize(predicted, dim=-1)
                        norm_target = torch.nn.functional.normalize(train, dim=-1)
                        if objective == 'cosine':
                            loss = 1 - (norm_pred * norm_target).sum(-1).mean()
                        else:
                            logits = (norm_pred @ norm_target.T) / 0.07
                            loss = torch.nn.functional.cross_entropy(logits, offsets_train)
                        if not bool(torch.isfinite(loss)):
                            raise ValueError('Nonfinite loss')
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(params, 5.0, error_if_nonfinite=True)
                        optim.step()
                        loss_val = float(loss.detach())
                    with torch.no_grad():
                        pred = decoder(torch.nn.functional.normalize(embedding @ projection, dim=-1))
                        row = {'seed': seed, 'width': width, 'train_projection': train_projection,
                               'objective': objective, 'last_loss': loss_val,
                               'train': score(pred[:256], embedding, offsets_train),
                               'held': score(pred[256:], embedding, offsets_held)}
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    del decoder, projection, params, optim, pred
    if p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(PREV / 'bridge.pt') != previous_sha:
        raise ValueError('Protected input changed')
    report = {'previous_rescored': rescored, 'previous_reported': previous['after'], 'rows': rows,
              'steps': STEPS, 'base_sha256': base_sha, 'previous_candidate_sha256': previous_sha,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Finite 320 candidate token embedding set, held tokens used only for evaluation.',
                              'Inputs are embedding rows, not natural-language memory writes or model generation.',
                              '90 dimensions allocated to one token displaces other slots; not matched sequence capacity.',
                              'Two seeds and one limited split; no held tuning or production deployment.',
                              'Contrastive training discriminates among training tokens only.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'rescored': rescored, 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
