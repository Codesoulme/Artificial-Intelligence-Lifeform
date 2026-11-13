"""Compare jointly learned slot projection with the fixed-projection bridge."""
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_lane_embedding_bridge_pilot import Bridge, metrics, OUTPUT as PRIOR, BATCH, SEED, STEPS
from scripts.deepseek_spora_prefix_features import read_bf16_tensor
from scripts.deepseek_spora_structured_lane_noise_audit import projection_for

OUTPUT = ROOT / 'runs/deepseek-spora-lane-joint-projection-pilot-20261113'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = PRIOR / 'result.json'
    prior = json.loads(source.read_text(encoding='utf-8'))
    path = MODEL / 'model.safetensors'
    if sha(path) != WEIGHT_SHA or prior['base_sha256'] != WEIGHT_SHA:
        raise ValueError('Base model mismatch')
    train_list, held_list = prior['train_ids'], prior['held_ids']
    if len(train_list) != 8192 or len(held_list) != 320 or set(train_list) & set(held_list):
        raise ValueError('Unexpected token split')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    pool = []
    for token in range(tokenizer.vocab_size):
        text = tokenizer.decode([token], skip_special_tokens=False)
        if ('\u4e00' <= text[:1] <= '\u9fff'
                and tokenizer.encode(text, add_special_tokens=False) == [token]):
            pool.append(token)
    if not (set(train_list) | set(held_list)).issubset(pool):
        raise ValueError('Tokens no longer in candidate pool')
    with path.open('rb') as stream:
        header_size = struct.unpack('<Q', stream.read(8))[0]
        if header_size > 100_000_000 or header_size > path.stat().st_size - 8:
            raise ValueError('Invalid tensor header')
        header = json.loads(stream.read(header_size))
        raw = read_bf16_tensor(stream, header, 8 + header_size, path.stat().st_size,
                               'model.embed_tokens.weight').cuda().float()
    embeddings = F.normalize(raw, dim=-1)
    train = torch.tensor(train_list, device='cuda', dtype=torch.long)
    held = torch.tensor(held_list, device='cuda', dtype=torch.long)
    candidates = torch.tensor(sorted(pool), device='cuda', dtype=torch.long)
    all_vocab = torch.arange(raw.shape[0], device='cuda')
    results, weights = [], {}
    for width in (9, 22):
        torch.manual_seed(SEED)
        projection = nn.Parameter(projection_for(raw.shape[1], width, 29071).cuda())
        bridge = Bridge(width, raw.shape[1]).cuda()
        optimizer = torch.optim.AdamW([{'params': bridge.parameters(), 'lr': .0003},
                                       {'params': [projection], 'lr': .0001}],
                                      weight_decay=.0001, foreach=False)
        generator = torch.Generator(device='cpu').manual_seed(SEED)
        trace = []
        for step in range(STEPS):
            sample = torch.randperm(len(train_list), generator=generator)[:BATCH].to('cuda')
            ids = train.index_select(0, sample)
            target = embeddings.index_select(0, ids)
            code = F.normalize(raw.index_select(0, ids) @ projection, dim=-1)
            prediction = bridge(code)
            contrastive = F.cross_entropy((prediction @ target.T) * 20.0,
                                          torch.arange(BATCH, device='cuda'))
            loss = contrastive + (1.0 - (prediction * target).sum(-1).mean())
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([projection, *bridge.parameters()], 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step in (0, 249, 499, 749, STEPS - 1):
                trace.append({'step': step + 1, 'loss': float(loss.detach())})
                print(json.dumps({'width': width, **trace[-1]}), flush=True)
        bridge.eval()
        with torch.no_grad():
            codebook = F.normalize(raw @ projection, dim=-1)
        row = {'width': width, 'trace': trace,
               'train_cjk': metrics(bridge, codebook, embeddings, train[:256], candidates),
               'held_cjk': metrics(bridge, codebook, embeddings, held, candidates),
               'held_full_vocab': metrics(bridge, codebook, embeddings, held[:64], all_vocab),
               'projection_delta_norm': float((projection.detach().cpu() -
                                               projection_for(raw.shape[1], width, 29071)).norm())}
        results.append(row)
        weights[str(width)] = {'bridge': {k: v.detach().cpu() for k, v in bridge.state_dict().items()},
                               'projection': projection.detach().cpu()}
        print(json.dumps({'width': width, 'train_cjk': row['train_cjk'],
                          'held_cjk': row['held_cjk'], 'held_full_vocab': row['held_full_vocab']}), flush=True)
        del bridge, optimizer, projection, codebook
    if sha(path) != WEIGHT_SHA:
        raise ValueError('Base model changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    torch.save({'weights': weights, 'base_sha256': WEIGHT_SHA,
                'source_report_sha256': sha(source),
                'train_ids_sha256': hashlib.sha256(json.dumps(train_list).encode()).hexdigest()}, OUTPUT / 'bridge.pt')
    result = {'status': 'isolated_joint_projection_no_deployment', 'results': results,
              'base_sha256': WEIGHT_SHA, 'source_report_sha256': sha(source), 'bridge_sha256': sha(OUTPUT / 'bridge.pt'),
              'train_count': len(train_list), 'held_count': len(held_list), 'steps_per_width': STEPS,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Training changes encoder and decoder jointly; old encoded records are not compatible.',
                              'Held token IDs were used in earlier development experiments.',
                              'Reconstructs frozen embeddings, not language generation or memory matrix outputs.',
                              'Only 64 held tokens are checked against full vocabulary.',
                              '22-dimensional code uses more capacity than the existing fact slot.',
                              'One training seed, not enough to attribute differences to the projection alone.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
