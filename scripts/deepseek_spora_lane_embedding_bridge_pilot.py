"""Train a slot-code to frozen-language-embedding bridge on disjoint tokens."""
import hashlib
import json
import random
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
from scripts.deepseek_spora_prefix_features import read_bf16_tensor
from scripts.deepseek_spora_structured_lane_noise_audit import PREVIOUS, projection_for

OUTPUT = ROOT / 'runs/deepseek-spora-lane-embedding-bridge-pilot-20261113'
STEPS = 1000
BATCH = 256
TRAIN_COUNT = 8192
SEED = 78621


class Bridge(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 512), nn.GELU(), nn.Linear(512, hidden))

    def forward(self, x):
        return F.normalize(self.net(x.float()), dim=-1)


@torch.no_grad()
def metrics(bridge, codebook, embeddings, queries, candidates, batch=16):
    matches, cosine, ranks = 0, [], []
    for start in range(0, len(queries), batch):
        ids = queries[start:start + batch]
        prediction = bridge(codebook.index_select(0, ids))
        sims = prediction @ embeddings.index_select(0, candidates).T
        # One shared score matrix prevents numerical mismatches in rank calculations.
        order = torch.searchsorted(candidates, ids)
        if bool((order >= len(candidates)).any()) or not torch.equal(candidates[order], ids):
            raise ValueError('Evaluation candidate missing')
        actual = sims.gather(1, order[:, None]).squeeze(1)
        higher = (sims > actual[:, None]).sum(-1)
        ties = (sims == actual[:, None]).sum(-1)
        matches += int(((higher == 0) & (ties == 1)).sum())
        ranks.extend((higher + 1).cpu().tolist())
        cosine.extend(actual.cpu().tolist())
    return {'count': len(queries), 'candidate_count': len(candidates), 'correct_unique': matches,
            'mean_cosine': sum(cosine) / len(cosine), 'mean_rank': sum(ranks) / len(ranks)}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    path = MODEL / 'model.safetensors'
    if sha(path) != WEIGHT_SHA:
        raise ValueError('Unexpected base weights')
    prior = json.loads(PREVIOUS.read_text(encoding='utf-8'))
    if prior['base_sha256'] != WEIGHT_SHA:
        raise ValueError('Prior token split mismatch')
    held_list = prior['train_ids'] + prior['held_ids']
    if len(held_list) != 320 or len(set(held_list)) != 320:
        raise ValueError('Unexpected held set')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(SEED)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    pool = []
    for token in range(tokenizer.vocab_size):
        text = tokenizer.decode([token], skip_special_tokens=False)
        if ('\u4e00' <= text[:1] <= '\u9fff'
                and tokenizer.encode(text, add_special_tokens=False) == [token]):
            pool.append(token)
    if not set(held_list).issubset(pool):
        raise ValueError('Held tokens not in stable CJK pool')
    candidates_list = sorted(pool)
    available = sorted(set(pool) - set(held_list))
    random.Random(SEED).shuffle(available)
    if len(available) < TRAIN_COUNT:
        raise ValueError('Insufficient distinct training tokens')
    train_list = available[:TRAIN_COUNT]
    if set(train_list) & set(held_list):
        raise ValueError('Token leakage')
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
    candidates = torch.tensor(candidates_list, device='cuda', dtype=torch.long)
    all_vocab = torch.arange(raw.shape[0], device='cuda')
    results, weights = [], {}
    for width in (9, 22):
        torch.manual_seed(SEED)
        codebook = F.normalize(raw @ projection_for(raw.shape[1], width, 29071).cuda(), dim=-1)
        bridge = Bridge(width, raw.shape[1]).cuda()
        optimizer = torch.optim.AdamW(bridge.parameters(), lr=0.0003, weight_decay=0.0001, foreach=False)
        generator = torch.Generator(device='cpu').manual_seed(SEED)
        trace = []
        for step in range(STEPS):
            sample = torch.randperm(len(train_list), generator=generator)[:BATCH].to('cuda')
            ids = train.index_select(0, sample)
            target = embeddings.index_select(0, ids)
            prediction = bridge(codebook.index_select(0, ids))
            contrastive = F.cross_entropy((prediction @ target.T) * 20.0,
                                          torch.arange(BATCH, device='cuda'))
            loss = contrastive + (1.0 - (prediction * target).sum(-1).mean())
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step in (0, 249, 499, 749, STEPS - 1):
                trace.append({'step': step + 1, 'loss': float(loss.detach())})
                print(json.dumps({'width': width, **trace[-1]}), flush=True)
        bridge.eval()
        row = {'width': width, 'trace': trace,
               'train_cjk': metrics(bridge, codebook, embeddings, train[:256], candidates),
               'held_cjk': metrics(bridge, codebook, embeddings, held, candidates),
               'held_full_vocab': metrics(bridge, codebook, embeddings, held[:64], all_vocab)}
        results.append(row)
        weights[str(width)] = {k: v.detach().cpu() for k, v in bridge.state_dict().items()}
        print(json.dumps({'width': width, 'train_cjk': row['train_cjk'],
                          'held_cjk': row['held_cjk'], 'held_full_vocab': row['held_full_vocab']}), flush=True)
        del bridge, optimizer, codebook
    if sha(path) != WEIGHT_SHA:
        raise ValueError('Base model changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    torch.save({'weights': weights, 'base_sha256': WEIGHT_SHA, 'seed': SEED,
                'train_ids_sha256': hashlib.sha256(json.dumps(train_list).encode()).hexdigest()}, OUTPUT / 'bridge.pt')
    result = {'status': 'isolated_embedding_bridge_no_deployment', 'results': results,
              'base_sha256': WEIGHT_SHA, 'bridge_sha256': sha(OUTPUT / 'bridge.pt'),
              'train_count': len(train_list), 'held_count': len(held_list), 'train_ids': train_list,
              'held_ids': held_list, 'seed': SEED, 'steps_per_width': STEPS,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Reconstructs frozen embeddings, not DeepSeek generation or autonomous memory.',
                              'Held token IDs exclude training but were used in prior development experiments.',
                              'CJK roundtrip pool is selected, not a distribution representative of natural conversations.',
                              'Full-vocabulary evaluation uses only first 64 held tokens.',
                              '22-dimensional codes consume more lane capacity than the present 9-dimensional fact slot.',
                              'Direct codes use shared fixed projection; no memory matrix or noisy writes in this experiment.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
