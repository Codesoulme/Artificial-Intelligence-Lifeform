"""Capacity and correction probe for isolated continuous text fast matrices."""
import json
import random
import re
import tempfile
import time
from pathlib import Path

from scripts import deepseek_spora_long_memory_generate_parity as p

OUTPUT = p.ROOT / 'runs/deepseek-spora-continuous-text-capacity-20261112.json'
SLOTS = 4
PROJECTION_SEEDS = (29071, 29072)
DATA_SEED = 17041


def sampled_sequences(tokenizer):
    rng = random.Random(DATA_SEED)
    ids = list(range(tokenizer.vocab_size))
    rng.shuffle(ids)
    pool = []
    for token in ids:
        text = tokenizer.decode([token], skip_special_tokens=False)
        if (re.fullmatch(r'[\u4e00-\u9fff]{1,2}', text) and
                tokenizer.encode(text, add_special_tokens=False) == [token]):
            pool.append(token)
            if len(pool) == 300:
                break
    if len(pool) < 100:
        raise ValueError('Insufficient round-trip CJK tokens')
    results, seen = [], set()
    for attempt in range(10000):
        length = 1 + (len(results) % 3)
        tokens = tuple(rng.sample(pool, length))
        text = tokenizer.decode(tokens, skip_special_tokens=False)
        if tokens not in seen and tokenizer.encode(text, add_special_tokens=False) == list(tokens):
            results.append({'text': text, 'token_ids': list(tokens)})
            seen.add(tokens)
            if len(results) == 65:
                return results, len(pool)
    raise ValueError('Could not produce enough round-trip token sequences')


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    p.torch.set_num_threads(2)
    p.torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    examples, pool_size = sampled_sequences(tokenizer)
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        p.runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        if any(x.is_meta for x in list(chat.base.parameters()) + list(chat.base.buffers())):
            raise ValueError('Unmaterialized layer')
        organism = chat.organism.eval()
        original = p.frozen_digest(organism, chat.state)
        m, count = organism.config.memory_width, organism.config.key_count
        if count < 64:
            raise ValueError('Insufficient distinct addresses')
        width = m // SLOTS
        embedding = chat.base.model.embed_tokens.weight.detach()
        device = next(organism.parameters()).device
        mode = p.torch.zeros(1, dtype=p.torch.long, device=device)
        gate = p.torch.ones(1, device=device)
        for seed in PROJECTION_SEEDS:
            generator = p.torch.Generator(device='cpu').manual_seed(seed)
            projection = p.torch.randn((embedding.shape[1], width), generator=generator).to(device)
            with p.torch.no_grad():
                codebook = p.torch.nn.functional.normalize(embedding.float() @ projection, dim=-1)
            if not bool(p.torch.isfinite(codebook).all()):
                raise ValueError('Invalid codebook')

            def target(item):
                ids = item['token_ids'] + [tokenizer.eos_token_id]
                vec = p.torch.zeros((1, m), device=device)
                for slot, token in enumerate(ids):
                    vec[0, slot * width:(slot + 1) * width] = codebook[token]
                return vec

            def write(fast, velocity, key, item):
                value = target(item)
                for site_index, site in enumerate(organism.memories):
                    fast[site_index], velocity[site_index] = site.write_temporal(
                        fast[site_index], velocity[site_index], site.current_basis[key:key + 1].to(device),
                        value, gate, mode, False)

            def measure(fast, expected, group, write_seconds):
                p.torch.cuda.synchronize()
                begin = time.perf_counter()
                exact_per_site, per_item = [0] * len(organism.memories), [True] * len(expected)
                failures, min_margin = [], float('inf')
                with p.torch.no_grad():
                    for site_index, site in enumerate(organism.memories):
                        for start in range(0, len(expected), 16):
                            subset = expected[start:start + 16]
                            addresses = site.current_basis[start:start + len(subset)].to(device)
                            matrix = fast[site_index][0].expand(len(subset), -1, -1)
                            vectors = p.torch.bmm(matrix, addresses.unsqueeze(-1)).squeeze(-1)
                            segments = vectors[:, :SLOTS * width].reshape(-1, width)
                            scores = p.torch.nn.functional.normalize(segments, dim=-1) @ codebook.T
                            best = scores.topk(2, dim=-1)
                            tokens = best.indices[:, 0].reshape(len(subset), SLOTS).tolist()
                            margins = (best.values[:, 0] - best.values[:, 1]).reshape(len(subset), SLOTS)
                            for j, item in enumerate(subset):
                                predicted = tokens[j]
                                if tokenizer.eos_token_id in predicted:
                                    predicted = predicted[:predicted.index(tokenizer.eos_token_id) + 1]
                                target_ids = item['token_ids'] + [tokenizer.eos_token_id]
                                ok = predicted == target_ids
                                exact_per_site[site_index] += int(ok)
                                per_item[start + j] &= ok
                                min_margin = min(min_margin, float(margins[j, :len(target_ids)].min()))
                                if not ok:
                                    failures.append({'site': site_index, 'key': start + j,
                                                     'expected': item['text'], 'expected_ids': target_ids,
                                                     'decoded': tokenizer.decode(predicted, skip_special_tokens=True),
                                                     'predicted_ids': predicted})
                p.torch.cuda.synchronize()
                elapsed = time.perf_counter() - begin
                unknown_norm = None
                if len(expected) < count:
                    address = organism.memories[0].current_basis[len(expected):len(expected) + 1].to(device)
                    unknown_norm = float(organism.memories[0].retrieve([fast[0][0]], address).norm())
                row = {'seed': seed, 'group': group, 'addresses': len(expected), 'exact_per_site': exact_per_site,
                       'exact_both_sites': sum(per_item), 'failures': failures, 'minimum_token_margin': min_margin,
                       'write_seconds': write_seconds, 'read_seconds': elapsed,
                       'unwritten_address_norm_site0': unknown_norm}
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != 'failures'}), flush=True)

            for size in (8, 32, 64):
                fast, velocity = organism.initial_state(1)
                for i in range(len(fast)):
                    fast[i][0] = p.torch.zeros_like(fast[i][0])
                p.torch.cuda.synchronize()
                begin = time.perf_counter()
                for key, item in enumerate(examples[:size]):
                    write(fast, velocity, key, item)
                p.torch.cuda.synchronize()
                write_seconds = time.perf_counter() - begin
                measure(fast, examples[:size], 'written', write_seconds)
                if size == 64:
                    p.torch.cuda.synchronize()
                    begin = time.perf_counter()
                    write(fast, velocity, 0, examples[64])
                    p.torch.cuda.synchronize()
                    measure(fast, [examples[64]] + examples[1:64], 'one_correction', time.perf_counter() - begin)
        if p.frozen_digest(organism, chat.state) != original:
            raise ValueError('Original individual changed')
    if p.sha(p.runtime.CHECKPOINT) != checkpoint_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected input changed')
    result = {'rows': rows, 'examples': examples, 'projection_seeds': PROJECTION_SEEDS,
              'data_seed': DATA_SEED, 'token_pool': pool_size, 'slot_width': width,
              'checkpoint_sha256': checkpoint_sha, 'base_sha256': p.WEIGHT_SHA,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': p.torch.cuda.max_memory_allocated(),
              'limitations': ['Random CJK token combinations, not natural sentences or dialogue.',
                              'Single-token vocabulary pool joined to 1-3 token sequences; fixed four slots.',
                              '64 known orthogonal numeric addresses, no dynamic addressing or unknown-value policy.',
                              'Full-vocabulary nearest neighbor codebook from base embedding, not neural language generation.',
                              'Direct experimental fast-matrix writes bypass provenance, history and persistence.',
                              'Two projection seeds and one data seed; no real individual changed.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'completed': len(rows), 'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()
