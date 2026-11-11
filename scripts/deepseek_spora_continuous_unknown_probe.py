"""Inspect unknown-address decoding and norm separation without a deployment gate."""
import json
import tempfile
import time
from pathlib import Path

from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_spora_continuous_text_capacity import sampled_sequences, SLOTS, PROJECTION_SEEDS

OUTPUT = p.ROOT / 'runs/deepseek-spora-continuous-unknown-probe-20261112.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    p.torch.set_num_threads(2)
    p.torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    examples, _ = sampled_sequences(tokenizer)
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        p.runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        if any(x.is_meta for x in list(chat.base.parameters()) + list(chat.base.buffers())):
            raise ValueError('Unmaterialized layer')
        organism = chat.organism.eval()
        original = p.frozen_digest(organism, chat.state)
        device = next(organism.parameters()).device
        width = organism.config.memory_width // SLOTS
        embedding = chat.base.model.embed_tokens.weight.detach()
        mode = p.torch.zeros(1, dtype=p.torch.long, device=device)
        gate = p.torch.ones(1, device=device)
        for seed in PROJECTION_SEEDS:
            generator = p.torch.Generator(device='cpu').manual_seed(seed)
            projection = p.torch.randn((embedding.shape[1], width), generator=generator).to(device)
            with p.torch.no_grad():
                codebook = p.torch.nn.functional.normalize(embedding.float() @ projection, dim=-1)
            fast, velocity = organism.initial_state(1)
            for index in range(len(fast)):
                fast[index][0] = p.torch.zeros_like(fast[index][0])
            for key, item in enumerate(examples[:32]):
                target = p.torch.zeros((1, organism.config.memory_width), device=device)
                for slot, token in enumerate(item['token_ids'] + [tokenizer.eos_token_id]):
                    target[0, slot * width:(slot + 1) * width] = codebook[token]
                for index, site in enumerate(organism.memories):
                    fast[index], velocity[index] = site.write_temporal(
                        fast[index], velocity[index], site.current_basis[key:key + 1].to(device),
                        target, gate, mode, False)
            for index, site in enumerate(organism.memories):
                basis = site.current_basis[:64].to(device)
                matrix = fast[index][0].detach().expand(64, -1, -1)
                with p.torch.no_grad():
                    vectors = p.torch.bmm(matrix, basis.unsqueeze(-1)).squeeze(-1)
                    first = vectors[:, :width]
                    norms = first.norm(dim=-1)
                    scores = p.torch.nn.functional.normalize(first, dim=-1) @ codebook.T
                    top = scores.topk(2, dim=-1)
                    for key in range(64):
                        token = int(top.indices[key, 0])
                        rows.append({'seed': seed, 'site': index, 'key': key, 'known': key < 32,
                                     'norm': float(norms[key]), 'top_token': token,
                                     'top_text': tokenizer.decode([token], skip_special_tokens=False),
                                     'top_score': float(top.values[key, 0]),
                                     'margin': float(top.values[key, 0] - top.values[key, 1])})
            print(json.dumps({'projection_seed': seed, 'queries': 128}), flush=True)
        if p.frozen_digest(organism, chat.state) != original:
            raise ValueError('Original individual changed')
    if p.sha(p.runtime.CHECKPOINT) != checkpoint_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected source changed')
    known = [r for r in rows if r['known']]
    unknown = [r for r in rows if not r['known']]
    summary = {'known': len(known), 'unknown': len(unknown),
               'minimum_known_norm': min(r['norm'] for r in known),
               'maximum_unknown_norm': max(r['norm'] for r in unknown),
               'minimum_unknown_top_score': min(r['top_score'] for r in unknown),
               'maximum_unknown_top_score': max(r['top_score'] for r in unknown)}
    result = {'summary': summary, 'rows': rows, 'checkpoint_sha256': checkpoint_sha,
              'base_sha256': p.WEIGHT_SHA, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': p.torch.cuda.max_memory_allocated(),
              'limitations': ['Zero initial matrices and orthogonal addresses create favorable known/unknown separation.',
                              'Two projections, one fixed random token sample and a single occupancy level.',
                              'Top token is only a forced codebook prediction; it is not a generated response.',
                              'No calibrated threshold, real-world unknown routing, temporal memory or persistence.',
                              'No real individual or default model modified.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
