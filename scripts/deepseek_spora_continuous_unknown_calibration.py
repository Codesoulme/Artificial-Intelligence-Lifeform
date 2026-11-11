"""Inspect unknown-address decoding and norm separation without a deployment gate."""
import json
import tempfile
import time
from pathlib import Path

from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_spora_continuous_text_capacity import sampled_sequences, SLOTS, PROJECTION_SEEDS

OUTPUT = p.ROOT / 'runs/deepseek-spora-continuous-unknown-calibration-20261112.json'


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
                fast[index][0] = chat.state.current[index].detach().clone()
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
                generator = p.torch.Generator(device='cpu').manual_seed(5192 + index)
                noise = p.torch.nn.functional.normalize(p.torch.randn(basis.shape, generator=generator).to(device), dim=-1)
                for noise_level in (0.0, 0.1, 0.3):
                  addresses = p.torch.nn.functional.normalize(basis + noise_level * noise, dim=-1)
                  with p.torch.no_grad():
                    vectors = p.torch.bmm(matrix, addresses.unsqueeze(-1)).squeeze(-1)
                    first = vectors[:, :width]
                    norms = first.norm(dim=-1)
                    scores = p.torch.nn.functional.normalize(first, dim=-1) @ codebook.T
                    top = scores.topk(2, dim=-1)
                    for key in range(64):
                        token = int(top.indices[key, 0])
                        rows.append({'seed': seed, 'site': index, 'key': key, 'known': key < 32,
                                     'noise_level': noise_level,
                                     'norm': float(norms[key]), 'top_token': token,
                                     'top_text': tokenizer.decode([token], skip_special_tokens=False),
                                     'top_score': float(top.values[key, 0]),
                                     'margin': float(top.values[key, 0] - top.values[key, 1])})
            print(json.dumps({'projection_seed': seed, 'queries': 128}), flush=True)
        if p.frozen_digest(organism, chat.state) != original:
            raise ValueError('Original individual changed')
    if p.sha(p.runtime.CHECKPOINT) != checkpoint_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected source changed')
    summary = []
    for seed in PROJECTION_SEEDS:
        for site in range(2):
            clean = [r for r in rows if r['seed'] == seed and r['site'] == site and r['noise_level'] == 0 and r['key'] % 2 == 0]
            thresholds = sorted({r['norm'] for r in clean})
            threshold = min(thresholds, key=lambda t: sum((r['norm'] >= t) != r['known'] for r in clean))
            for noise_level in (0.0, 0.1, 0.3):
                test = [r for r in rows if r['seed'] == seed and r['site'] == site and r['noise_level'] == noise_level and r['key'] % 2 == 1]
                summary.append({'seed': seed, 'site': site, 'noise_level': noise_level, 'threshold': threshold,
                                'false_known': sum(not r['known'] and r['norm'] >= threshold for r in test),
                                'false_unknown': sum(r['known'] and r['norm'] < threshold for r in test),
                                'known_total': sum(r['known'] for r in test), 'unknown_total': sum(not r['known'] for r in test)})
    result = {'summary': summary, 'rows': rows, 'checkpoint_sha256': checkpoint_sha,
              'base_sha256': p.WEIGHT_SHA, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': p.torch.cuda.max_memory_allocated(),
              'limitations': ['Uses a clone of the checkpoint current matrix, which may already contain unrelated old facts.',
                              'Known means written by this experiment, not verified unknown in the original individual.',
                              'Even-key clean calibration and odd-key tests share same state and random text distribution.',
                              'Synthetic perturbation, two projections and one occupancy level; no real-world router.',
                              'Top token is only a forced codebook prediction; it is not a generated response.',
                              'No deployed threshold, temporal memory or persistence.',
                              'No real individual or default model modified.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
