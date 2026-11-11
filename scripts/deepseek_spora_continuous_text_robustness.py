"""Probe retrieval perturbations for isolated continuous text memory matrices."""
import json
import tempfile
import time
from pathlib import Path

from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_spora_continuous_text_capacity import sampled_sequences, SLOTS, PROJECTION_SEEDS

OUTPUT = p.ROOT / 'runs/deepseek-spora-continuous-text-robustness-20261112.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    p.torch.set_num_threads(2)
    p.torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    examples, _ = sampled_sequences(tokenizer)
    examples = examples[:64]
    conditions = [('baseline', 0.0), ('address_noise', 0.01), ('address_noise', 0.03),
                  ('address_noise', 0.1), ('address_noise', 0.3), ('neighbor_mix', 0.1),
                  ('neighbor_mix', 0.25), ('neighbor_mix', 0.5), ('matrix_noise', 0.01), ('matrix_noise', 0.1)]
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        p.runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        if any(x.is_meta for x in list(chat.base.parameters()) + list(chat.base.buffers())):
            raise ValueError('Unmaterialized layer')
        organism = chat.organism.eval()
        original = p.frozen_digest(organism, chat.state)
        width = organism.config.memory_width // SLOTS
        dimension = organism.config.memory_width
        device = next(organism.parameters()).device
        embedding = chat.base.model.embed_tokens.weight.detach()
        mode = p.torch.zeros(1, dtype=p.torch.long, device=device)
        gate = p.torch.ones(1, device=device)
        for seed in PROJECTION_SEEDS:
            generator = p.torch.Generator(device='cpu').manual_seed(seed)
            projection = p.torch.randn((embedding.shape[1], width), generator=generator).to(device)
            with p.torch.no_grad():
                codebook = p.torch.nn.functional.normalize(embedding.float() @ projection, dim=-1)
            if not bool(p.torch.isfinite(codebook).all()):
                raise ValueError('Invalid codebook')
            fast, velocity = organism.initial_state(1)
            for i in range(len(fast)):
                fast[i][0] = p.torch.zeros_like(fast[i][0])
            for key, example in enumerate(examples):
                target = p.torch.zeros((1, dimension), device=device)
                for slot, token in enumerate(example['token_ids'] + [tokenizer.eos_token_id]):
                    target[0, slot * width:(slot + 1) * width] = codebook[token]
                for index, site in enumerate(organism.memories):
                    fast[index], velocity[index] = site.write_temporal(fast[index], velocity[index],
                        site.current_basis[key:key + 1].to(device), target, gate, mode, False)
            for kind, strength in conditions:
                exact = [0] * len(organism.memories)
                both = [True] * len(examples)
                cosines, failures = [], []
                with p.torch.no_grad():
                    for index, site in enumerate(organism.memories):
                        basis = site.current_basis[:64].to(device)
                        generator = p.torch.Generator(device='cpu').manual_seed(8000 + index)
                        direction = p.torch.randn(basis.shape, generator=generator).to(device)
                        direction = p.torch.nn.functional.normalize(direction, dim=-1)
                        if kind == 'address_noise':
                            addresses = p.torch.nn.functional.normalize(basis + strength * direction, dim=-1)
                        elif kind == 'neighbor_mix':
                            addresses = p.torch.nn.functional.normalize((1 - strength) * basis + strength * basis.roll(-1, 0), dim=-1)
                        else:
                            addresses = basis
                        cosines.extend((basis * addresses).sum(-1).tolist())
                        matrix = fast[index][0].detach()
                        if kind == 'matrix_noise':
                            noise = p.torch.randn(matrix.shape, generator=generator).to(device)
                            matrix = matrix + strength * matrix.square().mean().sqrt() * noise
                        for start in range(0, len(examples), 16):
                            subset = examples[start:start + 16]
                            vectors = p.torch.bmm(matrix.expand(len(subset), -1, -1),
                                                 addresses[start:start + len(subset)].unsqueeze(-1)).squeeze(-1)
                            segments = vectors[:, :SLOTS * width].reshape(-1, width)
                            scores = p.torch.nn.functional.normalize(segments, dim=-1) @ codebook.T
                            predictions = scores.argmax(-1).reshape(len(subset), SLOTS).tolist()
                            for j, example in enumerate(subset):
                                tokens = predictions[j]
                                if tokenizer.eos_token_id in tokens:
                                    tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
                                ok = tokens == example['token_ids'] + [tokenizer.eos_token_id]
                                exact[index] += int(ok)
                                both[start + j] &= ok
                                if not ok and len(failures) < 6:
                                    failures.append({'site': index, 'key': start + j,
                                                     'expected_ids': example['token_ids'] + [tokenizer.eos_token_id],
                                                     'predicted_ids': tokens})
                row = {'projection_seed': seed, 'kind': kind, 'strength': strength,
                       'exact_per_site': exact, 'exact_both_sites': sum(both), 'total': len(examples),
                       'mean_query_cosine': sum(cosines) / len(cosines), 'failure_samples': failures}
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != 'failure_samples'}), flush=True)
        if p.frozen_digest(organism, chat.state) != original:
            raise ValueError('Original individual changed')
    if p.sha(p.runtime.CHECKPOINT) != checkpoint_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected input changed')
    result = {'rows': rows, 'checkpoint_sha256': checkpoint_sha, 'base_sha256': p.WEIGHT_SHA,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': p.torch.cuda.max_memory_allocated(),
              'limitations': ['64 random CJK token sequences reused from development capacity probe.',
                              'Two projections and deterministic synthetic query/matrix perturbations.',
                              'All writes use orthogonal addresses; neighbor mix occurs only on read.',
                              'Matrix noise is Gaussian scaled to whole-matrix RMS, not a measured quantizer.',
                              'Full-vocabulary nearest-neighbor decoding; no routing, chat or persistence.',
                              'No real individual or default checkpoint changed.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'conditions': len(rows), 'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()
