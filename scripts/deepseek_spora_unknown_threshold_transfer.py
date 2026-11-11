"""Test previously selected thresholds on new synthetic states and perturbations."""
import json
import random
import tempfile
import time
from pathlib import Path

from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts import deepseek_spora_continuous_text_capacity as data

OUTPUT = p.ROOT / 'runs/deepseek-spora-unknown-threshold-transfer-20261112.json'
CALIBRATION = p.ROOT / 'runs/deepseek-spora-unknown-noise-calibration-analysis-20261112.json'
DATA_SEED = 37987


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    selected = json.loads(CALIBRATION.read_text(encoding='utf-8'))
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    if selected['checkpoint_sha256'] != checkpoint_sha or selected['base_sha256'] != p.WEIGHT_SHA:
        raise ValueError('Calibration provenance mismatch')
    thresholds = {(r['seed'], r['site']): r['threshold'] for r in selected['rows']
                  if r['policy'] == 'clean_midpoint' and r['noise_level'] == 0.0}
    if len(thresholds) != 4:
        raise ValueError('Incomplete frozen thresholds')
    p.torch.set_num_threads(2)
    p.torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    data.DATA_SEED = DATA_SEED
    examples, _ = data.sampled_sequences(tokenizer)
    rng = random.Random(DATA_SEED)
    keys = list(range(64))
    rng.shuffle(keys)
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        p.runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        if any(x.is_meta for x in list(chat.base.parameters()) + list(chat.base.buffers())):
            raise ValueError('Unmaterialized layer')
        organism = chat.organism.eval()
        original = p.frozen_digest(organism, chat.state)
        device = next(organism.parameters()).device
        width = organism.config.memory_width // data.SLOTS
        embedding = chat.base.model.embed_tokens.weight.detach()
        mode = p.torch.zeros(1, dtype=p.torch.long, device=device)
        gate = p.torch.ones(1, device=device)
        for seed in data.PROJECTION_SEEDS:
            generator = p.torch.Generator(device='cpu').manual_seed(seed)
            projection = p.torch.randn((embedding.shape[1], width), generator=generator).to(device)
            with p.torch.no_grad():
                codebook = p.torch.nn.functional.normalize(embedding.float() @ projection, dim=-1)
            for size in (8, 32, 48):
                fast, velocity = organism.initial_state(1)
                for index in range(len(fast)):
                    fast[index][0] = chat.state.current[index].detach().clone()
                targets = dict(zip(keys[:size], examples[:size]))
                for key, item in targets.items():
                    value = p.torch.zeros((1, organism.config.memory_width), device=device)
                    for slot, token in enumerate(item['token_ids'] + [tokenizer.eos_token_id]):
                        value[0, slot * width:(slot + 1) * width] = codebook[token]
                    for index, site in enumerate(organism.memories):
                        fast[index], velocity[index] = site.write_temporal(fast[index], velocity[index],
                            site.current_basis[key:key + 1].to(device), value, gate, mode, False)
                for index, site in enumerate(organism.memories):
                    basis = site.current_basis[:64].to(device)
                    matrix = fast[index][0].detach().expand(64, -1, -1)
                    noise_gen = p.torch.Generator(device='cpu').manual_seed(98512 + index)
                    noise = p.torch.nn.functional.normalize(p.torch.randn(basis.shape, generator=noise_gen).to(device), dim=-1)
                    for kind, amount in (('random', 0.0), ('random', 0.1), ('random', 0.3), ('neighbor', 0.25), ('neighbor', 0.5)):
                        if kind == 'random':
                            addresses = p.torch.nn.functional.normalize(basis + amount * noise, dim=-1)
                        else:
                            addresses = p.torch.nn.functional.normalize((1 - amount) * basis + amount * basis.roll(-1, 0), dim=-1)
                        with p.torch.no_grad():
                            vectors = p.torch.bmm(matrix, addresses.unsqueeze(-1)).squeeze(-1)
                            norms = vectors[:, :width].norm(dim=-1)
                            scores = p.torch.nn.functional.normalize(vectors[:, :data.SLOTS * width].reshape(-1, width), dim=-1) @ codebook.T
                            tokens = scores.argmax(-1).reshape(64, data.SLOTS).tolist()
                        threshold = thresholds[(seed, index)]
                        false_known = sum(float(norms[k]) >= threshold for k in range(64) if k not in targets)
                        false_unknown = sum(float(norms[k]) < threshold for k in targets)
                        wrong_accepted = 0
                        for key, target in targets.items():
                            ids = tokens[key]
                            if tokenizer.eos_token_id in ids:
                                ids = ids[:ids.index(tokenizer.eos_token_id) + 1]
                            if float(norms[key]) >= threshold and ids != target['token_ids'] + [tokenizer.eos_token_id]:
                                wrong_accepted += 1
                        row = {'seed': seed, 'site': index, 'size': size, 'kind': kind, 'amount': amount,
                               'threshold': threshold, 'known': size, 'unwritten': 64 - size,
                               'false_known': false_known, 'false_unknown': false_unknown,
                               'wrong_accepted': wrong_accepted}
                        rows.append(row)
                        print(json.dumps(row), flush=True)
        if p.frozen_digest(organism, chat.state) != original:
            raise ValueError('Original state changed')
    if p.sha(p.runtime.CHECKPOINT) != checkpoint_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected input changed')
    result = {'rows': rows, 'data_seed': DATA_SEED, 'noise_seed': 98512, 'checkpoint_sha256': checkpoint_sha,
              'base_sha256': p.WEIGHT_SHA, 'seconds': time.perf_counter() - start,
              'peak_allocated_bytes': p.torch.cuda.max_memory_allocated(),
              'limitations': ['Same individual and projection seeds used in calibration; not cross-individual validation.',
                              'Different randomized text, key allocation, perturbation directions and occupancy.',
                              'Unwritten in this experiment is not guaranteed absent in original individual.',
                              'Neighbor mixing is at read only and forced, not observed router behavior.',
                              'Fixed codebook nearest neighbor decoding, no natural language or persistence.',
                              'No threshold deployed and no individual modified.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
