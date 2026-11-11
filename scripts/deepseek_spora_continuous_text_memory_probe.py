"""Development probe: continuous text codes in isolated native fast matrices.

The fixed language embedding supplies a shared decoding codebook; no model
generation, automatic text write, durability or production integration here.
"""
import json
import tempfile
import time
from pathlib import Path

from scripts import deepseek_spora_long_memory_generate_parity as p

OUTPUT = p.ROOT / 'runs/deepseek-spora-continuous-text-memory-probe-20261112.json'
PHRASES = ('可乐', '青玉铃', '红苹果', '杭州', '咖啡', '月光', '小猫', '晴天')
SLOTS = 4
SEED = 29071


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    p.torch.set_num_threads(2)
    p.torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    encoded = {s: tokenizer.encode(s, add_special_tokens=False) for s in PHRASES}
    if any(not ids or len(ids) >= SLOTS or tokenizer.eos_token_id in ids for ids in encoded.values()):
        raise ValueError('Phrases must fit available slots and exclude EOS')
    if any(tokenizer.decode(ids, skip_special_tokens=True) != text for text, ids in encoded.items()):
        raise ValueError('Tokenizer did not round-trip a phrase')
    with tempfile.TemporaryDirectory() as directory:
        p.runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        if any(x.is_meta for x in list(chat.base.parameters()) + list(chat.base.buffers())):
            raise ValueError('Unmaterialized layer')
        organism = chat.organism.eval()
        original = p.frozen_digest(organism, chat.state)
        m = organism.config.memory_width
        slot_width = m // SLOTS
        if slot_width < 2:
            raise ValueError('Insufficient continuous width')
        embedding = chat.base.model.embed_tokens.weight.detach()
        generator = p.torch.Generator(device='cpu').manual_seed(SEED)
        projection = p.torch.randn((embedding.shape[1], slot_width), generator=generator, dtype=p.torch.float32)
        projection = projection.to(embedding.device)
        with p.torch.no_grad():
            codebook = p.torch.nn.functional.normalize(embedding.float() @ projection, dim=-1)
        if not bool(p.torch.isfinite(codebook).all()):
            raise ValueError('Invalid embedding projection')
        device = next(organism.parameters()).device
        if codebook.device != device:
            raise ValueError('Projection and organism devices differ')
        fast, velocity = organism.initial_state(1)
        for site in range(len(fast)):
            fast[site][0] = p.torch.zeros_like(fast[site][0])
        mode = p.torch.zeros(1, dtype=p.torch.long, device=device)
        gate = p.torch.ones(1, device=device)

        def representation(text):
            ids = encoded[text] + [tokenizer.eos_token_id]
            result = p.torch.zeros((1, m), device=device)
            for slot, token in enumerate(ids):
                result[0, slot * slot_width:(slot + 1) * slot_width] = codebook[token]
            return result

        def write(key, text):
            target = representation(text)
            for index, site in enumerate(organism.memories):
                address = site.current_basis[key:key + 1].to(device)
                fast[index], velocity[index] = site.write_temporal(
                    fast[index], velocity[index], address, target, gate, mode, False)

        def read(key, target):
            observations = []
            for index, site in enumerate(organism.memories):
                address = site.current_basis[key:key + 1].to(device)
                with p.torch.no_grad():
                    vec = site.retrieve([fast[index][0]], address)[0]
                    predictions, margins = [], []
                    for slot in range(SLOTS):
                        segment = vec[slot * slot_width:(slot + 1) * slot_width]
                        scores = codebook @ p.torch.nn.functional.normalize(segment, dim=0)
                        best = scores.topk(2)
                        token = int(best.indices[0])
                        predictions.append(token)
                        margins.append(float(best.values[0] - best.values[1]))
                        if token == tokenizer.eos_token_id:
                            break
                answer = tokenizer.decode(predictions, skip_special_tokens=True)
                observations.append({'site': index, 'answer': answer, 'token_ids': predictions,
                                     'exact_tokens_including_eos': predictions == encoded[target] + [tokenizer.eos_token_id],
                                     'top1_margins': margins})
            return observations

        rows = []
        keys = (6, 10, 22, 42)
        for offset in (0, 4):
            # Reset isolated current matrices; existing individual is never modified.
            for index in range(len(fast)):
                fast[index][0] = p.torch.zeros_like(fast[index][0])
            for key, phrase in zip(keys, PHRASES[offset:offset + 4]):
                write(key, phrase)
            for key, phrase in zip(keys, PHRASES[offset:offset + 4]):
                rows.append({'group': f'round_{offset // 4 + 1}', 'key': key, 'expected': phrase,
                             'sites': read(key, phrase)})
            replacement = PHRASES[(offset + 4) % len(PHRASES)]
            write(keys[0], replacement)
            for key, phrase in zip(keys, (replacement,) + PHRASES[offset + 1:offset + 4]):
                rows.append({'group': f'correction_{offset // 4 + 1}', 'key': key, 'expected': phrase,
                             'sites': read(key, phrase)})
            print(json.dumps({'completed_round': offset // 4 + 1}, ensure_ascii=True), flush=True)
        if p.frozen_digest(organism, chat.state) != original:
            raise ValueError('Original individual changed')
    if p.sha(p.runtime.CHECKPOINT) != checkpoint_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected input changed')
    summary = {}
    for group in ('round_1', 'correction_1', 'round_2', 'correction_2'):
        selected = [row for row in rows if row['group'] == group]
        summary[group] = {'queries': len(selected),
                          'exact_both_sites': sum(all(site['exact_tokens_including_eos'] for site in row['sites']) for row in selected),
                          'exact_site_reads': sum(site['exact_tokens_including_eos'] for row in selected for site in row['sites'])}
    result = {'summary': summary, 'rows': rows, 'tokenized_inputs': encoded, 'seed': SEED,
              'slots': SLOTS, 'slot_width': slot_width, 'memory_width': m,
              'checkpoint_sha256': checkpoint_sha, 'base_sha256': p.WEIGHT_SHA,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': p.torch.cuda.max_memory_allocated(),
              'limitations': ['Eight authored phrases, at most three tokens plus EOS, four numeric addresses.',
                              'Development probe bypasses normal write controls, event history and persistence.',
                              'Random shared projection plus base vocabulary embedding is an explicit external codebook relative to the fast matrices.',
                              'Full vocabulary nearest-neighbor decoding, not language-model generation or automatic routing.',
                              'No unknown-memory detector, scalability benchmark, or independent generalization test.',
                              'No production individual, default reader, or checkpoint changed.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
