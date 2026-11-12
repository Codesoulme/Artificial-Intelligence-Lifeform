"""Compose native codec identity recovery with a learned fact suffix.

This isolates the boundary between exact identity serialization and learned
language reconstruction. It is a research-only candidate and never updates
the formal individual or the source checkpoint.
"""
import json
import random
import tempfile
import time
from pathlib import Path

from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_spora_text_reconstruction_pilot import (
    MemoryTextReconstructor, MLPTokenCodeDecoder, WORDS, NAMES, fit_mlp_bridge,
)
from spora.continuous_text_memory import ContinuousTextCodec


OUT = p.ROOT / 'runs/continuous-reader-identity-codec-composition-pilot-20261113'
SEED = 99120
BRIDGE_STEPS = 800
DECODER_STEPS = 1000
FACT_WORDS = WORDS[:4]


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch = p.torch
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    source_sha = p.sha(p.runtime.CHECKPOINT)
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    with tempfile.TemporaryDirectory() as temp:
        p.runtime.WORKING_CHECKPOINT = Path(temp) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        model = chat.organism.eval()
        device = next(model.parameters()).device
        embedding = chat.base.model.embed_tokens.weight.detach()
        codec = ContinuousTextCodec(embedding.cpu(), memory_width=model.config.memory_width,
                                    slot_count=5, seed=29071,
                                    eos_token_id=tokenizer.eos_token_id).to(device)
        bridge = MLPTokenCodeDecoder(codec.slot_width, embedding.shape[1]).to(device)
        bridge_trace = fit_mlp_bridge(bridge, codec, embedding, BRIDGE_STEPS)
        bridge.eval().requires_grad_(False)
        suffix = MemoryTextReconstructor(embedding, model.config.memory_width, 5, bridge).to(device)

        def memory_for(key, value):
            fast, velocity = model.initial_state(1)
            for site in fast:
                site[0] = torch.zeros_like(site[0])
            name_ids = tokenizer.encode(NAMES[key], add_special_tokens=False)
            fact_ids = tokenizer.encode(WORDS[value], add_special_tokens=False)
            ids = name_ids + fact_ids
            encoded = codec.encode(torch.tensor(ids, dtype=torch.long, device=device))
            target = torch.zeros((1, model.config.memory_width), device=device)
            target[0, :encoded.numel()] = encoded
            gate = torch.ones(1, device=device)
            mode = torch.zeros(1, dtype=torch.long, device=device)
            with torch.no_grad():
                for index, site in enumerate(model.memories):
                    basis = site.current_basis[key:key + 1].to(device)
                    fast[index], velocity[index] = site.write_temporal(
                        fast[index], velocity[index], basis, target, gate, mode, False)
            reads = []
            for index, site in enumerate(model.memories):
                basis = site.current_basis[key:key + 1].to(device)
                reads.append(site.retrieve([fast[index][0]], basis))
            return ((reads[0] + reads[1]) * 0.5).detach(), name_ids, ids

        cases = []
        for key in range(3):
            for value, word in enumerate(FACT_WORDS):
                memory, name_ids, full_ids = memory_for(key, value)
                answer = f'喜欢的水果是{word}。'
                target = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
                cases.append({'group': 'train', 'key': key, 'value': value,
                              'memory': memory, 'name_ids': name_ids,
                              'full_ids': full_ids,
                              'target': torch.tensor([target], device=device)})
        held = []
        for value, word in enumerate(FACT_WORDS):
            memory, name_ids, full_ids = memory_for(3, value)
            answer = f'喜欢的水果是{word}。'
            target = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
            held.append({'group': 'held_identity', 'key': 3, 'value': value,
                         'memory': memory, 'name_ids': name_ids,
                         'full_ids': full_ids,
                         'target': torch.tensor([target], device=device)})

        optimizer = torch.optim.AdamW(suffix.parameters(), lr=2e-3, weight_decay=1e-4, foreach=False)
        order = list(range(len(cases)))
        trace = []
        for step in range(DECODER_STEPS):
            if step % len(order) == 0:
                random.shuffle(order)
            item = cases[order[step % len(order)]]
            optimizer.zero_grad(set_to_none=True)
            logits = suffix(item['memory'], item['target']).float()
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                     item['target'].reshape(-1))
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(suffix.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step in (0, 49, 199, 499, DECODER_STEPS - 1):
                trace.append({'step': step + 1, 'loss': float(loss.detach()), 'grad_norm': float(norm)})

        def evaluate(rows):
            output = []
            for item in rows:
                with torch.inference_mode():
                    identity = codec.decode(item['memory'][0, :].detach())['token_ids']
                    identity = identity[:len(item['name_ids'])]
                    suffix_ids = suffix.generate(item['memory'], eos_id=tokenizer.eos_token_id,
                                                  max_new_tokens=24)[0].tolist()
                answer = tokenizer.decode(identity + suffix_ids, skip_special_tokens=True).strip()
                expected = f'{NAMES[item["key"]]}喜欢的水果是{WORDS[item["value"]]}。'
                output.append({'group': item['group'], 'expected': expected, 'answer': answer,
                               'identity_expected_ids': item['name_ids'],
                               'identity_decoded_ids': identity,
                               'identity_exact': identity == item['name_ids'],
                               'fact': WORDS[item['value']],
                               'contains_fact': WORDS[item['value']] in answer,
                               'exact': answer == expected})
            return output

        train_after = evaluate(cases)
        held_after = evaluate(held)
    if p.sha(p.runtime.CHECKPOINT) != source_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected source changed')
    report = {
        'status': 'identity_codec_composition_pilot',
        'train': train_after,
        'held': held_after,
        'summary': {
            'train_identity_exact': sum(row['identity_exact'] for row in train_after),
            'held_identity_exact': sum(row['identity_exact'] for row in held_after),
            'train_fact': sum(row['contains_fact'] for row in train_after),
            'held_fact': sum(row['contains_fact'] for row in held_after),
            'train_exact': sum(row['exact'] for row in train_after),
            'held_exact': sum(row['exact'] for row in held_after),
        },
        'bridge_trace': bridge_trace,
        'decoder_trace': trace,
        'source_checkpoint_sha256': source_sha,
        'base_sha256': p.WEIGHT_SHA,
        'seconds': time.perf_counter() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'limitations': [
            'Explicit numeric addresses and manually constructed identity/fact sequences.',
            'Identity boundary length is supplied by the audit rather than inferred.',
            'The fact suffix decoder is question-independent and not connected to runtime generation.',
            'No open-domain routing, natural-language writes, or production adoption.',
        ],
    }
    OUT.mkdir(parents=False, exist_ok=False)
    torch.save({'format': 'spora-identity-codec-composition-v1',
                'bridge': {key: value.detach().cpu() for key, value in bridge.state_dict().items()},
                'suffix': {key: value.detach().cpu() for key, value in suffix.state_dict().items()},
                'source_sha256': source_sha, 'base_sha256': p.WEIGHT_SHA}, OUT / 'candidate.pt')
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
