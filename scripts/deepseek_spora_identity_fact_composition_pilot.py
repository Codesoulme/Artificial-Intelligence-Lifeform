"""Compose an independently recovered identity with a learned fact suffix decoder."""
import json
import random
import tempfile
import time
from pathlib import Path

from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_spora_text_reconstruction_pilot import (
    MemoryTextReconstructor, MLPTokenCodeDecoder, fit_mlp_bridge, WORDS, NAMES,
)
from spora.continuous_text_memory import ContinuousTextCodec


OUT = p.ROOT / 'runs/continuous-reader-identity-fact-composition-pilot-20261113'
SEED = 99117
BRIDGE_STEPS = 800
DECODER_STEPS = 1000


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
        # The suffix head receives fact slots only; identity is recovered by the
        # explicit codec diagnostic and prepended after generation.
        suffix = MemoryTextReconstructor(embedding, model.config.memory_width, 5, bridge).to(device)
        cases = []

        def memory_for(key, value):
            fast, velocity = model.initial_state(1)
            for site in fast:
                site[0] = torch.zeros_like(site[0])
            identity_id = tokenizer.encode(NAMES[key], add_special_tokens=False)[-1]
            fact_ids = tokenizer.encode(WORDS[value], add_special_tokens=False)
            encoded = codec.encode(torch.tensor([identity_id, *fact_ids], device=device))
            target = torch.zeros((1, model.config.memory_width), device=device)
            target[0, :encoded.numel()] = encoded
            with torch.no_grad():
                for index, site in enumerate(model.memories):
                    fast[index], velocity[index] = site.write_temporal(
                        fast[index], velocity[index], site.current_basis[key:key + 1].to(device),
                        target, torch.ones(1, device=device),
                        torch.zeros(1, dtype=torch.long, device=device), False)
            reads = [site.retrieve([fast[index][0]], site.current_basis[key:key + 1].to(device))
                     for index, site in enumerate(model.memories)]
            return ((reads[0] + reads[1]) * 0.5).detach()

        for key in range(3):
            for value, word in enumerate(WORDS):
                memory = memory_for(key, value)
                suffix_text = f'喜欢的水果是{word}。'
                target = tokenizer.encode(suffix_text, add_special_tokens=False) + [tokenizer.eos_token_id]
                cases.append({'group': 'train', 'key': key, 'value': value,
                              'memory': memory,
                              'target': torch.tensor([target], device=device)})
        held = []
        for value, word in enumerate(WORDS):
            held.append({'group': 'held_identity', 'key': 3, 'value': value,
                         'memory': memory_for(3, value),
                         'target': torch.tensor([tokenizer.encode(f'喜欢的水果是{word}。', add_special_tokens=False)
                                                 + [tokenizer.eos_token_id]], device=device)})
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
        vocab_embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)

        def compose(rows):
            output = []
            for item in rows:
                with torch.inference_mode():
                    suffix_ids = suffix.generate(item['memory'], eos_id=tokenizer.eos_token_id,
                                                  max_new_tokens=24)[0].tolist()
                # Identity slot is a diagnostic codebook read, explicitly kept
                # separate from the learned suffix decoder.
                first_read = item['memory'][0, :codec.slot_width]
                identity_vector = bridge(first_read.reshape(1, 1, -1)).float()[0, 0]
                scores = vocab_embedding @ torch.nn.functional.normalize(identity_vector, dim=-1)
                identity_id = int(scores.argmax())
                ids = [identity_id] + suffix_ids
                text = tokenizer.decode(ids, skip_special_tokens=True).strip()
                expected = f'{NAMES[item["key"]]}喜欢的水果是{WORDS[item["value"]]}。'
                output.append({'group': item['group'], 'expected': expected, 'answer': text,
                               'identity_expected': NAMES[item['key']],
                               'identity_decoded': tokenizer.decode([identity_id]),
                               'fact': WORDS[item['value']],
                               'contains_fact': WORDS[item['value']] in text,
                               'exact': text == expected})
            return output
        train_after = compose(cases)
        held_after = compose(held)
    if p.sha(p.runtime.CHECKPOINT) != source_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected source changed')
    OUT.mkdir(parents=False, exist_ok=False)
    candidate = OUT / 'candidate.pt'
    torch.save({'format': 'spora-identity-fact-composition-v1',
                'bridge': {key: value.detach().cpu() for key, value in bridge.state_dict().items()},
                'suffix': {key: value.detach().cpu() for key, value in suffix.state_dict().items()},
                'source_sha256': source_sha, 'base_sha256': p.WEIGHT_SHA}, candidate)
    report = {'status': 'identity_fact_composition_diagnostic', 'train': train_after,
              'held': held_after, 'train_exact': sum(x['exact'] for x in train_after),
              'held_exact': sum(x['exact'] for x in held_after),
              'held_fact': sum(x['contains_fact'] for x in held_after),
              'bridge_trace': bridge_trace, 'decoder_trace': trace,
              'source_sha256': source_sha, 'base_sha256': p.WEIGHT_SHA,
              'candidate_sha256': p.sha(candidate), 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Identity uses an explicit nearest-token diagnostic read.',
                              'Fact suffix is learned but question-independent.',
                              'Numeric addresses and controlled writes.',
                              'Not an adopted individual or open-domain memory path.']}
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'seconds': report['seconds'], 'train_exact': report['train_exact'],
                      'held_exact': report['held_exact'], 'held_fact': report['held_fact']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
