"""Train and audit a learned inverse for continuous identity slots.

The fixed codec remains the control. This pilot tests whether a learned
token classifier can replace the approximate embedding bridge without using
the language decoder or changing the formal individual.
"""
import json
import random
import tempfile
import time
from pathlib import Path

from scripts import deepseek_spora_long_memory_generate_parity as p
from spora.continuous_text_memory import ContinuousTextCodec


OUT = p.ROOT / 'runs/continuous-reader-identity-classifier-pilot-20261113'
SEED = 99119
NAMES = ('小林', '小周', '小王', '小赵')
WORDS = ('苹果', '香蕉', '西瓜', '葡萄')
SLOT_COUNT = 5
TRAIN_STEPS = 1800
BATCH = 256


def make_memory(model, codec, tokenizer, device, key, ids):
    torch = p.torch
    fast, velocity = model.initial_state(1)
    for site in fast:
        site[0] = torch.zeros_like(site[0])
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
    vectors = []
    for index, site in enumerate(model.memories):
        basis = site.current_basis[key:key + 1].to(device)
        vectors.append(site.retrieve([fast[index][0]], basis)[0])
    return torch.stack(vectors).mean(0), encoded


def decode_slots(head, vector, codec, tokenizer, expected_ids):
    torch = p.torch
    width = codec.slot_width
    tokens = []
    margins = []
    for slot in range(codec.slot_count):
        segment = vector[slot * width:(slot + 1) * width].reshape(1, -1)
        scores = head(torch.nn.functional.normalize(segment, dim=-1))[0]
        top = scores.topk(2)
        token = int(top.indices[0])
        tokens.append(token)
        margins.append(float(top.values[0] - top.values[1]))
        if token == tokenizer.eos_token_id:
            break
    expected = expected_ids + [tokenizer.eos_token_id]
    return {
        'token_ids': tokens,
        'decoded': tokenizer.decode(tokens, skip_special_tokens=True),
        'expected_ids': expected,
        'exact': tokens == expected,
        'margins': margins,
    }


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
        original = p.frozen_digest(model, chat.state)
        embedding = chat.base.model.embed_tokens.weight.detach()
        codec = ContinuousTextCodec(embedding.cpu(), memory_width=model.config.memory_width,
                                    slot_count=SLOT_COUNT, seed=29071,
                                    eos_token_id=tokenizer.eos_token_id).to(device)

        # The head is trained from scratch over the complete fixed vocabulary.
        # It learns token classification from a continuous slot, rather than
        # approximating the embedding matrix with a small regression bridge.
        head = torch.nn.Linear(codec.slot_width, codec.vocab_size, bias=False).to(device)
        torch.nn.init.normal_(head.weight, mean=0.0, std=0.02)
        optimizer = torch.optim.SGD(head.parameters(), lr=0.75, momentum=0.9,
                                    weight_decay=1e-5, foreach=False)
        generator = torch.Generator(device=device).manual_seed(SEED + 1)
        trace = []
        for step in range(TRAIN_STEPS):
            ids = torch.randint(codec.vocab_size, (BATCH,), generator=generator, device=device)
            inputs = codec.codebook.index_select(0, ids)
            logits = head(inputs)
            loss = torch.nn.functional.cross_entropy(logits, ids)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step in (0, 99, 399, 899, TRAIN_STEPS - 1):
                trace.append({'step': step + 1, 'loss': float(loss.detach()),
                              'grad_norm': float(norm)})

        names = []
        for key, name in enumerate(NAMES):
            ids = tokenizer.encode(name, add_special_tokens=False)
            vector, encoded = make_memory(model, codec, tokenizer, device, key, ids)
            native = codec.decode(vector)
            learned = decode_slots(head, vector, codec, tokenizer, ids)
            names.append({'name': name, 'input_ids': ids,
                          'native': {'token_ids': native['token_ids'],
                                     'decoded': tokenizer.decode(native['token_ids'], skip_special_tokens=True),
                                     'exact': native['token_ids'] == ids + [tokenizer.eos_token_id]},
                          'learned': learned})

        compositions = []
        for key, name in enumerate(NAMES):
            for word in WORDS:
                ids = tokenizer.encode(name + word, add_special_tokens=False)
                if len(ids) + 1 > SLOT_COUNT:
                    compositions.append({'name': name, 'word': word, 'status': 'capacity_exceeded'})
                    continue
                vector, _ = make_memory(model, codec, tokenizer, device, key, ids)
                learned = decode_slots(head, vector, codec, tokenizer, ids)
                compositions.append({'name': name, 'word': word, 'input_ids': ids,
                                     'learned': learned})

        random_ids = torch.randint(codec.vocab_size, (4096,), generator=generator, device=device)
        random_inputs = codec.codebook.index_select(0, random_ids)
        with torch.no_grad():
            random_pred = head(random_inputs).argmax(-1)
        random_accuracy = int((random_pred == random_ids).sum())

        if p.frozen_digest(model, chat.state) != original:
            raise ValueError('Original individual changed')
    if p.sha(p.runtime.CHECKPOINT) != source_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected source changed')

    report = {
        'status': 'identity_classifier_pilot',
        'train': {'steps': TRAIN_STEPS, 'batch': BATCH, 'trace': trace,
                  'random_codebook_accuracy': random_accuracy / len(random_ids)},
        'names': names,
        'compositions': compositions,
        'summary': {
            'names_learned_exact': sum(row['learned']['exact'] for row in names),
            'names_native_exact': sum(row['native']['exact'] for row in names),
            'composition_cases': sum('learned' in row for row in compositions),
            'composition_learned_exact': sum(row.get('learned', {}).get('exact', False) for row in compositions),
        },
        'codec': codec.config_dict(),
        'source_checkpoint_sha256': source_sha,
        'base_sha256': p.WEIGHT_SHA,
        'seconds': time.perf_counter() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'limitations': [
            'Explicit numeric addresses and manually constructed token sequences.',
            'The classifier is a research candidate and is not connected to runtime generation.',
            'No natural-language identity routing, open-domain name handling, or production persistence.',
        ],
    }
    OUT.mkdir(parents=False, exist_ok=False)
    torch.save({'format': 'spora-identity-classifier-v1',
                'head': {key: value.detach().cpu() for key, value in head.state_dict().items()},
                'codec': codec.config_dict(), 'source_sha256': source_sha,
                'base_sha256': p.WEIGHT_SHA}, OUT / 'candidate.pt')
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
