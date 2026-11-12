"""Train a small autoregressive text reconstructor from SPORA memory semantics.

This is a research-only text-level memory probe. It does not use the Qwen
decoder to answer, does not inject output logits, and does not modify the
individual. It asks whether a learned memory-to-text decoder can reconstruct
complete Chinese fact sentences from continuous memory pages.
"""
import json
import random
import tempfile
import time
from pathlib import Path

from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_spora_semantic_bridge_pilot import TokenCodeDecoder, fit_token_decoder
from spora.continuous_text_memory import ContinuousTextCodec


OUT = p.ROOT / 'runs/continuous-reader-text-reconstruction-pilot-20261113'
SEED = 88731
BRIDGE_STEPS = 400
DECODER_STEPS = 1200
HIDDEN = 384
WORDS = ('苹果', '香蕉', '西瓜', '葡萄', '草莓', '橘子', '桃子', '芒果',
         '梨子', '柠檬', '菠萝', '樱桃', '柚子', '荔枝', '榴莲', '蓝莓',
         '火龙果', '木瓜')
NAMES = ('小林', '小周', '小王', '小赵')


class MemoryTextReconstructor(p.torch.nn.Module):
    def __init__(self, embedding, memory_width, slot_count, bridge, hidden=HIDDEN):
        super().__init__()
        self.embedding = embedding.detach().float()
        self.embedding.requires_grad_(False)
        self.embedding_width = embedding.shape[1]
        self.slot_count = slot_count
        self.slot_width = memory_width // slot_count
        self.hidden_size = hidden
        self.bridge = bridge
        self.bridge.requires_grad_(False)
        self.memory_encoder = p.torch.nn.Sequential(
            p.torch.nn.Linear(4 * self.embedding_width, hidden),
            p.torch.nn.Tanh(),
        )
        self.gru = p.torch.nn.GRUCell(self.embedding_width, hidden)
        self.to_embedding = p.torch.nn.Linear(hidden, self.embedding_width, bias=False)

    def decode_memory(self, memory):
        slots = memory[:, :4 * self.slot_width].reshape(memory.shape[0], 4, self.slot_width)
        slots = p.torch.nn.functional.normalize(slots, dim=-1)
        slots = self.bridge(slots).reshape(memory.shape[0], -1)
        return self.memory_encoder(slots.float())

    def logits_from_hidden(self, hidden):
        projected = self.to_embedding(hidden.float())
        return projected @ self.embedding.T

    def forward(self, memory, target_ids):
        hidden = self.decode_memory(memory)
        outputs = []
        for position in range(target_ids.shape[1]):
            outputs.append(self.logits_from_hidden(hidden))
            if position + 1 < target_ids.shape[1]:
                token_embedding = self.embedding.index_select(0, target_ids[:, position])
                hidden = self.gru(token_embedding.float(), hidden)
        return p.torch.stack(outputs, dim=1)

    @p.torch.no_grad()
    def generate(self, memory, *, eos_id, max_new_tokens=32):
        hidden = self.decode_memory(memory)
        tokens = []
        for _ in range(max_new_tokens):
            next_id = self.logits_from_hidden(hidden).argmax(-1)
            tokens.append(next_id)
            token_embedding = self.embedding.index_select(0, next_id)
            hidden = self.gru(token_embedding.float(), hidden)
            if bool((next_id == eos_id).all()):
                break
        return p.torch.stack(tokens, dim=1)


def make_state(model, codec, tokenizer, device, key, value):
    fast, velocity = model.initial_state(1)
    for site in fast:
        site[0] = p.torch.zeros_like(site[0])
    ids = [tokenizer.encode(NAMES[key], add_special_tokens=False)[-1]]
    ids += tokenizer.encode(WORDS[value], add_special_tokens=False)
    encoded = codec.encode(p.torch.tensor(ids, dtype=p.torch.long, device=device))
    target = p.torch.zeros((1, model.config.memory_width), device=device)
    target[0, :encoded.numel()] = encoded
    with p.torch.no_grad():
        for index, site in enumerate(model.memories):
            fast[index], velocity[index] = site.write_temporal(
                fast[index], velocity[index], site.current_basis[key:key + 1].to(device),
                target, p.torch.ones(1, device=device),
                p.torch.zeros(1, dtype=p.torch.long, device=device), False)
    reads = []
    for index, site in enumerate(model.memories):
        reads.append(site.retrieve([fast[index][0]], site.current_basis[key:key + 1].to(device)))
    memory = ((reads[0] + reads[1]) * 0.5).detach()
    return memory


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
        original_digest = p.frozen_digest(model, chat.state)
        embedding = chat.base.model.embed_tokens.weight.detach()
        codec = ContinuousTextCodec(embedding.cpu(), memory_width=model.config.memory_width,
                                    slot_count=5, seed=29071,
                                    eos_token_id=tokenizer.eos_token_id).to(device)
        bridge = TokenCodeDecoder(codec.slot_width, embedding.shape[1]).to(device)
        bridge_trace = fit_token_decoder(bridge, codec, embedding, steps=BRIDGE_STEPS)
        bridge.eval().requires_grad_(False)
        reconstructor = MemoryTextReconstructor(embedding, model.config.memory_width, 5, bridge).to(device)
        cases = []
        for key in range(3):
            for value in range(len(WORDS)):
                memory = make_state(model, codec, tokenizer, device, key, value)
                answer = f'{NAMES[key]}喜欢的水果是{WORDS[value]}。'
                target = tokenizer.encode(answer, add_special_tokens=False)
                target.append(tokenizer.eos_token_id)
                cases.append({'group': 'train', 'key': key, 'value': value,
                              'answer': answer, 'memory': memory,
                              'target': torch.tensor([target], device=device)})
        held = []
        for value in range(len(WORDS)):
            memory = make_state(model, codec, tokenizer, device, 3, value)
            answer = f'{NAMES[3]}喜欢的水果是{WORDS[value]}。'
            target = tokenizer.encode(answer, add_special_tokens=False)
            target.append(tokenizer.eos_token_id)
            held.append({'group': 'held_name', 'key': 3, 'value': value,
                         'answer': answer, 'memory': memory,
                         'target': torch.tensor([target], device=device)})

        optimizer = torch.optim.AdamW(reconstructor.parameters(), lr=2e-3,
                                      weight_decay=1e-4, foreach=False)
        order = list(range(len(cases)))
        trace = []
        for step in range(DECODER_STEPS):
            if step % len(order) == 0:
                random.shuffle(order)
            item = cases[order[step % len(order)]]
            optimizer.zero_grad(set_to_none=True)
            logits = reconstructor(item['memory'], item['target']).float()
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), item['target'].reshape(-1))
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite text reconstruction loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), 1.0,
                                                  error_if_nonfinite=True)
            optimizer.step()
            if step in (0, 9, 49, 99, 199, 399, 799, DECODER_STEPS - 1):
                trace.append({'step': step + 1, 'loss': float(loss.detach()),
                              'grad_norm': float(norm)})
                print(json.dumps(trace[-1]), flush=True)

        def evaluate(rows):
            output = []
            for item in rows:
                with torch.inference_mode():
                    generated = reconstructor.generate(item['memory'],
                                                       eos_id=tokenizer.eos_token_id,
                                                       max_new_tokens=32)
                token_ids = generated[0].tolist()
                text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                output.append({'group': item['group'], 'name': NAMES[item['key']],
                               'expected': WORDS[item['value']], 'target': item['answer'],
                               'answer': text, 'token_ids': token_ids,
                               'exact': text == item['answer'],
                               'contains_expected': WORDS[item['value']] in text})
            return output

        train_after = evaluate(cases)
        held_after = evaluate(held)
        if p.frozen_digest(model, chat.state) != original_digest:
            raise ValueError('Original individual changed')
    if p.sha(p.runtime.CHECKPOINT) != source_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected source changed')
    OUT.mkdir(parents=False, exist_ok=False)
    candidate_path = OUT / 'reconstructor.pt'
    torch.save({'format': 'spora-text-reconstructor-v1',
                'state': {key: value.detach().cpu() for key, value in reconstructor.state_dict().items()},
                'bridge': {key: value.detach().cpu() for key, value in bridge.state_dict().items()},
                'source_sha256': source_sha, 'base_sha256': p.WEIGHT_SHA,
                'codec': codec.config_dict(), 'steps': DECODER_STEPS}, candidate_path)
    report = {'status': 'isolated_memory_to_text_reconstruction',
              'train': train_after, 'held': held_after,
              'train_contains': sum(x['contains_expected'] for x in train_after),
              'held_contains': sum(x['contains_expected'] for x in held_after),
              'train_exact': sum(x['exact'] for x in train_after),
              'held_exact': sum(x['exact'] for x in held_after),
              'bridge_trace': bridge_trace, 'decoder_trace': trace,
              'source_sha256': source_sha, 'base_sha256': p.WEIGHT_SHA,
              'candidate_sha256': p.sha(candidate_path),
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Direct numeric addresses and controlled writes.',
                              'Canonical authored fact sentences, no question conditioning.',
                              'Small GRU reconstructor is separate from the Qwen conversational decoder.',
                              'Held identity shares the finite fruit vocabulary and is not open-domain.',
                              'Candidate is not adopted by the individual.']}
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'seconds': report['seconds'], 'train_contains': report['train_contains'],
                      'held_contains': report['held_contains']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
