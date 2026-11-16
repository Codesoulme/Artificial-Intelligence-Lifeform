"""Train a frozen-LM first-token bridge from compressed SPORA memory codes.

Research probe only: fixed prompt, fixed slot and oracle address. The frozen
DeepSeek lm_head scores the adapter-modified final hidden state; target text is
never part of the student prompt. Token-ID-disjoint held-out types test narrow
generalization, not multi-token generation, semantics or natural QA.
"""
import collections
import hashlib
import json
import random
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch import nn
from torch.nn import functional as F

from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.language_training import atomic_save, frozen_digest
from spora.structured_text_memory import StructuredTextMemoryCodec

CORPUS = ROOT / 'runs/clean-zh-pretrain-v1/pretrain-train.jsonl'
MANIFEST = ROOT / 'runs/clean-zh-pretrain-v1/manifest.json'
OUT = ROOT / 'runs/deepseek-spora-chunk-first-token-bridge-20261117'
PROMPT = '请只输出记忆中保存的一个词元，不要解释。'
SEED = 21741
TRAIN_TYPES, HELD_TYPES, EVAL_TYPES = 2048, 256, 128
EXTENDED_TRAIN_TYPES = None
STEPS, BATCH = 320, 16


def split_token_types(counts, train_size, held_size, seed):
    if (not isinstance(counts, dict) or type(train_size) is not int or type(held_size) is not int
            or train_size < 1 or held_size < 2 or type(seed) is not int):
        raise ValueError('Invalid token split inputs')
    valid = sorted(token for token, count in counts.items()
                   if type(token) is int and token >= 0 and type(count) is int and count > 0)
    if len(valid) < train_size + held_size:
        raise ValueError('Insufficient eligible token types')
    random.Random(seed).shuffle(valid)
    return valid[:train_size], valid[train_size:train_size+held_size]


def extend_train_types(counts, train_ids, held_ids, total, seed):
    """Add types without changing the original train prefix or held-out set."""
    if type(total) is not int or total < len(train_ids):
        raise ValueError('Invalid expanded type count')
    forbidden = set(train_ids) | set(held_ids)
    available = sorted(token for token, count in counts.items()
                       if type(token) is int and token >= 0 and type(count) is int and count > 0
                       and token not in forbidden)
    random.Random(seed + 2).shuffle(available)
    needed = total - len(train_ids)
    if needed > len(available):
        raise ValueError('Insufficient extra token types')
    return list(train_ids) + available[:needed]


def chinese_token_counts(tokenizer, corpus):
    occurrences = collections.Counter()
    with corpus.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row.get('text'), str):
                raise ValueError('Invalid corpus row')
            occurrences.update(tokenizer.encode(row['text'], add_special_tokens=False))
    special = set(tokenizer.all_special_ids)
    result = {}
    for token, count in occurrences.items():
        if token in special:
            continue
        text = tokenizer.decode([token], skip_special_tokens=False)
        if 1 <= len(text) <= 4 and '\ufffd' not in text and re.fullmatch(r'[\u4e00-\u9fff]+', text):
            result[token] = int(count)
    return result


class FirstTokenBridge(nn.Module):
    """A small learned residual added before the frozen language output head."""
    def __init__(self, code_width, hidden_width, hidden=512):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(code_width, hidden), nn.GELU(), nn.Linear(hidden, hidden_width))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, codes):
        return self.layers(codes.float())


def actual_memory_code(model, codec, identity, value_id, device):
    """Write one packet to a fresh transient native matrix and read it back."""
    packet, metadata = codec.encode(identity, [int(value_id)])
    fast, velocity = model.initial_state(1)
    for site in fast:
        site[0] = torch.zeros_like(site[0])
    key = 1
    with torch.no_grad():
        for i, site in enumerate(model.memories):
            address = site.current_basis[key:key+1].to(device)
            fast[i], velocity[i] = site.write_temporal(
                fast[i], velocity[i], address, packet.reshape(1, -1).to(device),
                torch.ones(1, device=device), torch.zeros(1, dtype=torch.long, device=device), False)
        memory = torch.stack([site.retrieve([fast[i][0]], site.current_basis[key:key+1].to(device))[0]
                              for i, site in enumerate(model.memories)]).mean(0)
        try:
            recovered = codec.decode(memory, metadata)
            storage_exact = (recovered['identity']['token_ids'] == identity and
                             recovered['fact']['token_ids'] == [int(value_id)])
        except ValueError:
            storage_exact = False
        offset = codec.identity_slots * codec.identity_slot_width
        code = F.normalize(memory[offset:offset+codec.fact_slot_width].float(), dim=0)
    return code.detach(), storage_exact


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    source_sha = runtime.sha(CORPUS)
    if source_sha != manifest['files'][CORPUS.name]:
        raise ValueError('Training corpus hash mismatch')
    base_sha, checkpoint_sha = runtime.sha(runtime.MODEL / 'model.safetensors'), runtime.sha(runtime.CHECKPOINT)
    if base_sha != runtime.WEIGHT_SHA:
        raise ValueError('Base provenance mismatch')
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='spora-first-token-bridge-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        device = next(chat.organism.parameters()).device
        original_digest = frozen_digest(chat.organism, chat.state)
        counts = chinese_token_counts(chat.tokenizer, CORPUS)
        train_ids, held_ids = split_token_types(counts, TRAIN_TYPES, HELD_TYPES, SEED)
        if EXTENDED_TRAIN_TYPES is not None:
            train_ids = extend_train_types(counts, train_ids, held_ids, EXTENDED_TRAIN_TYPES, SEED)
        if set(train_ids) & set(held_ids) or len(set(train_ids+held_ids)) != len(train_ids)+HELD_TYPES:
            raise ValueError('Token type leakage')
        train_split_sha = hashlib.sha256(json.dumps(train_ids, separators=(',', ':')).encode('ascii')).hexdigest()
        held_split_sha = hashlib.sha256(json.dumps(held_ids, separators=(',', ':')).encode('ascii')).hexdigest()
        codec = StructuredTextMemoryCodec(chat.base.model.embed_tokens.weight.detach().cpu(),
                    memory_width=chat.organism.config.memory_width, identity_slots=3, fact_slots=5,
                    identity_slot_width=15, fact_slot_width=9, seed=29071,
                    eos_token_id=chat.tokenizer.eos_token_id).to(device)
        prompt = closed_think_inputs(chat.tokenizer, PROMPT, 'cuda')
        prompt_sha = hashlib.sha256(prompt['input_ids'].detach().cpu().numpy().tobytes()).hexdigest()
        with torch.inference_mode():
            h0 = chat.base.model(**prompt, use_cache=False, return_dict=True).last_hidden_state[:, -1].float().detach()
        head = chat.base.lm_head
        head.requires_grad_(False)
        bridge = FirstTokenBridge(codec.fact_slot_width, h0.shape[-1]).to(device)
        optimizer = torch.optim.AdamW(bridge.parameters(), lr=0.001, weight_decay=0.0001, foreach=False)
        train_tensor = torch.tensor(train_ids, dtype=torch.long, device=device)
        generator = torch.Generator(device=device).manual_seed(SEED+1)
        trace = []
        initial = {k: v.detach().clone() for k, v in bridge.state_dict().items()}
        for step in range(STEPS):
            index = torch.randint(len(train_ids), (BATCH,), generator=generator, device=device)
            label = train_tensor[index]
            code = codec.fact_codebook.index_select(0, label).detach()
            residual = bridge(code)
            logits = head((h0 + residual).to(head.weight.dtype)).float()
            loss = F.cross_entropy(logits, label)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite bridge loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = float(torch.sqrt(sum(p.grad.detach().float().square().sum()
                                             for p in bridge.parameters() if p.grad is not None)))
            if not bool(torch.isfinite(torch.tensor(gradient))):
                raise ValueError('Nonfinite bridge gradient')
            torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step in (0, 9, 49, 99, 199, STEPS-1) or (step + 1) % 500 == 0:
                point = {'step': step+1, 'loss': float(loss.detach()), 'gradient_norm': gradient}
                trace.append(point)
                print(json.dumps(point), flush=True)
        bridge.eval()
        change = float(torch.sqrt(sum((v.detach()-initial[k]).float().square().sum()
                                       for k, v in bridge.state_dict().items())))

        @torch.no_grad()
        def predict(codes):
            outputs = []
            for index in range(0, len(codes), 16):
                hidden = h0 + bridge(codes[index:index+16].to(device))
                outputs.extend(head(hidden.to(head.weight.dtype)).argmax(-1).cpu().tolist())
            return outputs

        def evaluate(ids):
            target = torch.tensor(ids, dtype=torch.long, device=device)
            codes = codec.fact_codebook.index_select(0, target).detach()
            predicted = predict(codes)
            return {'total': len(ids), 'exact': sum(p == t for p, t in zip(predicted, ids)),
                    'predicted_ids': predicted}

        train = evaluate(train_ids[:EVAL_TYPES])
        added_train = (evaluate(train_ids[TRAIN_TYPES:TRAIN_TYPES+EVAL_TYPES])
                       if len(train_ids) >= TRAIN_TYPES+EVAL_TYPES else None)
        held = evaluate(held_ids[:EVAL_TYPES])
        identity = chat.tokenizer.encode('用户', add_special_tokens=False)
        if not identity or len(identity)+1 > codec.identity_slots:
            raise ValueError('Unsupported test identity length')
        memory_codes, storage = [], []
        for value in held_ids[:EVAL_TYPES]:
            code, exact = actual_memory_code(chat.organism, codec, identity, value, device)
            memory_codes.append(code)
            storage.append(exact)
        memory_predictions = predict(torch.stack(memory_codes))
        no_memory = int(head(h0.to(head.weight.dtype)).argmax(-1)[0])
        pair_rows = []
        for i in range(0, EVAL_TYPES, 2):
            a, b = held_ids[i:i+2]
            pa, pb = memory_predictions[i:i+2]
            pair_rows.append({'pair': i//2, 'both_correct': pa == a and pb == b,
                              'outputs_differ': pa != pb, 'storage_both_exact': storage[i] and storage[i+1]})
        actual = {'total': EVAL_TYPES, 'storage_exact': sum(storage),
                  'exact': sum(p == t for p, t in zip(memory_predictions, held_ids[:EVAL_TYPES])),
                  'pair_total': len(pair_rows), 'pair_both_correct': sum(row['both_correct'] for row in pair_rows),
                  'pair_outputs_differ': sum(row['outputs_differ'] for row in pair_rows), 'pairs': pair_rows}
        if frozen_digest(chat.organism, chat.state) != original_digest:
            raise ValueError('Original individual changed')
        if any(param.requires_grad for param in head.parameters()):
            raise ValueError('Language output head unexpectedly trainable')
        checkpoint = {'format': 'spora-first-token-hidden-bridge-research-v1',
                      'state': {k: v.detach().cpu() for k, v in bridge.state_dict().items()},
                      'base_sha256': base_sha, 'integrated_sha256': checkpoint_sha,
                      'prompt_sha256': prompt_sha, 'seed': SEED, 'steps': STEPS,
                      'codec_config': codec.config_dict(), 'research_only': True}
    if (runtime.sha(runtime.MODEL / 'model.safetensors') != base_sha or
            runtime.sha(runtime.CHECKPOINT) != checkpoint_sha or runtime.sha(CORPUS) != source_sha):
        raise ValueError('Protected sources changed')
    OUT.mkdir()
    atomic_save(checkpoint, OUT / 'candidate.pt')
    result = {'status': 'isolated_first_token_bridge_not_deployed', 'prompt': PROMPT, 'prompt_sha256': prompt_sha,
              'source_sha256': {'base': base_sha, 'integrated': checkpoint_sha, 'corpus': source_sha},
              'candidate_sha256': runtime.sha(OUT / 'candidate.pt'), 'seed': SEED,
              'eligible_types': len(counts), 'train_types': len(train_ids), 'held_types': HELD_TYPES,
              'train_token_types_sha256': train_split_sha, 'held_token_types_sha256': held_split_sha,
              'evaluated_train_types': EVAL_TYPES, 'evaluated_held_types': EVAL_TYPES,
              'steps': STEPS, 'batch_size': BATCH, 'training_examples': STEPS*BATCH,
              'train': train, 'added_train': added_train, 'held_direct_code': held, 'held_native_memory': actual,
              'no_memory_top1_matches': sum(no_memory == token for token in held_ids[:EVAL_TYPES]),
              'gradient_trace': trace, 'parameter_change_norm': change,
              'seconds': time.perf_counter()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Single fixed prompt and first-token prediction; no multi-token or natural-language QA.',
                              'Chinese token IDs disjoint by type, but all come from one pretraining corpus split.',
                              'The frozen base may have seen all tokens during pretraining.',
                              'Training uses direct codebook vectors; test also measures explicit-address native memory reads.',
                              'Auxiliary adapter affects the output hidden state; memory and addresses are manually supplied.',
                              'No identity recognition, natural writing, durable learning, or ordinary-chat regression.',
                              'Candidate is a research artifact and not installed into the model runtime.']}
    with (OUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'train': train['exact'], 'held_direct': held['exact'],
                      'held_native': actual['exact'], 'storage': actual['storage_exact'],
                      'paired': actual['pair_both_correct']}), flush=True)


if __name__ == '__main__':
    main()
