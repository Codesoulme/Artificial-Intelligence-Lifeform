"""Isolated experiment for semantic decompression before language conditioning.

The prior reader sent a compressed continuous page directly into a small
memory-gated low-rank update.  This pilot first learns a frozen decoder from
the base model's token embeddings, then uses the decoded identity/fact slots
to condition the language update.  Numeric addresses and direct writes remain
deliberate controls; this is not an adopted runtime path.
"""
import json
import random
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_spora_reader_full_answer_pilot import unique_token_span
from spora.continuous_text_memory import ContinuousTextCodec


OUT = p.ROOT / 'runs/continuous-reader-semantic-bridge-pilot-20261112'
TEACHER_CANDIDATE = p.ROOT / 'runs/continuous-reader-memory-conditioned-lowrank-pilot-20261112/reader.pt'
SEED = 72381
BRIDGE_STEPS = 400
LANGUAGE_STEPS = 768
CODEC_SLOT_COUNT = 4
CODEC_ADD_EOS = True
STORE_FULL_FACT = False
BRIDGE_CONDITION_SLOTS = 2
CODEC_FACTORY = ContinuousTextCodec
WORDS = ('苹果', '香蕉', '西瓜', '葡萄', '草莓', '橘子', '桃子', '芒果',
         '梨子', '柠檬', '菠萝', '樱桃', '柚子', '荔枝', '榴莲', '蓝莓',
         '火龙果', '木瓜')


class TokenCodeDecoder(p.torch.nn.Module):
    """Decode one compressed code slot into the frozen embedding space."""

    def __init__(self, slot_width, embedding_width):
        super().__init__()
        self.linear = p.torch.nn.Linear(slot_width, embedding_width, bias=False)
        p.torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, codes):
        return p.torch.nn.functional.normalize(self.linear(codes.float()), dim=-1)


def fit_token_decoder(decoder, codec, embedding, *, steps=BRIDGE_STEPS, seed=SEED):
    """Train only an embedding reconstruction bridge, never the language base."""
    torch = p.torch
    generator = torch.Generator(device=embedding.device).manual_seed(seed + 17)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=2e-3, weight_decay=1e-4, foreach=False)
    trace = []
    for step in range(steps):
        token_ids = torch.randint(codec.vocab_size, (4096,), generator=generator, device=embedding.device)
        codes = codec.codebook.index_select(0, token_ids)
        targets = torch.nn.functional.normalize(embedding.index_select(0, token_ids).float(), dim=-1)
        prediction = decoder(codes)
        loss = 1.0 - (prediction * targets).sum(-1).mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite bridge loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step in (0, 9, 49, 99, 199, steps - 1):
            trace.append({'step': step + 1, 'cosine_loss': float(loss.detach())})
    return trace


def make_token_decoder(codec, embedding):
    return TokenCodeDecoder(codec.slot_width, embedding.shape[1]).to(embedding.device).eval()


class MemorySemanticGatedLowRankLinear(p.torch.nn.Module):
    """Research-only low-rank update conditioned on decoded memory slots."""

    def __init__(self, base, memory_context, memory_width, bridge, embedding_width,
                 condition_slots, rank=8):
        super().__init__()
        if not isinstance(base, p.torch.nn.Linear):
            raise ValueError('Expected a linear projection')
        self.base = base
        self.base.requires_grad_(False)
        self.memory_context = memory_context
        self.memory_width = memory_width
        self.slot_count = getattr(bridge, 'slot_count', CODEC_SLOT_COUNT)
        self.slot_width = memory_width // self.slot_count
        self.condition_slots = condition_slots
        self.bridge = bridge
        self.bridge.requires_grad_(False)
        self.a = p.torch.nn.Parameter(torch_empty(rank, base.in_features, base.weight.device))
        self.b = p.torch.nn.Parameter(p.torch.zeros(base.out_features, rank, device=base.weight.device,
                                                     dtype=p.torch.float32))
        self.condition = p.torch.nn.Parameter(
            torch_empty(rank, condition_slots * embedding_width, base.weight.device))
        p.torch.nn.init.kaiming_uniform_(self.a, a=5 ** 0.5)
        p.torch.nn.init.xavier_uniform_(self.condition)

    def forward(self, x):
        active = self.memory_context.get()
        base_output = self.base(x)
        if active is None:
            return base_output
        if x.shape[0] != 1:
            raise ValueError('Research gating supports only one row')
        matrices, addresses, mask, slots = active
        if mask is not None:
            if mask.numel() != 1:
                raise ValueError('Research gating supports a single read mask')
            if not bool(mask[0]):
                return base_output
        if (addresses is None or len(addresses) != 2 or len(matrices) != 2
                or slots is None or slots.numel() != 1 or int(slots[0]) != 0):
            raise ValueError('Semantic bridge requires two fixed current addresses')
        reads = []
        for site in range(2):
            address, matrix = addresses[site], matrices[site][0]
            if address.shape != (1, self.memory_width) or matrix.shape != (1, self.memory_width, self.memory_width):
                raise ValueError('Unexpected memory dimensions')
            query = p.torch.nn.functional.normalize(
                address.to(device=x.device, dtype=p.torch.float32), dim=-1)
            raw = p.torch.bmm(matrix.to(device=x.device, dtype=p.torch.float32),
                              query.unsqueeze(-1)).squeeze(-1)
            reads.append(raw)
        packed = torch_mean(reads[0], reads[1])[:, :self.slot_count * self.slot_width]
        memory = packed.reshape(1, self.slot_count, self.slot_width)
        slots = p.torch.nn.functional.normalize(memory[:, :self.condition_slots], dim=-1)
        if hasattr(self.bridge, 'set_cache_key'):
            self.bridge.set_cache_key(tuple(id(matrices[i][0]) for i in range(2))
                                      + tuple(id(addresses[i]) for i in range(2)))
        decoded = self.bridge(slots).reshape(1, -1)
        gate = p.torch.tanh(p.torch.nn.functional.linear(decoded, self.condition))
        hidden = p.torch.nn.functional.linear(x.float(), self.a) * gate.unsqueeze(1)
        delta = p.torch.nn.functional.linear(hidden, self.b)
        return base_output + delta.to(x.dtype)


def torch_empty(rows, cols, device):
    return p.torch.empty(rows, cols, device=device, dtype=p.torch.float32)


def torch_mean(left, right):
    return (left + right) * 0.5


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch = p.torch
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    rng = random.Random(SEED)
    started = time.perf_counter()
    source_sha = p.sha(p.runtime.CHECKPOINT)
    tok = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    word_ids = [tok.encode(word, add_special_tokens=False) for word in WORDS]
    if any(not ids or len(ids) > 3 for ids in word_ids):
        raise ValueError('Each word must occupy one to three tokens')
    if len({ids[0] for ids in word_ids}) != len(word_ids):
        raise ValueError('Fruit first-token collision')
    identity_ids = {0: tok.encode('小林', add_special_tokens=False)[-1],
                    1: tok.encode('小周', add_special_tokens=False)[-1],
                    2: tok.encode('小王', add_special_tokens=False)[-1],
                    3: tok.encode('小赵', add_special_tokens=False)[-1]}
    if len(set(identity_ids.values())) != len(identity_ids):
        raise ValueError('Identity token collision')

    with tempfile.TemporaryDirectory() as temp:
        p.runtime.WORKING_CHECKPOINT = Path(temp) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        decoder, model = chat.decoder.eval(), chat.organism.eval()
        if any(x.is_meta for x in list(chat.base.parameters()) + list(chat.base.buffers())):
            raise ValueError('Unmaterialized base')
        original_digest = p.frozen_digest(model, chat.state)
        device = next(model.parameters()).device
        embedding = chat.base.model.embed_tokens.weight.detach()
        # Build the vocabulary projection on CPU: the embedding-by-projection
        # temporary is large enough to fragment a 6 GB CUDA heap.
        codec = CODEC_FACTORY(embedding.detach().cpu(), memory_width=model.config.memory_width,
                              slot_count=CODEC_SLOT_COUNT, seed=29071,
                              eos_token_id=tok.eos_token_id if CODEC_ADD_EOS else None).to(device)
        bridge = make_token_decoder(codec, embedding)
        bridge.slot_count = CODEC_SLOT_COUNT
        bridge_trace = fit_token_decoder(bridge, codec, embedding)
        bridge.requires_grad_(False)
        decoder.requires_grad_(False)
        decoder.reader.requires_grad_(True)
        adapters = []
        for index in (18, 20):
            attention = decoder.base.model.layers[index].self_attn
            for name in ('q_proj', 'v_proj'):
                adapter = MemorySemanticGatedLowRankLinear(
                    getattr(attention, name), decoder._memory_context, model.config.memory_width,
                    bridge, embedding.shape[1], BRIDGE_CONDITION_SLOTS)
                setattr(attention, name, adapter)
                adapters.append((index, name, adapter))
        slow_patches = []
        for site in decoder.reader.native_sites:
            patcher = patch.object(site, 'slow_signal', side_effect=lambda x, query: torch.zeros_like(query))
            patcher.start()
            with torch.no_grad():
                site.read_out.bias.zero_()
            site.read_out.bias.requires_grad_(False)
            slow_patches.append(patcher)
        candidate = torch.load(TEACHER_CANDIDATE, map_location='cpu', weights_only=True)
        teacher_sha = p.sha(TEACHER_CANDIDATE)
        if (candidate['source_sha256'] != source_sha or candidate['base_sha256'] != p.WEIGHT_SHA
                or not candidate.get('adapter_requires_addressed_current_memory')):
            raise ValueError('Teacher candidate source mismatch')
        decoder.reader.load_state_dict(candidate['reader'], strict=True)
        with torch.no_grad():
            for (index, name, adapter), old in zip(adapters, candidate['adapters']):
                if old['layer'] != index or old['projection'] != name:
                    raise ValueError('Previous adapter position mismatch')
                adapter.a.copy_(old['a'])
                adapter.b.copy_(old['b'])

        def state_for(value, key):
            state = p.fresh_controlled_state(model)
            fast, velocity = model.initial_state(1)
            for site in fast:
                site[0] = torch.zeros_like(site[0])
            if value is not None:
                target = torch.zeros((1, model.config.memory_width), device=device)
                fact_ids = word_ids[value] if STORE_FULL_FACT else [word_ids[value][0]]
                encoded = codec.encode(torch.tensor([identity_ids[key], *fact_ids], device=device))
                target[0, :encoded.numel()] = encoded
                with torch.no_grad():
                    for i, site in enumerate(model.memories):
                        fast[i], velocity[i] = site.write_temporal(
                            fast[i], velocity[i], site.current_basis[key:key + 1].to(device),
                            target, torch.ones(1, device=device), torch.zeros(1, dtype=torch.long, device=device), False)
                        read = site.retrieve([fast[i][0]], site.current_basis[key:key + 1].to(device))
                        decoded = codec.decode(read[0, :encoded.numel()])['token_ids']
                        expected = [identity_ids[key], *fact_ids]
                        if CODEC_ADD_EOS:
                            expected.append(tok.eos_token_id)
                        if decoded != expected:
                            raise ValueError('Controlled code reconstruction failed')
            state.current = [x[0].detach() for x in fast]
            return state

        questions = ('请只说出{name}喜欢的水果名称。',
                     '记录里{name}喜欢哪种水果？只写名称。',
                     '{name}当前最喜欢的水果是什么？')
        cases = []
        states = {}
        schedule = [('train', key, value, question)
                    for key in (0, 1, 2) for value in range(16) for question in questions]
        schedule += [('held_name', 3, value, questions[0]) for value in range(18)]
        for group, key, value, question_template in schedule:
            states.setdefault((key, value), state_for(value, key))
            state = states[key, value]
            name = ('小林', '小周', '小王', '小赵')[key]
            question = question_template.format(name=name)
            answer = f'{name}喜欢的水果是{WORDS[value]}。'
            prompt = p.closed_think_inputs(tok, question, 'cuda')
            answer_tokens = tok.encode(answer, add_special_tokens=False)
            value_position = unique_token_span(answer_tokens, word_ids[value])
            target = torch.tensor([answer_tokens + [tok.eos_token_id]], device=device)
            ids = torch.cat((prompt['input_ids'], target), dim=1)
            spec = p._memory_args(model, state, {'key': key, 'intent': 'current', 'event_time': 0})
            cases.append({'group': group, 'key': key, 'value': value, 'name': name,
                          'question': question, 'state': state, 'prompt': prompt, 'ids': ids,
                          'spec': spec, 'value_position': value_position,
                          'digest': p.frozen_digest(model, state)})
        empty = state_for(None, 0)
        empty_digest = p.frozen_digest(model, empty)

        def evaluate():
            rows = []
            selected = [x for x in cases if x['group'] == 'held_name' or
                        (x['group'] == 'train' and x['value'] < 4 and x['question'] == questions[0].format(name=x['name']))]
            for item in selected:
                conditions = [('memory', item['spec'])]
                if (item['group'], item['value']) in (('train', 0), ('held_name', 16)):
                    conditions += [('empty', p._memory_args(model, empty,
                        {'key': item['key'], 'intent': 'current', 'event_time': 0})), ('off', None)]
                for condition, spec in conditions:
                    with torch.inference_mode():
                        kwargs = {} if spec is None else spec
                        forced = decoder(input_ids=item['ids'], attention_mask=torch.ones_like(item['ids']), **kwargs).logits.float()
                        prefix = item['prompt']['input_ids'].shape[1]
                        scores = forced[0, prefix - 1:prefix - 1 + item['ids'].shape[1] - prefix]
                        fact_scores = scores[item['value_position']]
                        correct_id = word_ids[item['value']][0]
                        competitor = word_ids[(item['value'] + 1) % 16][0]
                        output = decoder.generate(**item['prompt'], **({**spec, 'use_cache': True} if spec else {}),
                            max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id,
                            eos_token_id=tok.eos_token_id)
                    tail = output[0, prefix:].tolist()
                    answer = tok.decode(tail, skip_special_tokens=True).strip()
                    rows.append({'group': item['group'], 'name': item['name'], 'condition': condition,
                                 'question': item['question'], 'expected': WORDS[item['value']],
                                 'answer': answer, 'token_ids': tail, 'exact': answer == WORDS[item['value']],
                                 'contains_expected': WORDS[item['value']] in answer,
                                 'forced_fact_rank': int((fact_scores > fact_scores[correct_id]).sum()) + 1,
                                 'forced_pair_margin': float(fact_scores[correct_id] - fact_scores[competitor])})
            return rows

        before = evaluate()
        train = [x for x in cases if x['group'] == 'train']
        params = [x for x in decoder.reader.parameters() if x.requires_grad]
        params += [param for _, _, adapter in adapters for param in (adapter.a, adapter.b, adapter.condition)]
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0, foreach=False)
        trace = []
        for step in range(LANGUAGE_STEPS):
            if step % len(train) == 0:
                rng.shuffle(train)
            item = train[step % len(train)]
            prefix = item['prompt']['input_ids'].shape[1]
            optimizer.zero_grad(set_to_none=True)
            logits = decoder(input_ids=item['ids'], attention_mask=torch.ones_like(item['ids']), **item['spec']).logits.float()
            target = item['ids'][:, prefix:]
            scores = logits[:, prefix - 1:prefix - 1 + target.shape[1]]
            full_loss = torch.nn.functional.cross_entropy(scores.reshape(-1, scores.shape[-1]), target.reshape(-1))
            other_value = (item['value'] + 1 + rng.randrange(15)) % 16
            other_spec = p._memory_args(model, states[item['key'], other_value],
                                        {'key': item['key'], 'intent': 'current', 'event_time': 0})
            other_logits = decoder(input_ids=item['ids'], attention_mask=torch.ones_like(item['ids']),
                                   **other_spec).logits[:, prefix - 1 + item['value_position'], :].float()
            correct_id, wrong_id = word_ids[item['value']][0], word_ids[other_value][0]
            own = scores[:, item['value_position'], :]
            correct_margin = own[:, correct_id] - own[:, wrong_id]
            wrong_margin = other_logits[:, correct_id] - other_logits[:, wrong_id]
            contrast = 0.5 * (torch.nn.functional.softplus(1.0 - correct_margin).mean()
                              + torch.nn.functional.softplus(1.0 + wrong_margin).mean())
            fact_ce = torch.nn.functional.cross_entropy(own, torch.tensor([correct_id], device=device))
            swapped_ce = torch.nn.functional.cross_entropy(other_logits, torch.tensor([wrong_id], device=device))
            loss = 4.0 * full_loss + 0.5 * (fact_ce + swapped_ce) + contrast
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite language loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step in (0, 9, 49, 99, 199, 399, 767):
                trace.append({'step': step + 1, 'loss': float(loss.detach()),
                              'full_loss': float(full_loss.detach()), 'contrast': float(contrast.detach()),
                              'grad_norm': float(norm)})
            if step % 64 == 0:
                print(json.dumps(trace[-1] if trace and trace[-1]['step'] == step + 1 else {'step': step + 1,
                    'loss': float(loss.detach())}), flush=True)
        after = evaluate()
        if p.frozen_digest(model, chat.state) != original_digest or p.frozen_digest(model, empty) != empty_digest:
            raise ValueError('Protected state changed')
        if any(p.frozen_digest(model, x['state']) != x['digest'] for x in cases):
            raise ValueError('Experimental memory changed')
        reader_state = {k: v.detach().cpu().clone() for k, v in decoder.reader.state_dict().items()}
        adapter_state = [{'layer': index, 'projection': name, 'a': adapter.a.detach().cpu().clone(),
                          'b': adapter.b.detach().cpu().clone(), 'condition': adapter.condition.detach().cpu().clone()}
                         for index, name, adapter in adapters]
        bridge_state = {k: v.detach().cpu().clone() for k, v in bridge.state_dict().items()}
        bridge.eval()
        for patcher in slow_patches:
            patcher.stop()
    if p.sha(p.runtime.CHECKPOINT) != source_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected source changed')
    OUT.mkdir(parents=False, exist_ok=False)
    torch.save({'reader': reader_state, 'bridge': bridge_state, 'adapters': adapter_state,
                'source_sha256': source_sha, 'base_sha256': p.WEIGHT_SHA, 'research_only': True,
                'memory_conditioning': 'embedding_reconstructed_identity_fact_slots',
                'bridge_steps': BRIDGE_STEPS, 'language_steps': LANGUAGE_STEPS}, OUT / 'candidate.pt')
    report = {'status': 'semantic_memory_bridge_isolated_pilot', 'before': before, 'after': after,
              'bridge_trace': bridge_trace, 'language_trace': trace, 'seed': SEED,
              'bridge_steps': BRIDGE_STEPS, 'language_steps': LANGUAGE_STEPS,
              'source_sha256': source_sha, 'base_sha256': p.WEIGHT_SHA,
              'teacher_candidate_sha256': teacher_sha, 'candidate_sha256': p.sha(OUT / 'candidate.pt'),
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Numeric addresses and direct writes bypass natural-language routing.',
                              'The bridge is trained from frozen token embeddings, not language data.',
                              'Only identity and first fruit tokens are stored in two of four slots.',
                              'The bridge and language adapter are experimental and not adopted by the individual.',
                              'The finite authored split is not an open-memory or open-chat evaluation.']}
    with (OUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'seconds': report['seconds'], 'peak_allocated_bytes': report['peak_allocated_bytes'],
                      'after_rows': len(after)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
