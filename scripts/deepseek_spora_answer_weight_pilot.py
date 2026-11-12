"""Isolated experiment: generate short text from continuous fast-memory vectors.

Synthetic numeric addresses and direct writes are deliberate experimental controls.
This is not a natural-language write path or a replacement for the active reader.
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

OUT = p.ROOT / 'runs/continuous-reader-answer-weight-pilot-20261112'
TEACHER_CANDIDATE = p.ROOT / 'runs/continuous-reader-memory-conditioned-lowrank-pilot-20261112/reader.pt'
SEED = 72381
WORDS = ('苹果', '香蕉', '西瓜', '葡萄', '草莓', '橘子', '桃子', '芒果',
         '梨子', '柠檬', '菠萝', '樱桃', '柚子', '荔枝', '榴莲', '蓝莓', '火龙果', '木瓜')


class LowRankLinear(p.torch.nn.Module):
    """Experimental zero-initialized low-rank update on a frozen linear layer."""
    def __init__(self, base, rank=8):
        super().__init__()
        if not isinstance(base, p.torch.nn.Linear):
            raise ValueError('Expected a linear projection')
        self.base = base
        self.base.requires_grad_(False)
        self.a = p.torch.nn.Parameter(p.torch.empty(rank, base.in_features, device=base.weight.device, dtype=p.torch.float32))
        self.b = p.torch.nn.Parameter(p.torch.zeros(base.out_features, rank, device=base.weight.device, dtype=p.torch.float32))
        p.torch.nn.init.kaiming_uniform_(self.a, a=5 ** 0.5)
        self.enabled = True

    def forward(self, x):
        base_output = self.base(x)
        if not self.enabled:
            return base_output
        delta = p.torch.nn.functional.linear(p.torch.nn.functional.linear(x.float(), self.a), self.b)
        return base_output + delta.to(base_output.dtype)


class MemoryGatedLowRankLinear(LowRankLinear):
    """Research-only: modulate updates by two addressed current matrices."""
    def __init__(self, base, memory_context, memory_width, rank=8):
        super().__init__(base, rank)
        self.memory_context = memory_context
        self.memory_width = memory_width
        self.condition = p.torch.nn.Parameter(p.torch.empty(rank, memory_width, device=base.weight.device, dtype=p.torch.float32))
        p.torch.nn.init.xavier_uniform_(self.condition)

    def forward(self, x):
        active = self.memory_context.get()
        if active is None:
            return self.base(x)
        if x.shape[0] != 1:
            raise ValueError('Research gating supports only one row')
        matrices, addresses, mask, slots = active
        if mask is not None:
            if mask.numel() != 1:
                raise ValueError('Research gating supports a single read mask')
            if not bool(mask[0]):
                return self.base(x)
        if not self.enabled:
            return self.base(x)
        if (addresses is None or len(addresses) != 2 or len(matrices) != 2
                or slots is None or slots.numel() != 1 or int(slots[0]) != 0):
            raise ValueError('Research adapter requires two fixed addresses and current memory')
        reads = []
        for site in range(2):
            address, matrix = addresses[site], matrices[site][0]
            if address.shape != (1, self.memory_width) or matrix.shape != (1, self.memory_width, self.memory_width):
                raise ValueError('Unexpected memory dimensions')
            query = p.torch.nn.functional.normalize(address.to(device=x.device, dtype=p.torch.float32), dim=-1)
            reads.append(p.torch.bmm(matrix.to(device=x.device, dtype=p.torch.float32), query.unsqueeze(-1)).squeeze(-1))
        memory = (reads[0] + reads[1]) * 0.5
        gate = p.torch.tanh(p.torch.nn.functional.linear(memory, self.condition))
        hidden = p.torch.nn.functional.linear(x.float(), self.a) * gate.unsqueeze(1)
        delta = p.torch.nn.functional.linear(hidden, self.b)
        return self.base(x) + delta.to(x.dtype)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch = p.torch
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    rng = random.Random(SEED)
    start = time.perf_counter()
    source_sha = p.sha(p.runtime.CHECKPOINT)
    tok = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    word_ids = [tok.encode(word, add_special_tokens=False) for word in WORDS]
    if any(not ids or len(ids) > 3 for ids in word_ids):
        raise ValueError('Each word must occupy one to three tokens')
    if len({ids[0] for ids in word_ids}) != len(word_ids):
        raise ValueError('Training and held values must have distinct first tokens')
    with tempfile.TemporaryDirectory() as temp:
        p.runtime.WORKING_CHECKPOINT = Path(temp) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        decoder, model = chat.decoder.eval(), chat.organism.eval()
        if any(x.is_meta for x in list(chat.base.parameters()) + list(chat.base.buffers())):
            raise ValueError('Unmaterialized base')
        original_digest = p.frozen_digest(model, chat.state)
        device = next(model.parameters()).device
        codec = ContinuousTextCodec(chat.base.model.embed_tokens.weight.detach(),
                                    memory_width=model.config.memory_width, slot_count=4,
                                    seed=29071, eos_token_id=tok.eos_token_id)
        decoder.requires_grad_(False)
        decoder.reader.requires_grad_(True)
        # Add a small trainable update in later attention projections, after
        # the second memory reader. No base weights are modified or saved.
        adapters = []
        for index in (18, 20):
            attention = decoder.base.model.layers[index].self_attn
            for name in ('q_proj', 'v_proj'):
                adapter = MemoryGatedLowRankLinear(getattr(attention, name), decoder._memory_context,
                                                   model.config.memory_width)
                setattr(attention, name, adapter)
                adapters.append((index, name, adapter))
        # Research-only isolation: disable slow expert signal and output bias.
        # Patch lifetime is the process-scoped experimental decoder only.
        slow_patches = []
        for site in decoder.reader.native_sites:
            patcher = patch.object(site, 'slow_signal', side_effect=lambda x, query: torch.zeros_like(query))
            patcher.start()
            slow_patches.append(patcher)
            with torch.no_grad():
                site.read_out.bias.zero_()
            site.read_out.bias.requires_grad_(False)
        params = [x for x in decoder.reader.parameters() if x.requires_grad]
        params += [param for _, _, adapter in adapters for param in (adapter.a, adapter.b, adapter.condition)]
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0, foreach=False)
        mode = torch.zeros(1, dtype=torch.long, device=device)
        gate = torch.ones(1, device=device)

        def state_for(value, key):
            state = p.fresh_controlled_state(model)
            fast, velocity = model.initial_state(1)
            for site in fast:
                site[0] = torch.zeros_like(site[0])
            if value is not None:
                target = torch.zeros((1, model.config.memory_width), device=device)
                encoded = codec.encode(torch.tensor(word_ids[value], device=device))
                target[0, :encoded.numel()] = encoded
                with torch.no_grad():
                    for i, site in enumerate(model.memories):
                        fast[i], velocity[i] = site.write_temporal(
                            fast[i], velocity[i], site.current_basis[key:key + 1].to(device),
                            target, gate, mode, False)
                        read = site.retrieve([fast[i][0]], site.current_basis[key:key + 1].to(device))
                        decoded = codec.decode(read[0, :encoded.numel()])['token_ids']
                        if decoded != word_ids[value] + [tok.eos_token_id]:
                            raise ValueError('Independent codebook reconstruction failed')
            state.current = [x[0].detach() for x in fast]
            return state

        cases = []
        # Each held split changes exactly one dimension from a training combination.
        train_questions = ('请只说出{name}喜欢的水果名称。', '记录里{name}喜欢哪种水果？只写名称。',
                           '{name}当前最喜欢的水果是什么？')
        schedule = [('train', ((0, '小林'), (1, '小周'), (2, '小王')), range(16), q) for q in train_questions]
        schedule += [
            ('held_name', ((6, '小陈'),), (0, 1), train_questions[0]),
            ('held_question', ((0, '小林'),), (0, 1), '{name}爱吃什么水果？只回答名字。'),
            ('held_word', ((0, '小林'),), (16, 17), train_questions[0]),
        ]
        states = {}
        for group, names, values, question in schedule:
            for key, name in names:
                for value in values:
                    if (key, value) not in states:
                        states[key, value] = state_for(value, key)
                    state = states[key, value]
                    prompt = p.closed_think_inputs(tok, question.format(name=name), 'cuda')
                    teacher_prompt = p.closed_think_inputs(tok,
                        f'已知{name}喜欢的水果是{WORDS[value]}。' + question.format(name=name), 'cuda')
                    answer = f'{name}喜欢的水果是{WORDS[value]}。'
                    answer_tokens = tok.encode(answer, add_special_tokens=False)
                    value_position = unique_token_span(answer_tokens, word_ids[value])
                    target = torch.tensor([answer_tokens + [tok.eos_token_id]], device=device)
                    ids = torch.cat((prompt['input_ids'], target), dim=1)
                    teacher_ids = torch.cat((teacher_prompt['input_ids'], target), dim=1)
                    spec = p._memory_args(model, state, {'key': key, 'intent': 'current', 'event_time': 0})
                    cases.append(dict(group=group, name=name, key=key, value=value, question=question, state=state,
                                      prompt=prompt, teacher_prompt=teacher_prompt, teacher_ids=teacher_ids,
                                      value_position=value_position, ids=ids, spec=spec,
                                      digest=p.frozen_digest(model, state)))
        empty = state_for(None, 0)
        empty_digest = p.frozen_digest(model, empty)

        def evaluate():
            rows = []
            # A fixed question/name with different values makes copying the prompt insufficient.
            subset = [x for x in cases if (x['group'] != 'train' or
                      x['key'] == 0 and x['value'] < 2 and x['question'] == train_questions[0])]
            for item in subset:
                conditions = [('memory', item['spec'])]
                if (item['group'], item['value']) in (('train', 0), ('held_word', 16)):
                    conditions += [('empty', p._memory_args(model, empty,
                        {'key': item['key'], 'intent': 'current', 'event_time': 0})), ('off', {})]
                for condition, spec in conditions:
                    with torch.inference_mode():
                        forced = decoder(input_ids=item['ids'], attention_mask=torch.ones_like(item['ids']),
                                         **spec).logits.float()
                        prefix = item['prompt']['input_ids'].shape[1]
                        fact_scores = forced[0, prefix - 1 + item['value_position']]
                        correct_id = word_ids[item['value']][0]
                        correct_logprob = float(fact_scores.log_softmax(-1)[correct_id])
                        correct_rank = int((fact_scores > fact_scores[correct_id]).sum()) + 1
                        pair_id = word_ids[(item['value'] ^ 1)][0]
                        pair_margin = float(fact_scores[correct_id] - fact_scores[pair_id])
                        output = decoder.generate(**item['prompt'], **{**spec, 'use_cache': True},
                            max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id,
                            eos_token_id=tok.eos_token_id)
                    tail = output[0, item['prompt']['input_ids'].shape[1]:].tolist()
                    answer = tok.decode(tail, skip_special_tokens=True).strip()
                    rows.append({'group': item['group'], 'name': item['name'], 'condition': condition,
                                 'question': item['question'].format(name=item['name']),
                                 'expected': WORDS[item['value']], 'answer': answer, 'token_ids': tail,
                                 'exact': answer == WORDS[item['value']],
                                 'contains_expected': WORDS[item['value']] in answer,
                                 'forced_fact_logprob': correct_logprob, 'forced_fact_rank': correct_rank,
                                 'forced_fact_pair_margin': pair_margin,
                                 'eos': tok.eos_token_id in tail})
            for question in ('你好，请简单介绍自己。', '请写一句鼓励学习的话。'):
                prompt = p.closed_think_inputs(tok, question, 'cuda')
                with torch.inference_mode():
                    generated = decoder.generate(**prompt, use_cache=True, max_new_tokens=12, do_sample=False,
                        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
                tail = generated[0, prompt['input_ids'].shape[1]:].tolist()
                rows.append({'group': 'ordinary', 'condition': 'off', 'question': question,
                             'answer': tok.decode(tail, skip_special_tokens=True).strip(), 'token_ids': tail})
            return rows

        before = evaluate()
        train = [x for x in cases if x['group'] == 'train']
        trace = []
        for step in range(768):
            if step % len(train) == 0:
                rng.shuffle(train)
            item = train[step % len(train)]
            prefix = item['prompt']['input_ids'].shape[1]
            optimizer.zero_grad(set_to_none=True)
            logits = decoder(input_ids=item['ids'], attention_mask=torch.ones_like(item['ids']),
                             **item['spec']).logits
            target = item['ids'][:, prefix:]
            scores = logits[:, prefix - 1:prefix - 1 + target.shape[1]].float()
            if scores.shape[:2] != target.shape:
                raise ValueError('Incorrect causal alignment')
            loss = torch.nn.functional.cross_entropy(scores.reshape(-1, scores.shape[-1]), target.reshape(-1))
            # Same question, same person, different stored value. Penalize failure to
            # prefer the correct first token under its own memory, relative to the
            # paired alternative; no supplied output logits are used at generation.
            other_value = (item['value'] + 1 + rng.randrange(15)) % 16
            if word_ids[other_value][0] == word_ids[item['value']][0]:
                raise ValueError('Contrast targets must have distinct first tokens')
            other_spec = p._memory_args(model, states[item['key'], other_value],
                                        {'key': item['key'], 'intent': 'current', 'event_time': 0})
            other_logits = decoder(input_ids=item['ids'], attention_mask=torch.ones_like(item['ids']),
                                   **other_spec).logits[:, prefix - 1 + item['value_position'], :].float()
            correct_id, wrong_id = word_ids[item['value']][0], word_ids[other_value][0]
            correct_margin = scores[:, item['value_position'], correct_id] - scores[:, item['value_position'], wrong_id]
            wrong_margin = other_logits[:, correct_id] - other_logits[:, wrong_id]
            # Require the target to beat the alternative under its own memory,
            # and the alternative to beat the target under the swapped memory.
            # A positive difference between margins alone does not ensure either.
            contrast = 0.5 * (torch.nn.functional.softplus(1.0 - correct_margin).mean()
                              + torch.nn.functional.softplus(1.0 + wrong_margin).mean())
            # Full-vocabulary supervision at the actual fact position in both
            # counterfactual memories, not just a relative pairwise objective.
            fact_ce = torch.nn.functional.cross_entropy(scores[:, item['value_position'], :],
                torch.tensor([correct_id], device=device))
            swapped_ce = torch.nn.functional.cross_entropy(other_logits, torch.tensor([wrong_id], device=device))
            loss = 4.0 * loss + 0.5 * (fact_ce + swapped_ce) + contrast
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            trace.append({'step': step + 1, 'loss': float(loss.detach()), 'contrast': float(contrast.detach()),
                          'correct_margin': float(correct_margin.detach()), 'swapped_margin': float(wrong_margin.detach()),
                          'fact_ce': float(fact_ce.detach()), 'swapped_fact_ce': float(swapped_ce.detach()),
                          'grad_norm': float(norm)})
            print(json.dumps(trace[-1]), flush=True)
        after = evaluate()
        if p.frozen_digest(model, chat.state) != original_digest or p.frozen_digest(model, empty) != empty_digest:
            raise ValueError('Protected state changed')
        if any(p.frozen_digest(model, x['state']) != x['digest'] for x in cases):
            raise ValueError('Experimental memory changed')
        reader = {k: v.detach().cpu().clone() for k, v in decoder.reader.state_dict().items()}
        adapter_state = [{'layer': index, 'projection': name, 'a': adapter.a.detach().cpu().clone(),
                          'b': adapter.b.detach().cpu().clone(), 'condition': adapter.condition.detach().cpu().clone()}
                         for index, name, adapter in adapters]
        teacher_sha = p.sha(TEACHER_CANDIDATE)
        candidate = torch.load(TEACHER_CANDIDATE, map_location='cpu', weights_only=True)
        if (candidate['source_sha256'] != source_sha or candidate['base_sha256'] != p.WEIGHT_SHA
                or not candidate.get('adapter_requires_addressed_current_memory')):
            raise ValueError('Teacher candidate source mismatch')
        decoder.reader.load_state_dict(candidate['reader'], strict=True)
        with torch.no_grad():
            for site in decoder.reader.native_sites:
                site.read_out.bias.zero_()
            if len(candidate['adapters']) != len(adapters):
                raise ValueError('Previous adapter count mismatch')
            for (index, name, adapter), old in zip(adapters, candidate['adapters']):
                if old['layer'] != index or old['projection'] != name:
                    raise ValueError('Previous adapter position mismatch')
                for attr in ('a', 'b', 'condition'):
                    current = getattr(adapter, attr)
                    if current.shape != old[attr].shape:
                        raise ValueError('Previous adapter shape mismatch')
                    current.copy_(old[attr])
        teacher_after = evaluate()
        for patcher in slow_patches:
            patcher.stop()
        if p.frozen_digest(model, chat.state) != original_digest:
            raise ValueError('Original state changed during candidate evaluation')
        state_digests = [dict(group=x['group'], key=x['key'], value=WORDS[x['value']], sha256=x['digest']) for x in cases]
    if (p.sha(p.runtime.CHECKPOINT) != source_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA
            or p.sha(TEACHER_CANDIDATE) != teacher_sha):
        raise ValueError('Protected source changed')
    OUT.mkdir(parents=False, exist_ok=False)
    torch.save({'reader': reader, 'source_sha256': source_sha, 'base_sha256': p.WEIGHT_SHA,
                'research_only': True, 'requires_slow_disabled': True, 'requires_zero_output_bias': True,
                'adapters': adapter_state, 'adapter_requires_addressed_current_memory': True,
                'contrast_weight': 1.0, 'answer_ce_weight': 4.0,
                'contrast_kind': 'symmetric_margins_with_paired_fact_ce'}, OUT / 'reader.pt')
    report = {'before': before, 'after': after, 'teacher_after': teacher_after, 'trace': trace, 'states': state_digests,
              'words': dict(zip(WORDS, word_ids)), 'seed': SEED, 'steps': len(trace),
              'source_sha256': source_sha, 'base_sha256': p.WEIGHT_SHA,
              'reader_sha256': p.sha(OUT / 'reader.pt'), 'teacher_candidate_sha256': teacher_sha,
              'seconds': time.perf_counter() - start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Direct controlled memory writes and numeric addresses bypass routing and provenance.',
                              'Small authored split with two values per held dimension, not a broad independent test.',
                              'Slow experts disabled and readout bias fixed to zero in this isolated run only.',
                              'Reader and rank-8 updates to q/v projections of layers 18 and 20 trained for 768 steps.',
                              'Adapters use addressed vectors from both current matrices; no learned validity or identity decision.',
                              'Exact zero matrices make adapter residual zero; actual unknown matrices may be nonzero.',
                              'Previous conditioned candidate reloaded with its reader and all adapter weights.',
                              'Same seed, architecture, data and steps as prior conditioned run; loss weights changed.',
                              'Old candidate also has its slow path disabled and bias removed for evaluation.',
                              'Forced fact scores use the expected answer prefix and are not free generation.',
                              'Codec relies on base embedding and shared random projection; reconstruction is separate from generation.',
                              'Forced memory reads do not test whether the runtime should read or abstain.',
                              'Experimental reader is not adopted by the individual.']}
    with (OUT / 'result.json').open('x', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'seconds': report['seconds'], 'peak_allocated_bytes': report['peak_allocated_bytes'],
                      'after': after}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
