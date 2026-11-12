"""Isolated experiment: generate short text from continuous fast-memory vectors.

Synthetic numeric addresses and direct writes are deliberate experimental controls.
This is not a natural-language write path or a replacement for the active reader.
"""
import json
import random
import tempfile
import time
from pathlib import Path

from scripts import deepseek_spora_long_memory_generate_parity as p
from spora.continuous_text_memory import ContinuousTextCodec
from spora.experimental_write_gradient import controlled_write_target_gradient

OUT = p.ROOT / 'runs/continuous-reader-learned-codec-pilot-20261112'
SEED = 72381
WORDS = ('苹果', '香蕉', '西瓜', '葡萄', '草莓', '橘子', '桃子', '芒果', '梨子', '柠檬')


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
        projection = torch.nn.Parameter(codec.projection.detach().clone())
        embedding = chat.base.model.embed_tokens.weight.detach()
        params = [x for x in decoder.reader.parameters() if x.requires_grad] + [projection]
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0, foreach=False)
        mode = torch.zeros(1, dtype=torch.long, device=device)
        gate = torch.ones(1, device=device)

        def encode(value):
            indices = torch.tensor(word_ids[value] + [tok.eos_token_id], dtype=torch.long, device=device)
            vec = torch.nn.functional.normalize(embedding.index_select(0, indices).float() @ projection, dim=-1)
            vec = torch.nn.functional.pad(vec, (0, 0, 0, 4 - vec.shape[0])).reshape(1, -1)
            return torch.nn.functional.pad(vec, (0, model.config.memory_width - vec.shape[1]))

        def state_for(value, key, encoded=None):
            state = p.fresh_controlled_state(model)
            fast, velocity = model.initial_state(1)
            for site in fast:
                site[0] = torch.zeros_like(site[0])
            if value is not None:
                target = encode(value) if encoded is None else encoded
                with torch.no_grad():
                    for i, site in enumerate(model.memories):
                        fast[i], velocity[i] = site.write_temporal(
                            fast[i], velocity[i], site.current_basis[key:key + 1].to(device),
                            target, gate, mode, False)
                        read = site.retrieve([fast[i][0]], site.current_basis[key:key + 1].to(device))
                        expected = torch.sigmoid(site.log_lr) * target.detach()
                        if not torch.allclose(read, expected, atol=2e-5, rtol=2e-5):
                            raise ValueError('Controlled write does not reconstruct target')
            state.current = [x[0].detach() for x in fast]
            return state

        cases = []
        # Each held split changes exactly one dimension from a training combination.
        train_questions = ('请只说出{name}喜欢的水果名称。', '记录里{name}喜欢哪种水果？只写名称。')
        schedule = [('train', ((0, '小林'), (1, '小周')), range(8), q) for q in train_questions]
        schedule += [
            ('held_name', ((6, '小陈'),), (0, 1), train_questions[0]),
            ('held_question', ((0, '小林'),), (0, 1), '{name}爱吃什么水果？只回答名字。'),
            ('held_word', ((0, '小林'),), (8, 9), train_questions[0]),
        ]
        states = {}
        for group, names, values, question in schedule:
            for key, name in names:
                for value in values:
                    if (key, value) not in states:
                        states[key, value] = state_for(value, key)
                    state = states[key, value]
                    prompt = p.closed_think_inputs(tok, question.format(name=name), 'cuda')
                    target = torch.tensor([word_ids[value] + [tok.eos_token_id]], device=device)
                    ids = torch.cat((prompt['input_ids'], target), dim=1)
                    spec = p._memory_args(model, state, {'key': key, 'intent': 'current', 'event_time': 0})
                    cases.append(dict(group=group, name=name, key=key, value=value, question=question, state=state,
                                      prompt=prompt, ids=ids, spec=spec,
                                      digest=p.frozen_digest(model, state)))
        empty = state_for(None, 0)
        empty_digest = p.frozen_digest(model, empty)

        def evaluate():
            rows = []
            # A fixed question/name with different values makes copying the prompt insufficient.
            subset = [x for x in cases if (x['group'] != 'train' or
                      x['key'] == 0 and x['value'] < 2 and x['question'] == train_questions[0])]
            for item in subset:
                fresh_state = state_for(item['value'], item['key'])
                fresh_spec = p._memory_args(model, fresh_state,
                    {'key': item['key'], 'intent': 'current', 'event_time': 0})
                conditions = [('memory', fresh_spec)]
                if (item['group'], item['value']) in (('train', 0), ('held_word', 8)):
                    conditions += [('empty', p._memory_args(model, empty,
                        {'key': item['key'], 'intent': 'current', 'event_time': 0})), ('off', {})]
                for condition, spec in conditions:
                    with torch.inference_mode():
                        output = decoder.generate(**item['prompt'], **{**spec, 'use_cache': True},
                            max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id,
                            eos_token_id=tok.eos_token_id)
                    tail = output[0, item['prompt']['input_ids'].shape[1]:].tolist()
                    answer = tok.decode(tail, skip_special_tokens=True).strip()
                    rows.append({'group': item['group'], 'name': item['name'], 'condition': condition,
                                 'question': item['question'].format(name=item['name']),
                                 'expected': WORDS[item['value']], 'answer': answer, 'token_ids': tail,
                                 'exact': answer == WORDS[item['value']], 'eos': tok.eos_token_id in tail})
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
        for step in range(128):
            if step % len(train) == 0:
                rng.shuffle(train)
            item = train[step % len(train)]
            prefix = item['prompt']['input_ids'].shape[1]
            optimizer.zero_grad(set_to_none=True)
            target_code = encode(item['value'])
            state = state_for(item['value'], item['key'], target_code)
            spec = p._memory_args(model, state, {'key': item['key'], 'intent': 'current', 'event_time': 0})
            addresses = [site.current_basis[item['key']:item['key']+1].to(device) for site in model.memories]
            with controlled_write_target_gradient(decoder.reader, model.memories, target_code,
                                                  addresses, addresses, [gate, gate]):
                logits = decoder(input_ids=item['ids'], attention_mask=torch.ones_like(item['ids']),
                                 **spec).logits
            target = item['ids'][:, prefix:]
            scores = logits[:, prefix - 1:prefix - 1 + target.shape[1]].float()
            if scores.shape[:2] != target.shape:
                raise ValueError('Incorrect causal alignment')
            loss = torch.nn.functional.cross_entropy(scores.reshape(-1, scores.shape[-1]), target.reshape(-1))
            # Same question, same person, different stored value. Penalize failure to
            # prefer the correct first token under its own memory, relative to the
            # paired alternative; no supplied output logits are used at generation.
            other_value = (item['value'] + 1 + rng.randrange(7)) % 8
            if word_ids[other_value][0] == word_ids[item['value']][0]:
                raise ValueError('Contrast targets must have distinct first tokens')
            other_code = encode(other_value)
            other_state = state_for(other_value, item['key'], other_code)
            other_spec = p._memory_args(model, other_state,
                                        {'key': item['key'], 'intent': 'current', 'event_time': 0})
            with controlled_write_target_gradient(decoder.reader, model.memories, other_code,
                                                  addresses, addresses, [gate, gate]):
                other_logits = decoder(**item['prompt'], **other_spec).logits[:, -1, :].float()
            correct_id, wrong_id = word_ids[item['value']][0], word_ids[other_value][0]
            correct_margin = scores[:, 0, correct_id] - scores[:, 0, wrong_id]
            wrong_margin = other_logits[:, correct_id] - other_logits[:, wrong_id]
            contrast = torch.nn.functional.softplus(1.0 - (correct_margin - wrong_margin)).mean()
            loss = loss + contrast
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            trace.append({'step': step + 1, 'loss': float(loss.detach()), 'contrast': float(contrast.detach()),
                          'encoder_grad_norm': float(projection.grad.norm()),
                          'grad_norm': float(norm)})
            print(json.dumps(trace[-1]), flush=True)
        after = evaluate()
        if p.frozen_digest(model, chat.state) != original_digest or p.frozen_digest(model, empty) != empty_digest:
            raise ValueError('Protected state changed')
        if any(p.frozen_digest(model, x['state']) != x['digest'] for x in cases):
            raise ValueError('Experimental memory changed')
        reader = {k: v.detach().cpu().clone() for k, v in decoder.reader.state_dict().items()}
        learned_projection = projection.detach().cpu().clone()
        state_digests = [dict(group=x['group'], key=x['key'], value=WORDS[x['value']], sha256=x['digest']) for x in cases]
    if p.sha(p.runtime.CHECKPOINT) != source_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected source changed')
    OUT.mkdir(parents=False, exist_ok=False)
    torch.save({'reader': reader, 'projection': learned_projection,
                'source_sha256': source_sha, 'base_sha256': p.WEIGHT_SHA}, OUT / 'reader.pt')
    report = {'before': before, 'after': after, 'trace': trace, 'states': state_digests,
              'words': dict(zip(WORDS, word_ids)), 'seed': SEED, 'steps': len(trace),
              'source_sha256': source_sha, 'base_sha256': p.WEIGHT_SHA,
              'reader_sha256': p.sha(OUT / 'reader.pt'), 'seconds': time.perf_counter() - start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Direct controlled memory writes and numeric addresses bypass routing and provenance.',
                              'Small authored split with two values per held dimension, not a broad independent test.',
                              'Projection is trained with an analytic gradient valid only for one controlled delta write.',
                              'Codec relies on the base embedding; no trainable language model or natural-language write path.',
                              'Forced memory reads do not test whether the runtime should read or abstain.',
                              'Experimental reader is not adopted by the individual.']}
    with (OUT / 'result.json').open('x', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'seconds': report['seconds'], 'peak_allocated_bytes': report['peak_allocated_bytes'],
                      'after': after}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
