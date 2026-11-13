"""Isolated writer-only optimization with frozen-writer and held-fact controls."""
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_bridge_reload_audit as audit
from spora.continuous_text_memory import ContinuousTextCodec
from spora.research_differentiable_memory import DifferentiablePacketWriter, differentiable_memory_context

torch, p = audit.torch, audit.p
audit.SOURCE = ROOT / 'runs/continuous-reader-cross-attention-pilot-20261113'
audit.OUT = ROOT / 'runs/deepseek-spora-differentiable-writer-pilot-20261113'
audit.EXTRA_ONLY = True
audit.PARITY_VALUE_INDICES = (0, 1, 3, 9, 16, 17)
TRAIN = ('苹果', '香蕉', '西瓜', '葡萄', '草莓', '橘子', '桃子', '芒果')
HELD = ('桑葚', '枇杷', '莲雾', '蛇果')
STEPS = 64
done = False


def observe(*, decoder, tokenizer, spec, cross_modules, group, name, expected, question):
    global done
    if done or expected != '苹果':
        return
    if len(cross_modules) != 1:
        raise ValueError('Unexpected cross-attention layout')
    saved_flag = cross_modules[0].enabled
    cross_modules[0].enabled = False
    try:
        _run(decoder, tokenizer, spec, name, question)
        done = True
    finally:
        cross_modules[0].enabled = saved_flag


def _run(decoder, tokenizer, spec, name, question):
    randomizer = random.Random(37261)
    torch.manual_seed(37261)
    addresses = spec['memory_addresses']
    originals = [x.detach() for x in spec['memory_state'].current]
    embedding = decoder.base.model.embed_tokens.weight.detach()
    codec = ContinuousTextCodec(embedding.cpu(), memory_width=90, slot_count=5,
                                seed=29071, eos_token_id=tokenizer.eos_token_id).to('cuda')
    writer = DifferentiablePacketWriter(embedding.shape[1], 90).to('cuda')
    torch.nn.init.zeros_(writer.packet.weight)
    optimizer = torch.optim.AdamW(writer.parameters(), lr=0.002, weight_decay=0, foreach=False)
    prompt = p.closed_think_inputs(tokenizer, question, 'cuda')
    suffix = tokenizer.encode(f'{name}喜欢的水果是', add_special_tokens=False)
    prefix = torch.cat((prompt['input_ids'], torch.tensor([suffix], device='cuda')), dim=1)
    identity_id = tokenizer.encode(name, add_special_tokens=False)[-1]
    data = {}
    for value in TRAIN + HELD:
        ids = tokenizer.encode(value, add_special_tokens=False)
        if not 1 <= len(ids) <= 3:
            raise ValueError('Fact exceeds the controlled packet capacity')
        with torch.no_grad():
            packet = codec.encode(torch.tensor([identity_id, *ids], device='cuda')).reshape(1, -1)
            states = []
            for matrix, address in zip(originals, addresses):
                key = torch.nn.functional.normalize(address.float(), dim=-1)
                old = torch.bmm(matrix.float(), key.unsqueeze(-1)).squeeze(-1)
                # Preserve the current writer's scale at this address.
                scale = old.reshape(1, 5, 18)[:, 0].norm(dim=-1, keepdim=True)
                states.append(matrix.float() + (packet * scale - old).unsqueeze(-1) * key.unsqueeze(1))
            observation = embedding[ids[0]].float().unsqueeze(0)
            mask = torch.zeros((1, 5, 18), dtype=torch.bool, device='cuda')
            mask[:, 1:1 + len(ids)] = True
            data[value] = {'ids': ids, 'matrices': states, 'observation': observation,
                           'mask': mask.reshape(1, 90)}

    def matrices_for(value, *, trained):
        item = data[value]
        if not trained:
            return item['matrices']
        return [writer(matrix, address, item['observation'], active_mask=item['mask'], residual=True)
                for matrix, address in zip(item['matrices'], addresses)]

    def fixed_prefix(value, *, trained, grad=False):
        matrices = matrices_for(value, trained=trained)
        with torch.set_grad_enabled(grad), differentiable_memory_context(decoder, matrices, addresses):
            logits = decoder.base(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                                  use_cache=False, logits_to_keep=1).logits[0, -1].float()
            loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0),
                  torch.tensor([data[value]['ids'][0]], device='cuda'))
            if grad:
                return loss
            target = data[value]['ids'][0]
            return {'loss': float(loss), 'rank': int((logits > logits[target]).sum()) + 1,
                    'top_token': int(logits.argmax())}

    def generate(value, *, trained):
        with torch.no_grad():
            matrices = matrices_for(value, trained=trained)
            with differentiable_memory_context(decoder, matrices, addresses):
                output = decoder.base.generate(**prompt, use_cache=True, max_new_tokens=24,
                    do_sample=False, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id)
        tail = output[0, prompt['input_ids'].shape[1]:].tolist()
        answer = tokenizer.decode(tail, skip_special_tokens=True).strip()
        return {'answer': answer, 'token_ids': tail, 'correct': audit.correct_answer(answer, name, value)}

    def evaluate(trained):
        return {value: {'forced': fixed_prefix(value, trained=trained),
                        'generated': generate(value, trained=trained)} for value in TRAIN + HELD}

    before = evaluate(False)
    initial_trained = evaluate(True)
    if before != initial_trained:
        raise ValueError('Zero writer did not reproduce the fixed writer')
    trace = []
    schedule = list(TRAIN)
    for step in range(STEPS):
        if step % len(schedule) == 0:
            randomizer.shuffle(schedule)
        value = schedule[step % len(schedule)]
        optimizer.zero_grad(set_to_none=True)
        loss = fixed_prefix(value, trained=True, grad=True)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(writer.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % 8 == 0:
            row = {'step': step + 1, 'loss': float(loss.detach()), 'gradient_l2': float(norm)}
            trace.append(row)
            print(json.dumps(row), flush=True)
        if (step + 1) % 16 == 0:
            path = audit.OUT / 'writer-recovery.pt'
            temp = audit.OUT / 'writer-recovery.tmp'
            with temp.open('wb') as stream:
                torch.save({'step': step + 1, 'writer': writer.state_dict(),
                            'optimizer': optimizer.state_dict(), 'schedule': schedule,
                            'rng': randomizer.getstate()}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
    after = evaluate(True)
    fixed_after = evaluate(False)
    if fixed_after != before:
        raise ValueError('Fixed-writer control changed during optimization')
    report = {'before': before, 'after': after, 'fixed_after': fixed_after, 'trace': trace,
              'train_values': TRAIN, 'held_values': HELD, 'steps': STEPS,
              'writer_weight_l2': float(writer.packet.weight.detach().double().norm()),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['One name, one question and supervised fact-prefix next-token loss.',
                              'Frozen reader and base; writer sees first value token embedding.',
                              'Direct factual packets, EOS and active-slot boundaries are supplied externally.',
                              'Held words previously used in development, not an independent acceptance set.',
                              'Zero writer baseline has no matched optimizer compute; no natural routing or long-term updates.']}
    (audit.OUT / 'writer-result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'writer_result_saved': True}), flush=True)


audit.PROBE_OBSERVER = observe
if __name__ == '__main__':
    audit.main()
    if not done:
        raise ValueError('Writer pilot did not run')
