"""Train only the existing neural memory reader on full answers in isolation."""
import json
import random
import tempfile
import time
from pathlib import Path

from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_named_binding_probe import KEYS, NAMES

OUTPUT = p.ROOT / 'runs/deepseek-spora-reader-full-answer-pilot-20261112'
SEED = 98117
STEPS = 12


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_sha = p.sha(p.runtime.CHECKPOINT)
    p.torch.manual_seed(SEED)
    rng = random.Random(SEED)
    p.torch.set_num_threads(2)
    p.torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    with tempfile.TemporaryDirectory() as directory:
        p.runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        decoder, organism = chat.decoder.eval(), chat.organism.eval()
        if any(x.is_meta for x in list(chat.base.parameters()) + list(chat.base.buffers())):
            raise ValueError('Unmaterialized layer')
        original_digest = p.frozen_digest(organism, chat.state)
        decoder.requires_grad_(False)
        decoder.reader.requires_grad_(True)
        parameters = [x for x in decoder.reader.parameters() if x.requires_grad]
        if not parameters:
            raise ValueError('No trainable reader parameters')
        optimizer = p.torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.0, foreach=False)
        device = next(organism.parameters()).device
        cases = []
        for group, indices in (('train', (0, 1)), ('held', (6,))):
            for index in indices:
                key, name = KEYS[index], NAMES[index]
                for value in (3, 4):
                    state = p.fresh_controlled_state(organism)
                    for fact in (1, value):
                        state, decision = p.observe_sourced(organism, state, key, fact, 'direct', 1.0)
                        if not decision['current_write']:
                            raise ValueError('Controlled write rejected')
                    keys = p.torch.tensor([key], dtype=p.torch.long, device=device)
                    with p.torch.no_grad():
                        logits = p.query(organism, state, keys, p.torch.zeros_like(keys), p.torch.zeros_like(keys))
                    if int(logits.argmax(-1)[0]) != value:
                        raise ValueError('Native evidence mismatch')
                    template = ('请说出{name}现在爱喝的饮品。' if group == 'train' else
                                '提到饮料时，{name}如今会选什么？')
                    question = template.format(name=name)
                    answer = f'{name}现在喜欢喝{p.WORDS[value]}。'
                    prompt = p.closed_think_inputs(tokenizer, question, 'cuda')
                    target = tokenizer(answer, add_special_tokens=False, return_tensors='pt').input_ids.to('cuda')
                    target = p.torch.cat((target, p.torch.tensor([[tokenizer.eos_token_id]], device=device)), dim=1)
                    ids = p.torch.cat((prompt['input_ids'], target), dim=1)
                    spec = p._memory_args(organism, state, {'key': key, 'intent': 'current', 'event_time': 0})
                    cases.append({'group': group, 'key': key, 'name': name, 'value': value, 'state': state,
                                  'state_digest': p.frozen_digest(organism, state), 'question': question,
                                  'answer': answer, 'prompt': prompt, 'ids': ids, 'spec': spec})

        def evaluate():
            rows = []
            for item in cases:
                prefix = item['prompt']['input_ids'].shape[1]
                with p.torch.inference_mode():
                    ids = decoder.generate(**item['prompt'], **{**item['spec'], 'use_cache': True},
                        max_new_tokens=24, do_sample=False, pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id)
                tail = ids[0, prefix:].tolist()
                text = tokenizer.decode(tail, skip_special_tokens=True).strip()
                rows.append({'group': item['group'], 'name': item['name'], 'question': item['question'],
                             'expected_value': p.WORDS[item['value']], 'target_sentence': item['answer'],
                             'answer': text, 'token_ids': tail, 'ended_eos': tokenizer.eos_token_id in tail})
            return rows

        before = evaluate()
        order = [item for item in cases if item['group'] == 'train']
        trace = []
        for step in range(STEPS):
            if step % len(order) == 0:
                rng.shuffle(order)
            item = order[step % len(order)]
            prefix_len = item['prompt']['input_ids'].shape[1]
            target_len = item['ids'].shape[1] - prefix_len
            optimizer.zero_grad(set_to_none=True)
            result = decoder(input_ids=item['ids'], attention_mask=p.torch.ones_like(item['ids']),
                             **item['spec'])
            scores = result.logits[:, prefix_len - 1:prefix_len - 1 + target_len].float()
            targets = item['ids'][:, prefix_len:]
            if scores.shape[:2] != targets.shape:
                raise ValueError('Causal answer alignment mismatch')
            loss = p.torch.nn.functional.cross_entropy(scores.reshape(-1, scores.shape[-1]), targets.reshape(-1))
            if not bool(p.torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            loss.backward()
            grad = p.torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            trace.append({'step': step + 1, 'loss': float(loss.detach()), 'gradient_norm': float(grad)})
            print(json.dumps(trace[-1]), flush=True)
        after = evaluate()
        if p.frozen_digest(organism, chat.state) != original_digest:
            raise ValueError('Original state changed')
        if any(p.frozen_digest(organism, item['state']) != item['state_digest'] for item in cases):
            raise ValueError('Temporary state changed')
        reader_state = {key: value.detach().cpu().clone() for key, value in decoder.reader.state_dict().items()}
    if p.sha(p.runtime.CHECKPOINT) != source_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected source changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    p.torch.save({'state': reader_state, 'source_checkpoint_sha256': source_sha,
                  'base_sha256': p.WEIGHT_SHA, 'steps': STEPS, 'seed': SEED}, OUTPUT / 'reader.pt')
    report = {'before': before, 'after': after, 'trace': trace, 'source_checkpoint_sha256': source_sha,
              'base_sha256': p.WEIGHT_SHA, 'candidate_sha256': p.sha(OUTPUT / 'reader.pt'),
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': p.torch.cuda.max_memory_allocated(),
              'limitations': ['Four training examples, two held examples; two seen values and artificial facts.',
                              'Full-answer teacher forcing with no output bias; not normal chat training.',
                              'Address and fact supplied numerically; no routing, unknown-memory or ordinary-chat evaluation.',
                              'Candidate is isolated and is not adopted by default.',
                              'This pilot tests feasibility, not independent generalization.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'steps': len(trace), 'seconds': report['seconds'],
                      'peak_allocated_bytes': report['peak_allocated_bytes']}), flush=True)


if __name__ == '__main__':
    main()
