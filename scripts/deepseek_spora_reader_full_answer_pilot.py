"""Train only the existing neural memory reader on full answers in isolation."""
import json
import argparse
import math
import random
import tempfile
import time
from pathlib import Path

from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_named_binding_probe import KEYS, NAMES

OUTPUT = p.ROOT / 'runs/deepseek-spora-reader-full-answer-pilot-20261112'
SEED = 98117
STEPS = 12


def main(output=OUTPUT, learning_rate=1e-4, objective='full', steps=STEPS, replay_weight=0.0):
    output = Path(output).resolve()
    if not output.is_relative_to((p.ROOT / 'runs').resolve()) or output == (p.ROOT / 'runs').resolve():
        raise ValueError('Output must be a new directory under runs')
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1e-3:
        raise ValueError('Invalid learning rate')
    if objective not in ('full', 'value_contrastive', 'mixed'):
        raise ValueError('Invalid objective')
    if type(steps) is not int or not 1 <= steps <= 1000:
        raise ValueError('Invalid step count')
    if not math.isfinite(replay_weight) or not 0 <= replay_weight <= 10 or (replay_weight and objective != 'mixed'):
        raise ValueError('Replay requires mixed objective and weight in [0, 10]')
    if output.exists():
        raise FileExistsError(output)
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
        optimizer = p.torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0, foreach=False)
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
                    value_ids = tokenizer(p.WORDS[value], add_special_tokens=False).input_ids
                    competitor_ids = [tokenizer(p.WORDS[v], add_special_tokens=False).input_ids for v in (1, 3, 4) if v != value]
                    if len(value_ids) != 1 or any(len(ids) != 1 for ids in competitor_ids):
                        raise ValueError('Pilot requires single-token value classes')
                    positions = (target[0] == value_ids[0]).nonzero(as_tuple=True)[0]
                    if positions.numel() != 1:
                        raise ValueError('Target value must occur at exactly one answer position')
                    ids = p.torch.cat((prompt['input_ids'], target), dim=1)
                    spec = p._memory_args(organism, state, {'key': key, 'intent': 'current', 'event_time': 0})
                    cases.append({'group': group, 'key': key, 'name': name, 'value': value, 'state': state,
                                  'state_digest': p.frozen_digest(organism, state), 'question': question,
                                  'answer': answer, 'prompt': prompt, 'ids': ids, 'spec': spec,
                                  'value_position': int(positions[0]), 'value_token': value_ids[0],
                                  'competitor_tokens': [ids[0] for ids in competitor_ids]})

        replays = []
        if replay_weight:
            for index in (2, 3):
                key, name = KEYS[index], NAMES[index]
                for value in (3, 4):
                    state = p.fresh_controlled_state(organism)
                    for fact in (1, value):
                        state, decision = p.observe_sourced(organism, state, key, fact, 'direct', 1.0)
                        if not decision['current_write']:
                            raise ValueError('Replay write rejected')
                    for template in ('{name}当下喜欢什么饮料？', '你知道{name}爱喝什么吗？'):
                        question = template.format(name=name)
                        prompt = p.closed_think_inputs(tokenizer, question, 'cuda')
                        spec = p._memory_args(organism, state, {'key': key, 'intent': 'current', 'event_time': 0})
                        with p.torch.no_grad():
                            teacher = decoder(**prompt, **spec).logits[:, -1, :].float().log_softmax(-1).cpu()
                        replays.append({'question': question, 'key': key, 'value': value, 'prompt': prompt,
                                        'state': state, 'state_digest': p.frozen_digest(organism, state),
                                        'spec': spec, 'teacher': teacher})

        def evaluate():
            rows = []
            for item in cases:
                prefix = item['prompt']['input_ids'].shape[1]
                with p.torch.inference_mode():
                    forced = decoder(input_ids=item['ids'], attention_mask=p.torch.ones_like(item['ids']),
                                     **item['spec'])
                    target = item['ids'][:, prefix:]
                    scores = forced.logits[:, prefix - 1:prefix - 1 + target.shape[1]].float()
                    forced_loss = p.torch.nn.functional.cross_entropy(
                        scores.reshape(-1, scores.shape[-1]), target.reshape(-1))
                    value_logits = scores[0, item['value_position']]
                    value_margin = value_logits[item['value_token']] - value_logits[item['competitor_tokens']].max()
                    first_logits = scores[0, 0]
                    first_margin = first_logits[item['value_token']] - first_logits[item['competitor_tokens']].max()
                    ids = decoder.generate(**item['prompt'], **{**item['spec'], 'use_cache': True},
                        max_new_tokens=24, do_sample=False, pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id)
                tail = ids[0, prefix:].tolist()
                text = tokenizer.decode(tail, skip_special_tokens=True).strip()
                rows.append({'group': item['group'], 'name': item['name'], 'question': item['question'],
                             'expected_value': p.WORDS[item['value']], 'target_sentence': item['answer'],
                             'answer': text, 'token_ids': tail, 'ended_eos': tokenizer.eos_token_id in tail,
                             'teacher_forced_loss': float(forced_loss), 'value_margin': float(value_margin),
                             'first_value_margin': float(first_margin)})
            return rows

        def ordinary_answers():
            rows = []
            for question in ('你好，请简单介绍自己。', '请写一句鼓励学习的话。'):
                inputs = p.closed_think_inputs(tokenizer, question, 'cuda')
                with p.torch.inference_mode():
                    ids = decoder.generate(**inputs, use_cache=True, max_new_tokens=24, do_sample=False,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                tail = ids[0, inputs['input_ids'].shape[1]:].tolist()
                rows.append({'question': question, 'token_ids': tail,
                             'answer': tokenizer.decode(tail, skip_special_tokens=True).strip()})
            return rows

        before = evaluate()
        ordinary_before = ordinary_answers()
        order = [item for item in cases if item['group'] == 'train']
        trace = []
        for step in range(steps):
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
            if objective in ('full', 'mixed'):
                loss = p.torch.nn.functional.cross_entropy(scores.reshape(-1, scores.shape[-1]), targets.reshape(-1))
            else:
                value_logits = scores[:, item['value_position'], :]
                target_id = item['value_token']
                competitors = value_logits[:, item['competitor_tokens']]
                margin_loss = p.torch.nn.functional.softplus(competitors - value_logits[:, target_id:target_id + 1]).mean()
                loss = p.torch.nn.functional.cross_entropy(value_logits, targets[:, item['value_position']]) + margin_loss
            if not bool(p.torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            if objective == 'mixed':
                (loss * 0.5).backward()
                direct_target = p.torch.tensor([[item['value_token'], tokenizer.eos_token_id]], device=device)
                direct_ids = p.torch.cat((item['prompt']['input_ids'], direct_target), dim=1)
                direct = decoder(input_ids=direct_ids, attention_mask=p.torch.ones_like(direct_ids), **item['spec'])
                direct_scores = direct.logits[:, prefix_len - 1:prefix_len + 1].float()
                direct_loss = p.torch.nn.functional.cross_entropy(
                    direct_scores.reshape(-1, direct_scores.shape[-1]), direct_target.reshape(-1))
                if not bool(p.torch.isfinite(direct_loss)):
                    raise ValueError('Nonfinite direct-answer loss')
                (direct_loss * 0.5).backward()
                loss = (loss.detach() + direct_loss.detach()) * 0.5
            else:
                loss.backward()
            replay_loss_value = None
            if replay_weight:
                replay = replays[step % len(replays)]
                replay_logits = decoder(**replay['prompt'], **replay['spec']).logits[:, -1, :].float()
                replay_loss = p.torch.nn.functional.kl_div(replay_logits.log_softmax(-1),
                    replay['teacher'].to(device), log_target=True, reduction='batchmean')
                if not bool(p.torch.isfinite(replay_loss)):
                    raise ValueError('Nonfinite replay loss')
                (replay_weight * replay_loss).backward()
                replay_loss_value = float(replay_loss.detach())
            grad = p.torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            trace.append({'step': step + 1, 'loss': float(loss.detach()), 'gradient_norm': float(grad),
                          'replay_kl': replay_loss_value})
            print(json.dumps(trace[-1]), flush=True)
        after = evaluate()
        ordinary_after = ordinary_answers()
        if p.frozen_digest(organism, chat.state) != original_digest:
            raise ValueError('Original state changed')
        if any(p.frozen_digest(organism, item['state']) != item['state_digest'] for item in cases):
            raise ValueError('Temporary state changed')
        if any(p.frozen_digest(organism, item['state']) != item['state_digest'] for item in replays):
            raise ValueError('Replay state changed')
        reader_state = {key: value.detach().cpu().clone() for key, value in decoder.reader.state_dict().items()}
    if p.sha(p.runtime.CHECKPOINT) != source_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected source changed')
    output.mkdir(parents=False, exist_ok=False)
    p.torch.save({'state': reader_state, 'source_checkpoint_sha256': source_sha,
                  'base_sha256': p.WEIGHT_SHA, 'steps': steps, 'seed': SEED,
                  'learning_rate': learning_rate, 'objective': objective, 'replay_weight': replay_weight}, output / 'reader.pt')
    report = {'before': before, 'after': after, 'trace': trace, 'source_checkpoint_sha256': source_sha,
              'ordinary_before': ordinary_before, 'ordinary_after': ordinary_after,
              'ordinary_tokens_match': [a['token_ids'] == b['token_ids'] for a, b in zip(ordinary_before, ordinary_after)],
              'learning_rate': learning_rate, 'objective': objective, 'replay_weight': replay_weight,
              'replay_examples': [{'question': r['question'], 'key': r['key'], 'value': r['value']} for r in replays],
              'base_sha256': p.WEIGHT_SHA,
              'candidate_sha256': p.sha(output / 'reader.pt'),
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': p.torch.cuda.max_memory_allocated(),
              'limitations': ['Four training examples, two held examples; two seen values and artificial facts.',
                              'Teacher forcing with no output bias; not normal chat training.',
                              'Address and fact supplied numerically; no routing or unknown-memory evaluation.',
                              'Only two ordinary prompts tested without any memory; no evaluation with unrelated memory.',
                              'Candidate is isolated and is not adopted by default.',
                              'This pilot tests feasibility, not independent generalization.']}
    with (output / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'steps': len(trace), 'seconds': report['seconds'],
                      'peak_allocated_bytes': report['peak_allocated_bytes']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--objective', choices=('full', 'value_contrastive', 'mixed'), default='full')
    parser.add_argument('--steps', type=int, default=STEPS)
    parser.add_argument('--replay-weight', type=float, default=0.0)
    options = parser.parse_args()
    main(options.output, options.learning_rate, options.objective, options.steps, options.replay_weight)
