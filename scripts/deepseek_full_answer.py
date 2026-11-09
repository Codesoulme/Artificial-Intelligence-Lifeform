"""Train DeepSeek memory readout on complete numeric replies in a closed-think diagnostic."""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, REVISION, WEIGHT_SHA
from scripts.open_base_addressed_alignment import QUERY_PATH, QUERY_SHA, neural_addresses, TRAIN_PROMPTS, HELD_PROMPTS
from scripts.open_base_addressed_full_answer import create_snapshots, DEV_VALUES
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.open_base_real_memory_alignment import TRAIN_PAIRS, HELD_PAIRS
from spora.growing import load
from spora.language_training import frozen_digest
from spora.open_base_memory import OpenBaseMemory, memory_hooks


REASONING_TEXTS = ('我要读取该对象的当前值。\n', '我需要确认记录的当前数值。\n', '根据记录回答这个问题。\n')
DIVERSE_TRAIN_PAIRS = TRAIN_PAIRS + ((12, 13), (14, 15), (16, 17), (18, 19),
                                     (40, 41), (42, 43), (44, 45), (46, 47))


def thinking_inputs(tokenizer, question, device):
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': question}], tokenize=False,
                                            add_generation_prompt=True)
    if '<think>' not in prompt or '</think>' in prompt:
        raise ValueError('Unexpected DeepSeek generation template')
    return tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)


def answer_batch(tokenizer, question, answers, device, reasoning_text=None):
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None or not answers:
        raise ValueError('Missing answers, padding token or EOS')
    prefix_inputs = (closed_think_inputs(tokenizer, question, device) if reasoning_text is None else
                     thinking_inputs(tokenizer, question, device))
    prefix = prefix_inputs['input_ids'][0].tolist()
    endings = [tokenizer.encode((reasoning_text + '</think>\n\n' if reasoning_text is not None else '') + str(value),
                                add_special_tokens=False) for value in answers]
    if any(not tokens for tokens in endings):
        raise ValueError('Empty answer')
    sequences = [prefix + tokens + [tokenizer.eos_token_id] for tokens in endings]
    length = max(map(len, sequences))
    ids = torch.full((len(sequences), length), tokenizer.pad_token_id, dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    for i, seq in enumerate(sequences):
        ids[i, :len(seq)] = torch.tensor(seq, device=device)
        labels[i, len(prefix):len(seq)] = ids[i, len(prefix):len(seq)]
        attention[i, :len(seq)] = 1
    return {'input_ids': ids, 'attention_mask': attention}, labels


@torch.no_grad()
def evaluate(model, layers, adapter, tokenizer, reader, organism, snapshots, mode, reasoning=False):
    rows = []
    empty_cache = {}
    for pair in HELD_PAIRS:
        for pos, key in enumerate(pair):
            for template in (HELD_PROMPTS[:1] if reasoning else HELD_PROMPTS):
                question = template.format(key=key)
                addresses, parsed = neural_addresses(reader, organism, question)
                if parsed['entity'] != key:
                    raise ValueError('Wrong neural address')
                for value in ((1, 2, 7, 15) if reasoning else DEV_VALUES):
                    state = snapshots[(pair, value)]
                    if mode == 'swapped':
                        state = SimpleNamespace(current=[x.flip(0) for x in state.current], pages=[])
                    elif mode == 'empty':
                        state = SimpleNamespace(current=[torch.zeros_like(x) for x in state.current], pages=[])
                    expected = ([value, (value + 1) % 16] if pos == 0 else [(value + 1) % 16, value])
                    if mode == 'swapped':
                        expected.reverse()
                    if mode == 'empty' and question in empty_cache:
                        response = empty_cache[question]
                    else:
                        inputs = (thinking_inputs(tokenizer, question, 'cuda') if reasoning else
                                  closed_think_inputs(tokenizer, question, 'cuda'))
                        inputs = {k: v.expand(2, -1) for k, v in inputs.items()}
                        with memory_hooks(layers, adapter, state, current_only=True, addresses=addresses):
                            output = model.generate(**inputs, max_new_tokens=64 if reasoning else 8, do_sample=False,
                                                    pad_token_id=tokenizer.eos_token_id)
                        response = [tokenizer.decode(row[inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                                    for row in output]
                        if mode == 'empty':
                            empty_cache[question] = response
                    final = [(text.split('</think>', 1)[1].strip() if '</think>' in text else None)
                             for text in response] if reasoning else response
                    correct = [text in (str(target), str(target) + '.', str(target) + '。')
                               for text, target in zip(final, expected)]
                    rows.append({'pair': list(pair), 'key': key, 'question': question, 'seed_value': value,
                                 'expected': expected, 'responses': response, 'final_answers': final, 'correct': correct})
    return {'pairs': len(rows), 'both_correct': sum(all(r['correct']) for r in rows),
            'examples': 2 * len(rows), 'correct': sum(sum(r['correct']) for r in rows), 'rows': rows}


def run(output, steps, initial=None, reasoning=False, diverse=False):
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT / 'runs').resolve()) or output == (ROOT / 'runs').resolve() or output.exists():
        raise ValueError('Output must be a new directory under runs')
    if type(steps) is not int or not 1 <= steps <= 4000:
        raise ValueError('Invalid step count')
    initial = Path(initial).resolve() if initial is not None else None
    if initial is not None and not initial.is_relative_to((ROOT / 'runs').resolve()):
        raise ValueError('Initial adapter must be under runs')
    initial_sha = sha(initial) if initial is not None else None
    def verify():
        return (sha(MODEL / 'model.safetensors') == WEIGHT_SHA and sha(INDIVIDUAL) == INDIVIDUAL_SHA
                and sha(QUERY_PATH) == QUERY_SHA)
    if not verify():
        raise ValueError('Protected source mismatch')
    torch.manual_seed(4604)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    reader, reader_state, _ = load(QUERY_PATH, 'cpu')
    if not hasattr(reader, 'chinese_query') or any(not torch.equal(a.current_basis, b.current_basis)
            for a, b in zip(organism.memories, reader.memories)):
        raise ValueError('Incompatible neural addresses')
    before, reader_before = frozen_digest(organism, original), frozen_digest(reader, reader_state)
    train_pairs = DIVERSE_TRAIN_PAIRS if diverse else TRAIN_PAIRS
    snapshots, actions = create_snapshots(organism, original, train_pairs + HELD_PAIRS)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    adapter = OpenBaseMemory(model.config.hidden_size, organism.config.memory_width).to('cuda')
    if initial is not None:
        checkpoint = torch.load(initial, map_location='cpu', weights_only=True)
        if (checkpoint.get('format') not in ('deepseek-spora-closed-think-full-answer-v1',
                                              'deepseek-spora-reasoning-full-answer-v1')
                or checkpoint.get('weight_sha256') != WEIGHT_SHA
                or checkpoint.get('individual_sha256') != INDIVIDUAL_SHA
                or checkpoint.get('query_sha256') != QUERY_SHA or checkpoint.get('layers') != [5, 17]
                or checkpoint.get('current_only') is not True):
            raise ValueError('Initial adapter provenance mismatch')
        adapter.load_state_dict(checkpoint['state'], strict=True)
    layers = [model.model.layers[5], model.model.layers[17]]
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.001, weight_decay=0.)
    tasks = [(pair, pos, template, value) for pair in train_pairs for pos in range(2)
             for template in TRAIN_PROMPTS for value in range(16)]
    generator = torch.Generator().manual_seed(4604)
    order = torch.randperm(len(tasks), generator=generator).tolist()
    log = []
    for step in range(steps):
        if step and step % len(order) == 0:
            order = torch.randperm(len(tasks), generator=generator).tolist()
        pair, pos, template, value = tasks[order[step % len(order)]]
        question = template.format(key=pair[pos])
        addresses, parsed = neural_addresses(reader, organism, question)
        if parsed['entity'] != pair[pos]:
            raise ValueError('Wrong training address')
        answers = [value, (value + 1) % 16] if pos == 0 else [(value + 1) % 16, value]
        inputs, labels = answer_batch(tokenizer, question, answers, 'cuda',
                                      REASONING_TEXTS[step % len(REASONING_TEXTS)] if reasoning else None)
        optimizer.zero_grad(set_to_none=True)
        with memory_hooks(layers, adapter, snapshots[(pair, value)], current_only=True, addresses=addresses):
            logits = model(**inputs, use_cache=False).logits[:, :-1, :].float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == steps:
            entry = {'step': step + 1, 'loss': float(loss.detach()), 'gradient_norm': float(norm)}
            log.append(entry)
            print(json.dumps(entry), flush=True)
    adapter.eval()
    if (any(p.grad is not None for p in model.parameters()) or frozen_digest(organism, original) != before
            or frozen_digest(reader, reader_state) != reader_before or not verify()
            or (initial is not None and sha(initial) != initial_sha)):
        raise ValueError('Protected inputs changed')
    output.mkdir(parents=False, exist_ok=False)
    snapshot = output / 'adapter.pt'
    torch.save({'format': ('deepseek-spora-reasoning-full-answer-v1' if reasoning else
                           'deepseek-spora-closed-think-full-answer-v1'),
                'state': {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA,
                'query_sha256': QUERY_SHA, 'layers': [5, 17], 'current_only': True}, snapshot)
    print('adapter checkpoint saved; evaluation starting', flush=True)
    results = {}
    for mode in ('normal', 'swapped', 'empty'):
        results[mode] = evaluate(model, layers, adapter, tokenizer, reader, organism, snapshots, mode, reasoning)
        print(json.dumps({'mode': mode, 'correct': results[mode]['correct'],
                          'examples': results[mode]['examples']}), flush=True)
    if (frozen_digest(organism, original) != before or frozen_digest(reader, reader_state) != reader_before
            or not verify() or (initial is not None and sha(initial) != initial_sha)):
        raise ValueError('Protected inputs changed during evaluation')
    report = {'status': 'controlled_full_answer_diagnostic_only', 'revision': REVISION,
              'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA, 'query_sha256': QUERY_SHA,
              'adapter_sha256': sha(snapshot), 'initial_adapter_sha256': initial_sha,
              'optimizer_restarted': initial is not None, 'reasoning_format': reasoning,
              'train_pairs': train_pairs, 'held_pairs': HELD_PAIRS,
              'steps': steps, 'log': log, 'results': results,
              'action_counts': {action: actions.count(action) for action in sorted(set(actions))},
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': [('Short fixed reasoning text is supervised; no multi-step reasoning is tested.' if reasoning else
                               'Thinking is explicitly closed before the answer; no reasoning is tested.'),
                              'P26 is a separately trained finite numbered-question reader.',
                              'All states share one original background and two controlled writes.',
                              'All values from 0 to 15 were included in training.',
                              'P26 may have seen the held keys and prompts.',
                              'Swapped conditions reuse the same two states in reverse order.',
                              'Empty-state responses are cached by question and are not independent samples.',
                              'Exact numeric scoring may reject correct full-sentence answers.']}
    with (output / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({mode: {k: v for k, v in result.items() if k != 'rows'}
                      for mode, result in results.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial', type=Path)
    parser.add_argument('--reasoning-format', action='store_true')
    parser.add_argument('--diverse', action='store_true')
    args = parser.parse_args()
    run(args.output, args.steps, args.initial, args.reasoning_format, args.diverse)
