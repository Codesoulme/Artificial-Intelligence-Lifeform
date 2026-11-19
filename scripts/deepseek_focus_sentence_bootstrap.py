"""Train a small answer-opening adapter and let the native read supply the fact."""

import argparse
import gc
import json
import time
from pathlib import Path

from deepseek_focus_rollout_gate import HELD_QUESTIONS, screen, summarize
from deepseek_focus_timing_gate import CALIBRATION, TRAIN, complete, encode, train_gate
from deepseek_focus_value_position_20261120 import (
    MODEL, ROOT, SOURCE, SOURCE_RUN, check, fingerprint, focused_memory_args,
    load, load_integrated, prefix, sha, torch, write,
)

OUT = ROOT / 'runs/deepseek-focus-sentence-bootstrap'
FIXED_PROBE = 'previous_word'


@torch.inference_mode()
def gather(decoder, tokenizer, prompt, opening, controls, columns):
    tokens = torch.tensor([prompt + opening], device='cuda')
    output = decoder(input_ids=tokens, use_cache=False, output_hidden_states=True, **controls)
    hidden = output.hidden_states[-1][0, len(prompt) - 1:].float().cpu()
    scores = output.logits[0, len(prompt) - 1:-1].float()
    inside = scores.index_select(1, columns).cpu()
    scores[:, columns] = -torch.inf
    outside = scores.logsumexp(dim=-1).cpu()
    return hidden, inside, outside


def fit_style(hidden, inside, outside, target):
    adapter = torch.nn.Sequential(torch.nn.Linear(hidden.shape[-1], 24), torch.nn.Tanh(),
                                  torch.nn.Linear(24, inside.shape[-1]))
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=.014, weight_decay=.02)
    vectors = torch.nn.functional.normalize(hidden, dim=-1)
    for _ in range(350):
        optimizer.zero_grad(set_to_none=True)
        bias = adapter(vectors)
        logits = torch.cat((inside + bias, outside[:, None]), dim=1)
        loss = torch.nn.functional.cross_entropy(logits, target) + .002 * bias.square().mean()
        loss.backward()
        optimizer.step()
    return adapter.eval().requires_grad_(False), float(loss.detach())


@torch.inference_mode()
def answer(decoder, tokenizer, prompt, controls, columns, style, gate, evidence):
    tokens = torch.tensor([prompt], device='cuda')
    steps = []
    fired = False
    for index in range(40):
        output = decoder(input_ids=tokens, use_cache=False, output_hidden_states=not fired, **controls)
        scores = output.logits[0, -1].float()
        probability = None
        applied = False
        if not fired:
            hidden = torch.nn.functional.normalize(output.hidden_states[-1][0, -1].float(), dim=-1)
            probability = float(gate(hidden).sigmoid())
            if index > 0 and probability >= .8:
                scores = scores + evidence
                applied = fired = True
            else:
                scores[columns] += style(hidden)
        token = int(scores.argmax())
        steps.append({'token': token, 'gate_probability': probability, 'applied': applied})
        if token == tokenizer.eos_token_id:
            break
        tokens = torch.cat((tokens, tokens.new_tensor([[token]])), dim=1)
    text = tokenizer.decode([step['token'] for step in steps], skip_special_tokens=True).strip()
    return text, steps


def main(out):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if out.exists():
        raise FileExistsError(out)
    if torch.cuda.mem_get_info()[0] < 3.7 * 1024**3:
        raise RuntimeError('Insufficient free GPU memory for the frozen model')
    plan = check()
    train_fixtures = [f for f in plan['fixtures'] if f['world'] == 0]
    held_fixtures = [f for f in plan['fixtures'] if f['world'] == 1]
    words = {f['expected_word'] for f in plan['fixtures']}
    torch.set_num_threads(2)
    torch.manual_seed(21418)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    out.mkdir()
    write(out / 'plan.json', {'source_plan_sha256': sha(SOURCE_RUN / 'plan.json'),
          'script_sha256': sha(Path(__file__)), 'training_openings': [(q, s) for q, s, _ in TRAIN],
          'calibration_openings': [(q, s) for q, s, _ in CALIBRATION],
          'held_questions': HELD_QUESTIONS, 'training_fixtures': [f['name'] for f in train_fixtures],
          'held_fixtures': [f['name'] for f in held_fixtures], 'threshold': .8, 'max_tokens': 40})
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, dtype=torch.float16,
        attn_implementation='sdpa', device_map={'': 'cuda'}, low_cpu_mem_usage=True).eval()
    decoder, parent, old = load_integrated(SOURCE, base, MODEL, individual_device='cuda')
    del parent, old
    decoder.eval().requires_grad_(False)
    gc.collect()
    torch.cuda.empty_cache()

    def prompt(question):
        return encode(tokenizer, prefix(tokenizer, question))

    def get_state(fixture):
        organism, state, _ = load(SOURCE_RUN / (fixture['name'] + '.pt'), 'cuda')
        if fingerprint(organism, state) != fixture['state_digest']:
            raise ValueError('Fixture restore mismatch')
        controls, selected = focused_memory_args(organism, state)
        if (selected['key'], selected['event_time']) != (fixture['key'], fixture['event']):
            raise ValueError('Event selection mismatch')
        return organism, state, dict(memory_state=state, **controls)

    opening_tokens = [encode(tokenizer, opening) for _, opening, _ in TRAIN]
    if any(word in opening for _, opening, _ in TRAIN for word in words):
        raise ValueError('A style target contains a memory value')
    column_ids = sorted({token for opening in opening_tokens for token in opening})
    columns = torch.tensor(column_ids, device='cuda')
    positions = {token: index for index, token in enumerate(column_ids)}
    features, style_logits, other_logits, style_targets, gate_vectors, gate_labels = ([] for _ in range(6))
    for index, fixture in enumerate(train_fixtures):
        organism, state, controls = get_state(fixture)
        for shift in (0, 2, 5):
            template = (index + shift) % len(TRAIN)
            question, _, _ = TRAIN[template]
            opening = opening_tokens[template]
            hidden, inside, outside = gather(decoder, tokenizer, prompt(question), opening, controls, columns)
            features.append(hidden[:-1])
            style_logits.append(inside)
            other_logits.append(outside)
            style_targets.extend(positions[token] for token in opening)
            gate_vectors.append(hidden)
            gate_labels.extend([0.] * len(opening) + [1.])
        del organism, state, controls
    style, loss = fit_style(torch.cat(features), torch.cat(style_logits), torch.cat(other_logits),
                            torch.tensor(style_targets))
    gate = train_gate(torch.cat(gate_vectors), torch.tensor(gate_labels))
    torch.save({'gate': gate.state_dict(), 'style': style.state_dict(), 'columns': column_ids}, out / 'adapter.pt')
    saved = torch.load(out / 'adapter.pt', weights_only=True)
    style.load_state_dict(saved['style'])
    gate.load_state_dict(saved['gate'])
    style.to('cuda')
    gate.to('cuda')
    print(json.dumps({'training_openings': len(features), 'opening_tokens': len(style_targets),
                      'final_style_loss': loss, 'style_columns': len(column_ids)}), flush=True)
    probe_question = next(q['text'] for q in plan['questions'] if q['name'] == FIXED_PROBE)
    probe = torch.tensor([prompt(probe_question)], device='cuda')
    rows = []
    with torch.inference_mode(), (out / 'progress.jsonl').open('x', encoding='utf8') as stream:
        without = decoder(input_ids=probe, use_cache=False).logits[0, -1].float()
        for fixture in held_fixtures:
            organism, state, controls = get_state(fixture)
            evidence = decoder(input_ids=probe, use_cache=False, **controls).logits[0, -1].float() - without
            if not torch.isfinite(evidence).all():
                raise ValueError('Non-finite read evidence')
            for question in HELD_QUESTIONS:
                inputs = prompt(question)
                for mode in ('reader', 'prior_gate', 'rollout_gate'):
                    if mode == 'reader':
                        response, steps = complete(decoder, tokenizer, inputs, controls)
                    else:
                        scores = evidence if mode == 'rollout_gate' else torch.zeros_like(evidence)
                        response, steps = answer(decoder, tokenizer, inputs, controls, columns, style, gate, scores)
                    row = dict(group='memory', fixture=fixture['name'], question=question, mode=mode,
                               expected=fixture['expected_word'], response=response, steps=steps,
                               sentence_screen=screen(response, fixture['expected_word'], words))
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            if fingerprint(organism, state) != fixture['state_digest']:
                raise ValueError('Inference changed fixture')
            print('Evaluated', fixture['name'], len(rows), flush=True)
    summary = summarize(rows)
    names = {'memory/reader': 'reader', 'memory/prior_gate': 'style_only',
             'memory/rollout_gate': 'style_and_event_evidence'}
    summaries = {name: summary[key] for key, name in names.items()}
    write(out / 'result.json', {'rows': rows, 'summary': summaries, 'row_modes': names,
          'training_openings': len(features), 'opening_tokens': len(style_targets), 'style_loss': loss,
          'adapter_sha256': sha(out / 'adapter.pt'), 'plan_sha256': sha(out / 'plan.json'),
          'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'seconds': time.perf_counter() - started})
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    main(parser.parse_args().out)
