"""Replay fixed memory reads with field-specific answer and timing adapters."""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from deepseek_aligned_memory_calibration import NEW_FACTS, load_adapter
from deepseek_aligned_memory_dialogue import OUT as ADAPTER_RUN, RANK, TRAIN as SINGLE_TRAIN, answer, matrix_bias, score
from deepseek_factored_relation_router import BANK_RUN
from deepseek_focus_sentence_bootstrap import fit_style, gather
from deepseek_focus_timing_gate import train_gate
from deepseek_memory_dialogue_context import features_for
from deepseek_memory_turn_priority import OUT as PRIORITY_RUN
from deepseek_output_aligned_memory import (
    MODEL, ROOT, SEED, SOURCE, encode, fingerprint, fresh, load_integrated, read, sha, torch, write,
)

OUT = ROOT / 'runs/deepseek-memory-field-decoding'
FIELDS = ('drink', 'work', 'place')
MODES = {
    'original': (False, False),
    'field_style': (True, False),
    'field_timing': (False, True),
    'field_both': (True, True),
}


def training_examples(rows):
    result = defaultdict(list)
    for field in FIELDS:
        result[field].extend((dict(question=q), False, opening) for q, opening in SINGLE_TRAIN[field])
        dialogue = [r for r in rows if r['field'] == field]
        for i, row in enumerate(dialogue):
            opening = SINGLE_TRAIN[field][i % len(SINGLE_TRAIN[field])][1]
            result[field].append((row, row['kind'] == 'follow', opening))
    return result


def fit_field(decoder, tokenizer, examples, columns):
    positions = {token: i for i, token in enumerate(columns.tolist())}
    hidden, inside, outside, targets, timing, boundaries = ([] for _ in range(6))
    for row, history, text in examples:
        prompt = features_for(decoder, tokenizer, row, history)[3]
        opening = encode(tokenizer, text)
        vectors, scores, rest = gather(decoder, tokenizer, prompt, opening, {}, columns)
        hidden.append(vectors[:-1])
        inside.append(scores)
        outside.append(rest)
        targets.extend(positions[token] for token in opening)
        timing.append(vectors)
        boundaries.extend([0.] * len(opening) + [1.])
    style, _ = fit_style(torch.cat(hidden), torch.cat(inside), torch.cat(outside), torch.tensor(targets))
    gate = train_gate(torch.cat(timing), torch.tensor(boundaries))
    return style, gate


def summarize(rows):
    summary = {}
    for mode in MODES:
        group = [r for r in rows if r['mode'] == mode]
        supported = [r for r in group if r['expected'] is not None]
        wrong = [r for r in group if r['action'] == 'read' and r['field'] != r['selected']]
        pairs = defaultdict(list)
        for row in supported:
            pairs[row['key']].append(row)
        summary[mode] = dict(
            exact=sum(r['exact_fact'] for r in supported), strict=sum(r['strict'] for r in supported),
            total=len(supported), pairs=sum(all(r['strict'] for r in pair) for pair in pairs.values()),
            wrong_injections=sum(r['start'] is not None for r in wrong),
            fields={field: dict(injected=sum(r['start'] is not None for r in supported if r['field'] == field),
                                exact=sum(r['exact_fact'] for r in supported if r['field'] == field),
                                strict=sum(r['strict'] for r in supported if r['field'] == field),
                                total=sum(r['field'] == field for r in supported)) for field in FIELDS},
        )
    return summary


def main(out):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if out.exists():
        raise FileExistsError(out)
    if torch.cuda.mem_get_info()[0] < 3.7 * 1024**3:
        raise RuntimeError('Insufficient free GPU memory')
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(82438)
    prior_result = read(PRIORITY_RUN / 'result.json')
    adapter_path, banks_path = ADAPTER_RUN / 'candidate.pt', BANK_RUN / 'banks.pt'
    expected_digests = {
        adapter_path: read(ADAPTER_RUN / 'result.json')['candidate_sha256'],
        banks_path: read(BANK_RUN / 'result.json')['banks_sha256'],
        **{PRIORITY_RUN / name: prior_result['artifact_sha256'][name] for name in ('plan.json', 'routes.json')},
    }
    paths = (Path(__file__), SOURCE, PRIORITY_RUN / 'result.json', *expected_digests,
             *(MODEL / name for name in ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json')))
    hashes = fingerprint(paths)
    if any(hashes[str(path)] != digest for path, digest in expected_digests.items()):
        raise ValueError('A source artifact changed')
    prior_plan = read(PRIORITY_RUN / 'plan.json')
    routes = {r['key']: r for r in read(PRIORITY_RUN / 'routes.json')['held']['priority']}
    reference = [r for r in prior_result['rows'] if r['mode'] == 'priority']
    examples = training_examples(prior_plan['train'])
    held_questions = {r['question'] for r in routes.values()}
    all_values = {v for facts in NEW_FACTS.values() for pair in facts.values() for v in pair}
    for group in examples.values():
        for row, _, opening in group:
            if row['question'] in held_questions or any(v in row['question'] + row.get('history', '') + opening
                                                        for v in all_values):
                raise ValueError('Training examples overlap held text or stored facts')
    out.mkdir()
    write(out / 'plan.json', dict(sources=hashes, training=dict(examples), modes=MODES, reference_rows=len(reference)))

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
              dtype=torch.float16, attn_implementation='sdpa', device_map={'': 'cuda'}, low_cpu_mem_usage=True).eval()
    decoder, parent, state = load_integrated(SOURCE, base, MODEL, individual_device='cpu')
    del parent, state
    decoder.eval().requires_grad_(False)
    width = base.lm_head.weight.shape[1]
    _, shared_style, shared_timing, columns = load_adapter(torch.load(adapter_path, weights_only=True), width)
    field_adapters = {field: fit_field(decoder, tokenizer, examples[field], columns) for field in FIELDS}
    checkpoint = out / 'field-adapters.pt'
    torch.save({field: {'style': style.state_dict(), 'timing': timing.state_dict()}
                for field, (style, timing) in field_adapters.items()}, checkpoint)
    saved = torch.load(checkpoint, weights_only=True)
    for field, (style, timing) in field_adapters.items():
        style.load_state_dict(saved[field]['style'])
        timing.load_state_dict(saved[field]['timing'])
        style.to('cuda')
        timing.to('cuda')
    print('Saved and reloaded field adapters', flush=True)

    organism, _ = fresh()
    site = organism.memories[0]
    banks = torch.load(banks_path, weights_only=True, map_location='cpu')
    basis = torch.linalg.qr(torch.randn(width, RANK, generator=torch.Generator().manual_seed(SEED))).Q.to('cuda')
    prompts, biases, cache, rows = {}, {}, {}, []
    with torch.inference_mode(), (out / 'progress.jsonl').open('x', encoding='utf8') as stream:
        for mode, (use_style, use_timing) in MODES.items():
            for original in reference:
                result = dict(original, mode=mode)
                if original['action'] == 'read':
                    key, field, version = original['key'], original['selected'], original['version']
                    if key not in prompts:
                        prompts[key] = features_for(decoder, tokenizer, routes[key], not original['current_only'])[3]
                    bank_key = field, version
                    if bank_key not in biases:
                        bias = matrix_bias(base.lm_head, site, banks[bank_key], site.current_basis[22:23], basis)
                        if bias is None:
                            raise ValueError('Expected populated bank')
                        biases[bank_key] = prior_plan['gain_multiplier'] * bias
                    cache_key = mode, tuple(prompts[key]), bank_key
                    if cache_key not in cache:
                        field_style, field_timing = field_adapters[field]
                        cache[cache_key] = answer(decoder, tokenizer, prompts[key],
                                                 field_style if use_style else shared_style, columns,
                                                 field_timing if use_timing else shared_timing, biases[bank_key])
                    response, trace = cache[cache_key]
                    tokens = [step['token'] for step in trace]
                    start = next((i for i, step in enumerate(trace) if step['memory_position'] == 0), None)
                    opening, suffix = '', ''
                    if start is not None:
                        opening = tokenizer.decode(tokens[:start], skip_special_tokens=True)
                        suffix = tokenizer.decode(tokens[start:], skip_special_tokens=True).strip()
                    result.update(response=response, tokens=tokens, start=start, opening=opening, fact_suffix=suffix)
                    result['strict'] = score(result, all_values)
                if mode == 'original' and any(result[k] != v for k, v in original.items() if k != 'mode'):
                    raise ValueError(f"Original replay differs for {original['key']} version {original['version']}")
                result['exact_fact'] = (result['start'] is not None and result['selected'] == result['field']
                                        and result['fact_suffix'].rstrip('。.!！ ') == result['expected'])
                rows.append(result)
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                stream.flush()
            print('Completed', mode, len(rows), flush=True)
    if fingerprint(paths) != hashes:
        raise ValueError('An experiment input changed')
    summary = summarize(rows)
    write(out / 'result.json', dict(summary=summary, rows=rows, elapsed_seconds=time.perf_counter() - started,
          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
          artifact_sha256={name: sha(out / name) for name in ('plan.json', 'field-adapters.pt')}))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    main(parser.parse_args().out)
