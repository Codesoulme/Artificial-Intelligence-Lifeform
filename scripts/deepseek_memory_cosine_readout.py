"""Compare head-dot and cosine readout from the same continuous matrices."""

import argparse
import json
import time
from pathlib import Path

from deepseek_aligned_memory_calibration import load_adapter
from deepseek_aligned_memory_dialogue import OUT as ADAPTER_RUN, RANK, answer, matrix_bias, score
from deepseek_factored_relation_router import BANK_RUN
from deepseek_memory_dialogue_context import features_for
from deepseek_memory_field_decoding import FIELDS, OUT as FIELD_RUN
from deepseek_memory_field_transfer import OUT as TRANSFER_RUN
from deepseek_memory_turn_priority import OUT as PRIORITY_RUN
from deepseek_output_aligned_memory import (
    GAIN, MODEL, ROOT, SEED, SOURCE, encode, fingerprint, fresh, load_integrated, read, read_bank, sha, torch, write,
)
from spora.memory_output import MemorySequenceLogitsProcessor

OUT = ROOT / 'runs/deepseek-memory-cosine-readout'
GENERATE_OUT = ROOT / 'runs/deepseek-memory-cosine-generate'


@torch.inference_mode()
def projected_codebook(weight, basis):
    projected = torch.cat([chunk.float() @ basis for chunk in weight.split(8192)])
    norms = projected.norm(dim=-1)
    codes = torch.nn.functional.normalize(projected, dim=-1)
    scale = float(norms[norms > 0].median()) * GAIN * (weight.shape[1] / RANK) ** .5
    return codes, scale


@torch.inference_mode()
def cosine_bias(site, bank, address, codes, scale):
    slots = read_bank(site, bank, address).to(codes.device)[:, :RANK]
    if slots[0].norm() < .2:
        return None
    scores = torch.nn.functional.normalize(slots, dim=-1, eps=1e-4) @ codes.T
    scores[slots.norm(dim=-1) < .2] = 0
    return scores * scale


def replay_generate(decoder, tokenizer, prompt, tokens, start, bias):
    prefix = prompt + tokens[:start]
    ids = torch.tensor([prefix], device=bias.device)
    processor = MemorySequenceLogitsProcessor(2 * bias, len(prefix))
    generation = decoder.generate(input_ids=ids, attention_mask=torch.ones_like(ids), logits_processor=[processor],
                                  do_sample=False, max_new_tokens=32 - start, use_cache=True,
                                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    return generation[0, len(prefix):].tolist()


def main(out, verify_generate=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if out.exists():
        raise FileExistsError(out)
    if torch.cuda.mem_get_info()[0] < 3.7 * 1024**3:
        raise RuntimeError('Insufficient free GPU memory')
    started = time.perf_counter()
    torch.set_num_threads(2)
    old_adapter = ADAPTER_RUN / 'candidate.pt'
    bank_paths = {'transfer': TRANSFER_RUN / 'banks.pt', 'prior': BANK_RUN / 'banks.pt'}
    specialist = FIELD_RUN / 'field-adapters.pt'
    candidate_result = read(FIELD_RUN / 'result.json')
    transfer_result = read(TRANSFER_RUN / 'result.json')
    fixtures = {
        'transfer': {r['key']: r for r in read(TRANSFER_RUN / 'plan.json')['rows']},
        'prior': {r['key']: r for r in read(PRIORITY_RUN / 'routes.json')['held']['priority']},
    }
    transfer_routes = {r['key']: r for r in read(TRANSFER_RUN / 'routes.json')}
    expected_digests = {
        old_adapter: read(ADAPTER_RUN / 'result.json')['candidate_sha256'],
        specialist: candidate_result['artifact_sha256'][specialist.name],
        bank_paths['prior']: read(BANK_RUN / 'result.json')['banks_sha256'],
        **{TRANSFER_RUN / name: transfer_result['artifact_sha256'][name] for name in ('banks.pt', 'plan.json', 'routes.json')},
        PRIORITY_RUN / 'routes.json': read(PRIORITY_RUN / 'result.json')['artifact_sha256']['routes.json'],
    }
    protected = (Path(__file__), ROOT / 'spora/memory_output.py', SOURCE, FIELD_RUN / 'result.json',
                 TRANSFER_RUN / 'result.json', *expected_digests,
                 *(MODEL / name for name in ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json')))
    hashes = fingerprint(protected)
    if any(hashes[str(path)] != expected for path, expected in expected_digests.items()):
        raise ValueError('Source artifact changed')
    cohorts = {
        'transfer': [r for r in transfer_result['rows'] if r['trial'] == 'known_field' and r['mode'] == 'field'],
        'prior': [dict(r, entity=22) for r in candidate_result['rows'] if r['mode'] == 'field_style' and r['action'] == 'read'],
    }
    values = {r['expected'] for rows in cohorts.values() for r in rows if r['expected'] is not None}
    out.mkdir()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
              dtype=torch.float16, attn_implementation='sdpa', device_map={'': 'cuda'}, low_cpu_mem_usage=True).eval()
    decoder, parent, state = load_integrated(SOURCE, base, MODEL, individual_device='cpu')
    del parent, state
    decoder.eval().requires_grad_(False)
    head = base.lm_head
    width = head.weight.shape[1]
    _, _, timing, columns = load_adapter(torch.load(old_adapter, weights_only=True), width)
    checkpoint = torch.load(specialist, weights_only=True)
    styles = {}
    for field in FIELDS:
        style = torch.nn.Sequential(torch.nn.Linear(width, 24), torch.nn.Tanh(), torch.nn.Linear(24, len(columns)))
        style.load_state_dict(checkpoint[field]['style'])
        styles[field] = style.eval().requires_grad_(False).to('cuda')
    organism, _ = fresh()
    site = organism.memories[0]
    basis = torch.linalg.qr(torch.randn(width, RANK, generator=torch.Generator().manual_seed(SEED))).Q.to('cuda')
    codes, scale = projected_codebook(head.weight, basis)
    banks = {name: torch.load(path, weights_only=True) for name, path in bank_paths.items()}
    write(out / 'plan.json', dict(protected=hashes, cohorts=cohorts, scale=scale, gain_multiplier=2., rank=RANK,
                                 verify_generate=verify_generate))
    empty = torch.zeros_like(next(iter(banks['transfer'].values())))
    if cosine_bias(site, empty, site.current_basis[26:27], codes, scale) is not None:
        raise ValueError('Empty bank accepted')
    print('Initialized readouts and verified empty-bank rejection', flush=True)
    results, prompts, cache = [], {}, {}
    with torch.inference_mode(), (out / 'progress.jsonl').open('x', encoding='utf8') as stream:
        for suite, cases in cohorts.items():
            for old in cases:
                key, entity, field, version = old['key'], old['entity'], old['read_relation'], old['version']
                prompt_key = suite, key
                if prompt_key not in prompts:
                    inherits = transfer_routes[key]['inherits'] if suite == 'transfer' else not old['current_only']
                    prompts[prompt_key] = features_for(decoder, tokenizer, fixtures[suite][key], inherits)[3]
                prompt = prompts[prompt_key]
                address = site.current_basis[entity:entity + 1]
                bank = banks[suite][field, version]
                bias_key = suite, entity, field, version
                if bias_key not in cache:
                    cache[bias_key] = (matrix_bias(head, site, bank, address, basis), cosine_bias(site, bank, address, codes, scale))
                for mode, bias in zip(('head_dot', 'cosine'), cache[bias_key], strict=True):
                    if bias is None:
                        raise ValueError('Expected occupied bank')
                    response, trace = answer(decoder, tokenizer, prompt, styles[field], columns, timing, 2 * bias)
                    tokens = [step['token'] for step in trace]
                    if mode == 'head_dot' and tokens != old['tokens']:
                        raise ValueError(f'Baseline generation differs: {suite}/{key}/{version}')
                    start = next((i for i, step in enumerate(trace) if step['memory_position'] == 0), None)
                    opening = tokenizer.decode(tokens[:start], skip_special_tokens=True) if start is not None else ''
                    suffix = tokenizer.decode(tokens[start:], skip_special_tokens=True).strip() if start is not None else ''
                    target_ranks = None
                    if old['expected'] is not None:
                        expected = [*encode(tokenizer, old['expected']), tokenizer.eos_token_id]
                        scores = bias[:len(expected)]
                        target_scores = scores.gather(1, torch.tensor(expected, device=scores.device)[:, None])
                        target_ranks = (1 + (scores > target_scores).sum(dim=1)).tolist()
                    row = dict(suite=suite, mode=mode, key=key, entity=entity, version=version, relation=old['relation'],
                               expected=old['expected'], read_relation=field, start=start, opening=opening, fact_suffix=suffix,
                               response=response, tokens=tokens, memory_token_ranks=target_ranks)
                    row['strict'] = score(row, values)
                    row['stored_values_in_response'] = sorted(value for value in values if value in response)
                    if verify_generate and mode == 'cosine' and start is not None:
                        standard = replay_generate(decoder, tokenizer, prompt, tokens, start, bias)
                        row['generate_tokens'] = standard
                        row['generate_matches'] = standard == tokens[start:]
                    results.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
            print('Completed', suite, len(results), flush=True)
    if fingerprint(protected) != hashes:
        raise ValueError('An experiment input changed')
    write(out / 'result.json', dict(rows=results, elapsed_seconds=time.perf_counter() - started,
          peak_allocated_bytes=torch.cuda.max_memory_allocated(), plan_sha256=sha(out / 'plan.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    parser.add_argument('--verify-generate', action='store_true')
    args = parser.parse_args()
    main(args.out or (GENERATE_OUT if args.verify_generate else OUT), args.verify_generate)
