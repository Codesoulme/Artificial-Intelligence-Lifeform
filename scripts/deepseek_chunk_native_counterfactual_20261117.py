"""Causal first-token and greedy probe of stored chunk vectors in native reader.

The same generic prompt and address are used in each condition. Swapping only
the addressed memory vector tests sensitivity; targets are used for scoring,
never appended to the prompt or passed through an output-bias adapter.
"""
import hashlib
import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch.nn import functional as F

from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_chunk_history_ridge_holdout_20261117 import fixed_inputs
from scripts.deepseek_chunk_variable_crash_pilot_20261117 import INPUT, NAMES
from scripts.deepseek_chunk_history_durable_pilot_20261117 import codec_from_embedding
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.language_training import frozen_digest
from spora.named_address import address_basis_sha256
from spora.research_chunk_history_store import ResearchChunkHistoryStore

DB = ROOT / 'runs/deepseek-spora-chunk-history-paging-20261117/research.sqlite'
REPORT = ROOT / 'runs/deepseek-spora-chunk-native-counterfactual-20261117.json'
PROMPT = 'Repeat the exact four-token fragment stored in memory. Output only the fragment.'
INPUT_INDICES = (10, 6, 7, 8, 9)
CHUNK_INDEX = 1


def replace_query_value(matrix, address, target):
    """Return a copy whose one normalized-address read is target."""
    if (not isinstance(matrix, torch.Tensor) or matrix.ndim != 3 or matrix.shape[0] != 1
            or matrix.shape[1] != matrix.shape[2] or not isinstance(address, torch.Tensor)
            or not isinstance(target, torch.Tensor) or address.shape != (1, matrix.shape[1])
            or target.shape != address.shape or not bool(torch.isfinite(matrix).all())
            or not bool(torch.isfinite(address).all()) or not bool(torch.isfinite(target).all())
            or float(address.norm()) < 1e-8):
        raise ValueError('Invalid matrix, address or target')
    a = F.normalize(address.to(matrix).float(), dim=-1)
    m = matrix.float()
    old = torch.bmm(m, a.unsqueeze(-1)).squeeze(-1)
    return m + (target.to(m).float() - old).unsqueeze(-1) * a.unsqueeze(1)


def main():
    if REPORT.exists():
        raise FileExistsError(REPORT)
    paths = {'base': runtime.MODEL / 'model.safetensors', 'checkpoint': runtime.CHECKPOINT,
             'input': INPUT, 'database': DB}
    hashes = {name: runtime.sha(path) for name, path in paths.items()}
    if hashes['base'] != runtime.WEIGHT_SHA:
        raise ValueError('Base provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix='spora-native-chunk-causal-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        original_digest = frozen_digest(chat.organism, chat.state)
        device = next(chat.organism.parameters()).device
        codec = codec_from_embedding(chat.base.model.embed_tokens.weight.detach().cpu(), chat.tokenizer).to(device)
        store = ResearchChunkHistoryStore.open(DB, codec, device='cuda')
        try:
            state_digest = frozen_digest(store.model, store.state)
            if address_basis_sha256(store.model) != address_basis_sha256(chat.organism):
                raise ValueError('Memory basis does not match decoder individual')
            if store.state.cursor != 65 or len(store.state.pages) != 2:
                raise ValueError('Unexpected stored memory state')
            inputs = fixed_inputs(INPUT, chat.tokenizer)
            targets, addresses = [], []
            for person, input_index in enumerate(INPUT_INDICES):
                length = 12 if person == 0 else 24
                expected = inputs[input_index]['tokens'][:length]
                identity = chat.tokenizer.encode(NAMES[person], add_special_tokens=False)
                if store.read(person) != {'identity_ids': identity, 'fact_ids': expected}:
                    raise ValueError('Stored record does not reproduce test target')
                target = expected[CHUNK_INDEX*4:CHUNK_INDEX*4+4]
                if len(target) != 4:
                    raise ValueError('Target segment too short')
                targets.append(target)
                key = person*6 + CHUNK_INDEX
                addresses.append([F.normalize(site.current_basis[key:key+1].detach().float(), dim=-1)
                                  for site in store.model.memories])
            if len({target[0] for target in targets}) != len(targets):
                raise ValueError('First target tokens must differ')
            prompt = closed_think_inputs(chat.tokenizer, PROMPT, 'cuda')
            prompt_hash = hashlib.sha256(prompt['input_ids'].detach().cpu().numpy().tobytes()).hexdigest()
            decoder = chat.decoder.eval()

            def state_with(person, source):
                matrices = []
                for i, matrix in enumerate(store.state.current):
                    a, b = addresses[person][i], addresses[source][i]
                    target_value = torch.bmm(matrix.float(), b.unsqueeze(-1)).squeeze(-1)
                    replacement = replace_query_value(matrix, a, target_value)
                    matrices.append(replacement)
                return replace(store.state, current=matrices)

            def memory_args(state, person):
                return {'memory_state': state, 'memory_addresses': addresses[person],
                        'memory_slots': torch.tensor([0], dtype=torch.long, device=device),
                        'current_only': True}

            def next_logits(state, person):
                with torch.inference_mode():
                    result = decoder(**prompt, use_cache=False,
                                     **({} if state is None else memory_args(state, person)))
                    return result.logits[0, -1].float()

            def greedy(state, person):
                with torch.inference_mode():
                    output = decoder.generate(**prompt, **memory_args(state, person), max_new_tokens=4,
                                              do_sample=False, use_cache=True,
                                              pad_token_id=chat.tokenizer.pad_token_id,
                                              eos_token_id=chat.tokenizer.eos_token_id)
                return output[0, prompt['input_ids'].shape[1]:].detach().cpu().tolist()

            none = next_logits(None, 0)
            for person in range(len(targets)):
                other = (person + 1) % len(targets)
                normal, swapped = store.state, state_with(person, other)
                a, b = targets[person][0], targets[other][0]
                logits_normal = next_logits(normal, person)
                logits_swap = next_logits(swapped, person)
                normal_tail, swap_tail = greedy(normal, person), greedy(swapped, person)
                logp_normal, logp_swap, logp_none = [F.log_softmax(x, dim=-1) for x in (logits_normal, logits_swap, none)]
                row = {'person': person, 'swapped_from': other,
                       'normal_a_vs_b_logit_margin': float(logits_normal[a]-logits_normal[b]),
                       'swap_a_vs_b_logit_margin': float(logits_swap[a]-logits_swap[b]),
                       'no_memory_a_vs_b_logit_margin': float(none[a]-none[b]),
                       'normal_a_log_probability': float(logp_normal[a]),
                       'swap_b_log_probability': float(logp_swap[b]),
                       'no_memory_a_log_probability': float(logp_none[a]),
                       'first_token_pair_correct': bool(logits_normal[a] > logits_normal[b] and logits_swap[b] > logits_swap[a]),
                       'normal_greedy_exact': normal_tail == targets[person],
                       'swapped_greedy_exact': swap_tail == targets[other],
                       'paired_greedy_exact': normal_tail == targets[person] and swap_tail == targets[other],
                       'greedy_changed': normal_tail != swap_tail,
                       'normal_generated_ids': normal_tail, 'swapped_generated_ids': swap_tail,
                       'max_logit_change': float((logits_normal-logits_swap).abs().max())}
                rows.append(row)
                print(json.dumps({'person': person, 'first_pair': row['first_token_pair_correct'],
                                  'normal_exact': row['normal_greedy_exact'],
                                  'swap_exact': row['swapped_greedy_exact'],
                                  'greedy_changed': row['greedy_changed']}), flush=True)
            if (frozen_digest(store.model, store.state) != state_digest or
                    frozen_digest(chat.organism, chat.state) != original_digest):
                raise ValueError('Protected individual changed')
        finally:
            store.close()
    if any(runtime.sha(path) != hashes[name] for name, path in paths.items()):
        raise ValueError('Protected input changed')
    summary = {'pairs': len(rows), 'first_token_pair_correct': sum(r['first_token_pair_correct'] for r in rows),
               'normal_greedy_exact': sum(r['normal_greedy_exact'] for r in rows),
               'swapped_greedy_exact': sum(r['swapped_greedy_exact'] for r in rows),
               'paired_greedy_exact': sum(r['paired_greedy_exact'] for r in rows),
               'greedy_changed': sum(r['greedy_changed'] for r in rows)}
    report = {'status': 'native_chunk_counterfactual_development_probe', 'prompt': PROMPT,
              'prompt_token_sha256': prompt_hash, 'summary': summary, 'rows': rows,
              'source_sha256': hashes, 'seconds': time.perf_counter()-started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Five previously used artificial cases; English fragments, not natural Chinese QA.',
                              'Identity, chunk position and numeric address are manually specified.',
                              'The fragment-copy prompt is unusual and not trained for this reader.',
                              'Counterfactual matrix replacement is a research intervention, not a verified write.',
                              'Native reader only; no fixed-class memory-output adapter or external answer text in prompt.',
                              'First-token scores and greedy four-token exactness; no general chat or provenance claims.']}
    with REPORT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
