"""Check controlled independent SPORA states in a mixed DeepSeek batch."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_integrated_checkpoint_probe import OUTPUT as CHECKPOINT
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.deepseek_named_readout_alignment import combined_states, snapshot
from scripts.deepseek_semantic_value_pilot import WORDS, word_correct
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.integrated_checkpoint import load_integrated
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-integrated-state-batch-20261110.json'


def end_at_eos(tokens, eos):
    return tokens[:tokens.index(eos) + 1] if eos in tokens else tokens


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = {'weight': MODEL / 'model.safetensors', 'original': INDIVIDUAL,
             'individual': CHECKPOINT / 'individual.pt', 'individual_report': CHECKPOINT / 'result.json'}
    before = {name: sha(path) for name, path in files.items()}
    if before['weight'] != WEIGHT_SHA or before['original'] != INDIVIDUAL_SHA:
        raise ValueError('Protected source mismatch')
    with files['individual_report'].open(encoding='utf-8') as stream:
        if json.load(stream)['checkpoint_sha256'] != before['individual']:
            raise ValueError('Integrated checkpoint provenance mismatch')
    config_sha = feature_files_sha256(MODEL)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    model, organism, original = load_integrated(files['individual'], base, MODEL)
    model.eval()
    state_a = snapshot(organism, KEYS[0], 1, distractor=15)
    state_b = snapshot(organism, KEYS[0], 6, distractor=15)
    states = (original, state_a, state_b)
    digests = [frozen_digest(organism, state) for state in states]
    batch_ab, batch_ba = combined_states([state_a, state_b]), combined_states([state_b, state_a])
    inputs = closed_think_inputs(tokenizer, f'{NAMES[0]}喜欢的饮品是什么？', 'cuda')
    inputs = {key: value.expand(2, -1) for key, value in inputs.items()}
    addresses = [F.normalize(site.current_basis.to('cuda').float(), dim=-1)[KEYS[0]:KEYS[0] + 1]
                 for site in organism.memories]
    modes = {'plain': {}, 'ab': {'memory_state': batch_ab}, 'ba': {'memory_state': batch_ba},
             'mixed': {'memory_state': batch_ab, 'memory_read_mask': torch.tensor([True, False])}}
    logits, generated = {}, {}
    with torch.inference_mode():
        for name, options in modes.items():
            if name != 'plain':
                options = dict(options, memory_addresses=addresses, current_only=True)
            logits[name] = model(**inputs, **options, use_cache=False).logits[:, -1].float().cpu()
            output = model.generate(**inputs, **options, max_new_tokens=8, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
            generated[name] = [end_at_eos(row, tokenizer.eos_token_id)
                               for row in output[:, inputs['input_ids'].shape[1]:].tolist()]
        # Exercise the packed history pages as well as the current-memory path.
        individual_history_logits, individual_history_tokens = [], []
        for state in (state_a, state_b):
            single_inputs = {key: value[:1] for key, value in inputs.items()}
            output = model(**single_inputs, memory_state=state, memory_addresses=addresses,
                           current_only=False, use_cache=False)
            individual_history_logits.append(output.logits[:, -1].float().cpu()[0])
            output = model.generate(**single_inputs, memory_state=state, memory_addresses=addresses,
                                    current_only=False, max_new_tokens=8, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
            individual_history_tokens.append(end_at_eos(
                output[0, single_inputs['input_ids'].shape[1]:].tolist(), tokenizer.eos_token_id))
        history_inputs = {key: value.expand(2, -1) for key, value in inputs.items()}
        history_output = model(**history_inputs, memory_state=batch_ab, memory_addresses=addresses,
                               current_only=False, use_cache=False)
        packed_history_logits = history_output.logits[:, -1].float().cpu()
        history_generation = model.generate(**history_inputs, memory_state=batch_ab, memory_addresses=addresses,
                                            current_only=False, max_new_tokens=8, do_sample=False,
                                            pad_token_id=tokenizer.eos_token_id)
        packed_history_tokens = [end_at_eos(row, tokenizer.eos_token_id)
                                 for row in history_generation[:, history_inputs['input_ids'].shape[1]:].tolist()]
    history_differences = [float((packed_history_logits[i] - individual_history_logits[i]).abs().max())
                           for i in range(2)]
    history_token_checks = [packed_history_tokens[i] == individual_history_tokens[i] for i in range(2)]
    history_argmax_checks = [int(packed_history_logits[i].argmax()) == int(individual_history_logits[i].argmax())
                             for i in range(2)]
    differences = {'a_reordered': float((logits['ab'][0] - logits['ba'][1]).abs().max()),
                   'b_reordered': float((logits['ab'][1] - logits['ba'][0]).abs().max()),
                   'a_mixed': float((logits['ab'][0] - logits['mixed'][0]).abs().max()),
                   'disabled_plain': float((logits['mixed'][1] - logits['plain'][1]).abs().max()),
                   'a_vs_b': float((logits['ab'][0] - logits['ab'][1]).abs().max())}
    decoded = [tokenizer.decode(tokens, skip_special_tokens=True).strip() for tokens in generated['ab']]
    checks = {'a_reordered': generated['ab'][0] == generated['ba'][1],
              'b_reordered': generated['ab'][1] == generated['ba'][0],
              'a_mixed': generated['ab'][0] == generated['mixed'][0],
              'disabled_plain': generated['mixed'][1] == generated['plain'][1],
              'a_expected': word_correct(decoded[0], WORDS[1]), 'b_expected': word_correct(decoded[1], WORDS[6])}
    checks['history_a_matches_individual'] = history_differences[0] <= 0.1 and history_token_checks[0] and history_argmax_checks[0]
    checks['history_b_matches_individual'] = history_differences[1] <= 0.1 and history_token_checks[1] and history_argmax_checks[1]
    print(json.dumps({'differences': differences, 'checks': checks}, ensure_ascii=True), flush=True)
    print(json.dumps({'history_differences': history_differences, 'history_argmax_checks': history_argmax_checks,
                      'history_tokens': history_token_checks}, ensure_ascii=True), flush=True)
    if (not all(torch.isfinite(value).all() for value in logits.values())
            or any(differences[key] > 1e-4 for key in ('a_reordered', 'b_reordered', 'a_mixed', 'disabled_plain'))
            or differences['a_vs_b'] == 0 or not all(checks.values())):
        raise ValueError('State batch isolation comparison failed')
    if (any(sha(path) != before[name] for name, path in files.items()) or feature_files_sha256(MODEL) != config_sha
            or [frozen_digest(organism, state) for state in states] != digests):
        raise ValueError('Protected input or state changed')
    report = {'status': 'two_controlled_states_development_only', 'max_logit_differences': differences,
              'history_max_logit_differences': history_differences,
              'history_argmax_checks': history_argmax_checks,
              'history_generated_tokens': {'packed': packed_history_tokens,
                                           'individual': individual_history_tokens},
              'checks': checks, 'generated_tokens': generated, 'answers': decoded, 'component_sha256': before,
              'feature_config_sha256': config_sha, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Two fresh synthetic states use the same individual model and known address.',
                              'One duplicated prompt, two fixed values, and explicit caller-provided read mask.',
                              'No identity authentication, automatic routing, multi-process isolation, or adversarial test.',
                              'Masked rows are still read computationally; mask is not an access control boundary.',
                              'No general chat, concurrent calls, beam search or arbitrary fact storage evaluated.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'differences': differences, 'checks': checks}), flush=True)


if __name__ == '__main__':
    run()
