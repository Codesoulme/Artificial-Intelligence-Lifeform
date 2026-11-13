"""Inspect oracle-position token competition under correct and swapped evidence."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.deepseek_spora_online_lora_memory_pilot import OnlineDelta
from scripts.deepseek_spora_sentence_gate_pilot import SENTENCE, ADAPTER, checked, gold_inputs
from scripts.deepseek_spora_sentence_lora_pilot import HELD_TEMPLATE, make_state
from spora.memory_output import MemoryOutputAdapter

OUTPUT = ROOT / 'runs/deepseek-spora-sentence-token-competition-20261114.json'
CASES = [(i, v) for i in (6, 7) for v in (1, 3, 4, 7)]


def summarize(scores, bias, target_id, tokenizer, gain):
    adjusted = scores + gain * bias
    best = int(adjusted.argmax())
    rank = 1 + int((adjusted > adjusted[target_id]).sum())
    return {'target_score': float(adjusted[target_id]), 'target_rank': rank,
            'winner_id': best, 'winner': tokenizer.decode([best]),
            'winner_score': float(adjusted[best]),
            'target_minus_winner': float(adjusted[target_id] - adjusted[best])}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    sentence, sentence_path, sentence_sha = checked(SENTENCE, 'adapter.pt', 'adapter_sha256')
    output, adapter_path, adapter_sha = checked(ADAPTER, 'adapter.pt', 'adapter_sha256')
    if (base_sha != p.WEIGHT_SHA or sentence['base_sha256'] != base_sha
            or sentence['checkpoint_sha256'] != checkpoint_sha or output['base_sha256'] != base_sha):
        raise ValueError('Candidate provenance mismatch')
    for layer in (6, 12):
        path = ROOT / 'runs/deepseek-spora-diverse-memory-replay-20261111' / f'layer-{layer}.pt'
        if output['candidate_sha256'][str(layer)] != p.sha(path):
            raise ValueError('Output adapter layer mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix='spora-token-competition-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        original_digest = p.frozen_digest(chat.organism, chat.state)
        for layer, weights in zip(sentence['layers'], sentence['adapters']):
            mlp = chat.base.model.layers[layer].mlp
            module = OnlineDelta(mlp.down_proj, sentence['rank'])
            result = module.load_state_dict(weights, strict=False)
            missing = {'base.weight'} | ({'base.bias'} if module.base.bias is not None else set())
            if set(result.missing_keys) != missing or result.unexpected_keys:
                raise ValueError('Sentence candidate mismatch')
            mlp.down_proj = module.eval()
        adapter = MemoryOutputAdapter(**output['config']).to('cuda').eval()
        adapter.load_state_dict(output['state'], strict=True)
        for index, value in CASES:
            alternate = 7 if value != 7 else 4
            state = make_state(chat, KEYS[index], value)
            other_state = make_state(chat, KEYS[index], alternate)
            state_digest = p.frozen_digest(chat.organism, state)
            other_digest = p.frozen_digest(chat.organism, other_state)
            question = HELD_TEMPLATE.format(name=NAMES[index])
            prefix, target, position = gold_inputs(chat.tokenizer, NAMES[index], value, question)
            target_id = int(target[0, position])
            teacher_prefix = torch.cat((prefix['input_ids'], target[:, :position]), dim=1)
            args, evidence = p.runtime._prepare_memory_read(chat.organism, state, KEYS[index], 'current', None)
            _, swapped = p.runtime._prepare_memory_read(chat.organism, other_state, KEYS[index], 'current', None)
            with torch.inference_mode():
                memory_scores = chat.decoder(input_ids=teacher_prefix, attention_mask=torch.ones_like(teacher_prefix),
                                             return_dict=True, **args).logits[0, -1].float()
                bare_scores = chat.decoder(input_ids=teacher_prefix, attention_mask=torch.ones_like(teacher_prefix),
                                           use_cache=False, return_dict=True).logits[0, -1].float()
                bias = adapter(evidence).float()[0]
                swapped_bias = adapter(swapped).float()[0]
            results = {'no_reader': summarize(bare_scores, torch.zeros_like(bias), target_id, chat.tokenizer, 0),
                       'reader': summarize(memory_scores, bias, target_id, chat.tokenizer, 0),
                       'matched_gain_1': summarize(memory_scores, bias, target_id, chat.tokenizer, 1),
                       'matched_gain_2': summarize(memory_scores, bias, target_id, chat.tokenizer, 2),
                       'swapped_gain_1': summarize(memory_scores, swapped_bias, target_id, chat.tokenizer, 1)}
            row = {'name': NAMES[index], 'question': question, 'target': p.WORDS[value],
                   'target_token_id': target_id, 'target_token': chat.tokenizer.decode([target_id]),
                   'target_token_count': len(chat.tokenizer.encode(p.WORDS[value], add_special_tokens=False)),
                   'prefix_answer': chat.tokenizer.decode(target[0, :position], skip_special_tokens=True),
                   'native_id': int(evidence.argmax(-1)[0]), 'swapped_id': int(swapped.argmax(-1)[0]),
                   'matched_target_bias': float(bias[target_id]), 'swapped_target_bias': float(swapped_bias[target_id]),
                   'results': results}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
            if p.frozen_digest(chat.organism, state) != state_digest or p.frozen_digest(chat.organism, other_state) != other_digest:
                raise ValueError('Controlled state changed')
        if p.frozen_digest(chat.organism, chat.state) != original_digest:
            raise ValueError('Original state changed')
    for path, digest in ((p.MODEL / 'model.safetensors', base_sha), (p.runtime.CHECKPOINT, checkpoint_sha),
                         (sentence_path, sentence_sha), (adapter_path, adapter_sha)):
        if p.sha(path) != digest:
            raise ValueError('Protected file changed')
    summary = {label: {'top_target': sum(r['results'][label]['target_rank'] == 1 for r in rows)}
               for label in ('no_reader', 'reader', 'matched_gain_1', 'matched_gain_2', 'swapped_gain_1')}
    result = {'rows': rows, 'summary': summary, 'base_sha256': base_sha, 'checkpoint_sha256': checkpoint_sha,
              'sentence_sha256': sentence_sha, 'adapter_sha256': adapter_sha,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Gold answer prefix and numeric address supplied by experiment; not autonomous generation.',
                              'Only first target token is scored, not complete or faithful answers.',
                              'Deliberately swapped evidence is a diagnostic, not actual routed behavior.',
                              'Small development set, values and people were used elsewhere.',
                              'Gain 2 is an oracle-position sensitivity probe, not a calibrated policy.']}
    with OUTPUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
