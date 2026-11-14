"""Counterfactual reader/output ablation with DeepSeek original layers."""
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_addressed_read import closed_think_inputs

OUTPUT = ROOT / 'runs/deepseek-spora-original-reader-output-factorial-20261114.json'
CASES = (
    ('阿宁', 22, '从记录看，阿宁常喝什么饮料？', 2, 5),
    ('阿明晨', 8, '你记得阿明晨喜欢喝哪种饮品吗？', 3, 1),
    ('小林舟', 7, '你记得小林舟喜欢喝哪种饮品吗？', 2, 6),
)
MODES = (('both', True, True), ('reader_only', True, False),
         ('output_only', False, True), ('neither', False, False))


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base_sha, checkpoint_sha = p.sha(p.MODEL / 'model.safetensors'), p.sha(p.runtime.CHECKPOINT)
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Base checksum mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix='spora-original-factorial-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        base = p.load_full_streamed().eval()
        originals = {i: base.model.layers[i] for i in (6, 12)}
        chat = p.runtime.load_chat(base_loader=lambda: base)
        source_digest = p.frozen_digest(chat.organism, chat.state)
        replacements = {i: base.model.layers[i] for i in originals}
        for i in originals:
            base.model.layers[i] = originals[i]
        if set(chat.decoder.layer_indices) & set(originals) or chat.decoder.output_adapter is None:
            raise ValueError('Experiment requires preserved reader wrappers and an output adapter')
        for name, key, question, first, second in CASES:
            for value in (first, second):
                state = p.fresh_controlled_state(chat.organism)
                state, decision = p.observe_sourced(chat.organism, state, key, value, 'direct', 1.0)
                if not decision['current_write']:
                    raise ValueError('Controlled write rejected')
                state_digest = p.frozen_digest(chat.organism, state)
                args, evidence = p.runtime._prepare_memory_read(chat.organism, state, key, 'current', None)
                native = int(evidence.argmax(-1)[0])
                inputs = closed_think_inputs(chat.tokenizer, question, 'cuda')
                row = {'name': name, 'question': question, 'key': key, 'written_value_id': value,
                       'expected': p.WORDS[value], 'native_value_id': native, 'modes': {}}
                for label, reader_on, output_on in MODES:
                    reader_method = chat.decoder.reader.forward
                    intervention = {'wraps': reader_method} if reader_on else {
                        'side_effect': lambda hidden, *args, **kwargs: hidden}
                    with patch.object(chat.decoder.reader, 'forward', **intervention) as reader:
                        with torch.inference_mode():
                            ids = chat.decoder.generate(**inputs, **args,
                                memory_output_logits=evidence if output_on else None,
                                max_new_tokens=24, do_sample=False,
                                pad_token_id=chat.tokenizer.pad_token_id,
                                eos_token_id=chat.tokenizer.eos_token_id)
                    sites = sorted({call.args[2] for call in reader.call_args_list})
                    if sites != [0, 1]:
                        raise ValueError('Both reader sites must be exercised')
                    tail = ids[0, inputs['input_ids'].shape[1]:].tolist()
                    answer = chat.tokenizer.decode(tail, skip_special_tokens=True).strip()
                    row['modes'][label] = {'answer': answer, 'exact': answer.rstrip('。.!！ ') == p.WORDS[value],
                                           'reader_calls': reader.call_count, 'sites': sites,
                                           'ended': chat.tokenizer.eos_token_id in tail}
                rows.append(row)
                print(json.dumps(row, ensure_ascii=True), flush=True)
                if p.frozen_digest(chat.organism, state) != state_digest:
                    raise ValueError('Temporary state changed')
        for i in originals:
            base.model.layers[i] = replacements[i]
        if p.frozen_digest(chat.organism, chat.state) != source_digest:
            raise ValueError('Original individual changed')
    if p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(p.runtime.CHECKPOINT) != checkpoint_sha:
        raise ValueError('Protected file changed')
    summary = {'conditions': len(rows), 'native_correct': sum(r['written_value_id'] == r['native_value_id'] for r in rows)}
    for label, _, _ in MODES:
        summary[label] = {'exact': sum(r['modes'][label]['exact'] for r in rows),
                          'both_values_correct': sum(all(r['modes'][label]['exact'] for r in rows if r['name'] == name)
                                                     for name, _, _, _, _ in CASES),
                          'changed_answers': sum(len({r['modes'][label]['answer'] for r in rows if r['name'] == name}) == 2
                                                 for name, _, _, _, _ in CASES)}
    result = {'summary': summary, 'rows': rows, 'base_sha256': base_sha, 'checkpoint_sha256': checkpoint_sha,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Three previously inspected development prompts, two controlled values each.',
                              'Addresses supplied directly; no natural language routing or identity resolution.',
                              'Output adapter injects a learned vocabulary bias, not general textual memory.',
                              'Reader sites intervened on together; only exact single-word scores.',
                              'Values manually written; no multi-turn, natural learning, history or real user.',
                              'Separate fresh states model counterfactuals, not sequential fact correction.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
