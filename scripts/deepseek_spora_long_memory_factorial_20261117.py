"""Isolated reader/output-adapter factorial on matched long memory prompts."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoTokenizer
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_spora_boundary_runtime_pilot import ENROLLMENTS
from scripts.deepseek_spora_memory_answer_distill import _memory_args
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import query
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest

OUT = ROOT / 'runs/deepseek-spora-long-memory-factorial-20261117.json'
ORDINARY = ('只回复两个字：你好。', '小明有三本书，又买了两本，现在有几本？')


def generate(decoder, tokenizer, text, memory_args=None, memory_logits=None):
    inputs = closed_think_inputs(tokenizer, text, 'cuda')
    args = {} if memory_args is None else dict(memory_args)
    args['use_cache'] = True
    with torch.inference_mode():
        output = decoder.generate(**inputs, **args, memory_output_logits=memory_logits,
                                  max_new_tokens=24, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id,
                                  eos_token_id=tokenizer.eos_token_id)
    tail = output[0, inputs['input_ids'].shape[1]:].tolist()
    return {'token_ids': tail, 'text': tokenizer.decode(tail, skip_special_tokens=True).strip(),
            'ended': tokenizer.eos_token_id in tail}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    source_sha = p.sha(p.runtime.CHECKPOINT)
    if p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Base provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    rows, ordinary = [], []
    with tempfile.TemporaryDirectory(prefix='spora-factorial-') as directory:
        p.runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        decoder = chat.decoder.eval()
        original_digest = frozen_digest(chat.organism, chat.state)
        state = fresh_controlled_state(chat.organism)
        for person, old, new in p.FACTS:
            key = ENROLLMENTS[person][0]
            for value in (old, new):
                state, decision = observe_sourced(chat.organism, state, key, value, 'direct', 1.0)
                if not decision['current_write']:
                    raise ValueError('Controlled write rejected')
        state_digest = frozen_digest(chat.organism, state)
        device = next(chat.organism.parameters()).device
        for index, (person, old, new) in enumerate(p.FACTS):
            key, name, _ = ENROLLMENTS[person]
            for mode, target in (('current', new), ('past', old)):
                question = f'{name}现在喜欢哪种饮料？' if mode == 'current' else f'{name}以前喜欢哪种饮料？'
                text = p.FILLER * 4 + question
                args = _memory_args(chat.organism, state, {'key': key, 'intent': mode, 'event_time': index * 2})
                keys = torch.tensor([key], dtype=torch.long, device=device)
                modes = torch.tensor([0 if mode == 'current' else 1], dtype=torch.long, device=device)
                times = torch.tensor([0 if mode == 'current' else index * 2], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = query(chat.organism, state, keys, modes, times)
                if int(logits.argmax(-1)[0]) != target:
                    raise ValueError('Controlled native read disagrees with target')
                variants = {}
                for label, reader, adapter in (('neither', False, False), ('reader', True, False),
                                               ('adapter', False, True), ('both', True, True)):
                    variants[label] = generate(decoder, tokenizer, text, args if reader else None,
                                               logits if adapter else None)
                rows.append({'key': key, 'mode': mode, 'expected': p.WORDS[target], 'variants': variants})
                print(json.dumps({'key': key, 'mode': mode, 'completed': True}), flush=True)
        for text in ORDINARY:
            ordinary.append({'question': text, 'without_memory': generate(decoder, tokenizer, text)})
        if frozen_digest(chat.organism, state) != state_digest or frozen_digest(chat.organism, chat.state) != original_digest:
            raise ValueError('Controlled or source individual changed')
    if p.sha(p.runtime.CHECKPOINT) != source_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected source changed')
    report = {'status': 'isolated_factorial_development_not_acceptance', 'rows': rows,
              'ordinary': ordinary, 'checkpoint_sha256': source_sha, 'base_sha256': p.WEIGHT_SHA,
              'seconds': time.perf_counter()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Four previously inspected synthetic cases with program-supplied address and time.',
                              'Ordinary controls use no forced unrelated memory; not a full interference test.',
                              'First-token output adapter and reader are measured as currently implemented, not independent causal mechanisms.',
                              'No natural routing, durable writes, crash recovery or open-chat evaluation.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'cases': len(rows), 'ordinary': len(ordinary)}), flush=True)


if __name__ == '__main__':
    main()
