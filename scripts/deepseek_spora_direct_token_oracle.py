"""Upper-bound diagnosis for the memory-to-language output interface.

The oracle processor maps the controlled memory value directly to the target
answer token sequence. It is deliberately not a model component: it measures
whether generation can express a value once routing and memory retrieval are
held correct.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import LogitsProcessor
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_boundary_runtime_pilot import ENROLLMENTS
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_memory_answer_distill import _memory_args
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import query
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest


OUTPUT = ROOT / 'runs/deepseek-spora-direct-token-oracle-v2-20261114.json'
MAX_NEW = 24
FILLER = ('窗边有一本书，桌旁有一盏灯。清晨路边的树叶轻轻摇动。'
          '图书馆有人整理书架，公园里有人缓慢散步。傍晚路灯照亮安静的街道。')
PERSONS = (0, 1)
VALUES = (0, 1, 3, 6, 7)
QUESTIONS = (
    ('current', '{name}现在喜欢哪种饮料？'),
    ('past', '{name}以前喜欢哪种饮料？'),
)


class DirectTokenOracle(LogitsProcessor):
    """Force only the known target sequence during the answer span."""

    def __init__(self, target_ids, prompt_length, bias=20.0):
        if not target_ids or type(prompt_length) is not int or prompt_length < 1:
            raise ValueError('Target ids and prompt length are required')
        self.target_ids = tuple(int(x) for x in target_ids)
        self.prompt_length = prompt_length
        self.bias = float(bias)

    def __call__(self, input_ids, scores):
        step = input_ids.shape[1] - self.prompt_length
        if 0 <= step < len(self.target_ids):
            scores = scores.clone()
            scores[:, self.target_ids[step]] += self.bias
        return scores


def clean(answer):
    return answer.strip().rstrip('。.!！?？,，;；:： ').strip()


def exact(answer, expected):
    return clean(answer) == expected


def generate(decoder, tokenizer, inputs, spec, *, evidence=None, processor=None):
    kwargs = {**inputs, **spec, 'use_cache': True, 'max_new_tokens': MAX_NEW,
              'do_sample': False, 'pad_token_id': tokenizer.pad_token_id,
              'eos_token_id': tokenizer.eos_token_id}
    if evidence is not None:
        kwargs['memory_output_logits'] = evidence
    if processor is not None:
        kwargs['logits_processor'] = [processor]
    with torch.inference_mode():
        output = decoder.generate(**kwargs)
    tail = output[0, inputs['input_ids'].shape[1]:].tolist()
    return {'answer': tokenizer.decode(tail, skip_special_tokens=True).strip(),
            'token_ids': tail}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checkpoint_sha = sha(runtime.CHECKPOINT)
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Base digest mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = []
    tokenizer = runtime.AutoTokenizer.from_pretrained(MODEL, local_files_only=True,
                                                       trust_remote_code=False)
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        decoder = chat.decoder.eval()
        original_digest = frozen_digest(chat.organism, chat.state)
        device = next(chat.organism.parameters()).device
        state = fresh_controlled_state(chat.organism)
        for person in PERSONS:
            key = ENROLLMENTS[person][0]
            for value in (1, VALUES[-1]):
                state, decision = observe_sourced(chat.organism, state, key, value, 'direct', 1.0)
                if not decision['current_write']:
                    raise ValueError('Controlled memory write rejected')
        state_digest = frozen_digest(chat.organism, state)
        for person in PERSONS:
            key, name, _ = ENROLLMENTS[person]
            for mode, template in QUESTIONS:
                target = VALUES[-1] if mode == 'current' else 1
                question = template.format(name=name)
                inputs = closed_think_inputs(tokenizer, FILLER * 3 + question, 'cuda')
                prompt_length = inputs['input_ids'].shape[1]
                spec = _memory_args(chat.organism, state,
                                    {'key': key, 'intent': mode,
                                     'event_time': person * 2})
                keys = torch.tensor([key], dtype=torch.long, device=device)
                modes = torch.zeros_like(keys) if mode == 'current' else torch.ones_like(keys)
                times = torch.full_like(keys, person * 2)
                with torch.no_grad():
                    evidence = query(chat.organism, state, keys, modes, times)
                native_value = int(evidence.argmax(-1)[0])
                target_ids = tokenizer.encode(WORDS[target], add_special_tokens=False)
                outputs = {
                    'no_memory': generate(decoder, tokenizer, inputs, {}),
                    'memory_read_only': generate(decoder, tokenizer, inputs, spec),
                    'learned_output': generate(decoder, tokenizer, inputs, spec,
                                               evidence=evidence),
                    'direct_token_oracle': generate(
                        decoder, tokenizer, inputs, spec,
                        processor=DirectTokenOracle(target_ids, prompt_length)),
                }
                row = {
                    'person': name, 'key': key, 'mode': mode,
                    'expected_value': WORDS[target],
                    'native_memory_value': (WORDS[native_value] if 0 <= native_value < len(WORDS)
                                           else f'<unmapped-index:{native_value}>'),
                    'native_memory_index': native_value,
                    'native_memory_correct': native_value == target,
                    'target_token_ids': target_ids,
                    'question': question,
                    'answers': outputs,
                    'exact': {label: exact(value['answer'], WORDS[target])
                              for label, value in outputs.items()},
                }
                rows.append(row)
                print(json.dumps({
                    'completed': len(rows), 'person': name, 'mode': mode,
                    'memory_correct': row['native_memory_correct'],
                    'learned_correct': row['exact']['learned_output'],
                    'oracle_correct': row['exact']['direct_token_oracle'],
                }, ensure_ascii=False), flush=True)
        if frozen_digest(chat.organism, state) != state_digest:
            raise ValueError('Temporary memory changed during generation')
        if frozen_digest(chat.organism, chat.state) != original_digest:
            raise ValueError('Original individual changed during generation')
    if sha(runtime.CHECKPOINT) != checkpoint_sha or sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected source changed')
    summary = {'cases': len(rows), 'memory_correct': sum(r['native_memory_correct'] for r in rows)}
    for label in ('no_memory', 'memory_read_only', 'learned_output', 'direct_token_oracle'):
        summary[label] = sum(r['exact'][label] for r in rows)
    report = {
        'status': 'output_interface_upper_bound_diagnostic',
        'summary': summary,
        'rows': rows,
        'source_checkpoint_sha256': checkpoint_sha,
        'base_sha256': WEIGHT_SHA,
        'seconds': time.perf_counter() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'limitations': [
            'DirectTokenOracle receives the target answer token sequence from the test and is not deployable.',
            'Numerical addresses, writes, question mode and target vocabulary are supplied by the test.',
            'The finite beverage cases do not measure open-domain conversation or natural-language memory writing.',
            'Exact short-answer scoring can reject valid longer answers; raw generations are retained.',
            'No default checkpoint, base weight or real individual was modified.',
        ],
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
