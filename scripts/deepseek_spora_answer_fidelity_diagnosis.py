"""Separate native value retrieval from language generation on controlled facts."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_named_binding_probe import KEYS
from scripts.deepseek_spora_presence_routing_joint_audit import NAMES, VALUES

OUTPUT = ROOT / 'runs/deepseek-spora-answer-fidelity-diagnosis-20261114.json'
QUESTIONS = (
    (0, '请告诉我小林目前想喝哪种饮料？'),
    (3, '对于阿哲，记录里现在的饮料是什么？'),
    (2, '查一下小雨保存的饮料喜好。'),
)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Base digest mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix='spora-fidelity-') as directory:
        p.runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        original_digest = p.frozen_digest(chat.organism, chat.state)
        state = p.fresh_controlled_state(chat.organism)
        for index, _ in enumerate(NAMES):
            state, decision = p.observe_sourced(chat.organism, state, KEYS[index], VALUES[index], 'direct', 1.0)
            if not decision['current_write']:
                raise ValueError('Controlled write rejected')
        state_digest = p.frozen_digest(chat.organism, state)
        for index, question in QUESTIONS:
            key = KEYS[index]
            alternate = (VALUES[index] + 1) % len(p.WORDS)
            changed_state, decision = p.observe_sourced(chat.organism, state, key, alternate, 'direct', 1.0)
            if not decision['current_write']:
                raise ValueError('Counterfactual write rejected')
            for condition, selected_state, target in (('original', state, VALUES[index]),
                                                       ('changed', changed_state, alternate)):
                args, evidence = p.runtime._prepare_memory_read(chat.organism, selected_state, key, 'current', None)
                scores = evidence.detach().float().cpu()[0]
                values, indices = torch.topk(scores, min(4, scores.numel()))
                prompt = closed_think_inputs(chat.tokenizer, question, 'cuda')
                answers = {}
                for label, output_evidence in (('full', evidence), ('no_output_adapter', None)):
                    with torch.inference_mode():
                        ids = chat.decoder.generate(**prompt, **args, memory_output_logits=output_evidence,
                                                    max_new_tokens=32, do_sample=False,
                                                    pad_token_id=chat.tokenizer.pad_token_id,
                                                    eos_token_id=chat.tokenizer.eos_token_id)
                    answers[label] = chat.tokenizer.decode(ids[0, prompt['input_ids'].shape[1]:],
                                                           skip_special_tokens=True).strip()
                row = {'question': question, 'condition': condition, 'key': key, 'written_value_id': target,
                       'written_value': p.WORDS[target], 'retrieved_value_id': int(indices[0]),
                       'top_ids': indices.tolist(), 'top_scores': values.tolist(), 'answers': answers}
                rows.append(row)
                print(json.dumps(row, ensure_ascii=True), flush=True)
        if p.frozen_digest(chat.organism, state) != state_digest or p.frozen_digest(chat.organism, chat.state) != original_digest:
            raise ValueError('Protected state changed')
    if p.sha(p.runtime.CHECKPOINT) != checkpoint_sha or p.sha(p.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Protected file changed')
    result = {'rows': rows, 'checkpoint_sha256': checkpoint_sha, 'base_sha256': base_sha,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Direct numeric address bypasses natural-language routing.',
                              'Three authored questions and controlled numeric writes are development examples.',
                              'Disabling the output adapter retains the neural reader, not a no-memory baseline.',
                              'Top memory score is not proof of calibrated confidence or factual correctness.']}
    with OUTPUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
