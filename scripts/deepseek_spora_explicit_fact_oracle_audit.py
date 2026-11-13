"""Diagnostic: compare original/hybrid generation with explicitly stated facts.

This deliberately places facts in the prompt as a controlled teacher. It is
not persistent memory, does not train a model and must never be deployed as one.
"""
import gc
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

OUTPUT = ROOT / 'runs/deepseek-spora-explicit-fact-oracle-audit-20261113.json'
CASES = (
    ('fruit_a', '小赵', '喜欢的水果', '水果名称', ('山竹', '石榴')),
    ('fruit_b', '小赵', '喜欢的水果', '水果名称', ('杨梅', '龙眼')),
    ('fruit_c', '小赵', '喜欢的水果', '水果名称', ('桑葚', '枇杷')),
    ('drink', '小林', '喜欢的饮料', '饮料名称', ('咖啡', '茶')),
    ('hobby', '小周', '喜欢的活动', '活动名称', ('画画', '摄影')),
)


def prompts(name, relation, answer_type, value):
    question = f'{name}{relation}是什么？请只回答{answer_type}，不要解释。'
    known = f'以下是虚构记录：{name}{relation}是{value}。\n{question}'
    return question, known


def strict_answer(answer, value, name, relation):
    cleaned = answer.strip().rstrip('。.!！').strip()
    return cleaned in (value, f'{name}{relation}是{value}')


def evaluate(model, tokenizer, label):
    rows = []
    with torch.inference_mode():
        for group, name, relation, answer_type, values in CASES:
            question, _ = prompts(name, relation, answer_type, values[0])
            for condition, value in (('no_fact', None), ('fact_a', values[0]), ('fact_b', values[1])):
                prompt = question if value is None else prompts(name, relation, answer_type, value)[1]
                inputs = closed_think_inputs(tokenizer, prompt, 'cuda')
                output = model.generate(**inputs, do_sample=False, use_cache=False, max_new_tokens=24,
                                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                tokens = output[0, inputs['input_ids'].shape[1]:].tolist()
                answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                rows.append({'model': label, 'group': group, 'condition': condition, 'value': value,
                             'prompt': prompt, 'answer': answer, 'token_ids': tokens,
                             'ended_with_eos': tokenizer.eos_token_id in tokens,
                             'strict_correct': None if value is None else strict_answer(answer, value, name, relation)})
            print(json.dumps({'model': label, 'group': group, 'completed_rows': len(rows)}), flush=True)
    return rows


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    source_sha = p.sha(p.runtime.CHECKPOINT)
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Base digest mismatch')
    torch.set_num_threads(2)
    torch.manual_seed(71308)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    started = time.perf_counter()
    base = p.load_full_streamed()
    original = evaluate(base, tokenizer, 'original')
    del base
    gc.collect()
    torch.cuda.empty_cache()
    with tempfile.TemporaryDirectory() as directory:
        p.runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        from spora.structure_transfer import SPORALayerReplacement
        if not all(isinstance(chat.base.model.layers[i], SPORALayerReplacement) for i in (6, 12)):
            raise ValueError('Expected both SPORA replacement layers')
        state_hash = p.frozen_digest(chat.organism, chat.state)
        hybrid = evaluate(chat.base.eval(), tokenizer, 'hybrid_no_memory')
        if p.frozen_digest(chat.organism, chat.state) != state_hash:
            raise ValueError('Source individual state changed')
    if p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(p.runtime.CHECKPOINT) != source_sha:
        raise ValueError('Protected source changed')
    summary = {}
    for name, rows in (('original', original), ('hybrid_no_memory', hybrid)):
        explicit = [r for r in rows if r['value'] is not None]
        pairs = [[r for r in explicit if r['group'] == group] for group, *_ in CASES]
        summary[name] = {'explicit_correct': sum(r['strict_correct'] for r in explicit),
                         'explicit_total': len(explicit),
                         'both_pair_correct': sum(len(pair) == 2 and all(r['strict_correct'] for r in pair)
                                                  for pair in pairs),
                         'pairs': len(pairs), 'completed_eos': sum(r['ended_with_eos'] for r in rows),
                         'total_generations': len(rows)}
    result = {'status': 'explicit_prompt_diagnostic_not_memory', 'summary': summary, 'rows': original + hybrid,
              'base_sha256': base_sha, 'source_checkpoint_sha256': source_sha,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Explicit facts are placed in the text prompt solely for diagnosis.',
                              'The hybrid path uses replaced layers but no SPORA memory reader.',
                              'Small authored examples, one name per question and deterministic decoding.',
                              'Strict short-answer scorer may reject a correct explanatory answer.',
                              'Base pretraining overlap and independence from historical development are unknown.',
                              'No long-term memory, natural writing, identity routing, or training is evaluated.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
