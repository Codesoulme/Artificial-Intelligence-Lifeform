"""Compare bounded Chinese generation at three graft strengths; memory disabled."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoTokenizer
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_blend_language_probe import QUESTIONS
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_layer18_general_long_audit import PAIR, repetition
from scripts.deepseek_spora_memory_answer_distill import _load_candidate
from spora.structure_transfer import SPORALayerBlend

OUTPUT = ROOT / 'runs/deepseek-spora-blend-generation-probe-20261113.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Base digest mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed().eval()
    originals = {i: base.model.layers[i] for i in (6, 12)}
    candidates = {i: _load_candidate(PAIR / f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    questions = QUESTIONS[:3]
    encoded = [closed_think_inputs(tokenizer, q, 'cuda') for q in questions]
    rows = []
    for strength in (0.0, 0.25, 1.0):
        for i in (6, 12):
            base.model.layers[i] = originals[i] if strength == 0 else SPORALayerBlend(originals[i], candidates[i], strength).eval()
        for question, prompt in zip(questions, encoded):
            tick = time.perf_counter()
            with torch.inference_mode():
                ids = base.generate(**prompt, max_new_tokens=32, do_sample=False, use_cache=False,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            tail = ids[0, prompt['input_ids'].shape[1]:].tolist()
            rows.append({'strength': strength, 'question': question,
                         'answer': tokenizer.decode(tail, skip_special_tokens=True).strip(),
                         'token_ids': tail, 'ended': bool(tail and tail[-1] == tokenizer.eos_token_id),
                         'repeat_3gram_fraction': repetition(tail, 3), 'seconds': time.perf_counter() - tick})
        print(json.dumps({'strength': strength, 'cases': len(questions)}), flush=True)
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Base changed')
    report = {'status': 'bounded_blend_generation_development_probe', 'rows': rows,
              'base_sha256': WEIGHT_SHA, 'candidate_sha256': {str(i): sha(PAIR / f'layer-{i}.pt') for i in (6, 12)},
              'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Three authored prompts, maximum 32 tokens, greedy decoding with thinking closed.',
                              'No memory read, no human blind grading, and no assurance prompts are independent.',
                              'All configurations disable KV cache for comparability; measured time is not production throughput.']}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
