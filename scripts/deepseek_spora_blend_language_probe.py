"""Controlled teacher-forced comparison of progressive layer graft strengths."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_layer18_general_long_audit import PAIR
from scripts.deepseek_spora_memory_answer_distill import _load_candidate
from spora.structure_transfer import SPORALayerBlend

OUTPUT = ROOT / 'runs/deepseek-spora-blend-language-probe-20261113.json'
# Authored diagnostic prompts, not a verified independent or representative set.
QUESTIONS = ('请用一句话解释什么是耐心。', '一个人迟到了，怎样礼貌地道歉？',
             '如果杯子倒了，桌面会怎样？', '帮我把明天的任务排个顺序。',
             '两支笔加三支笔一共几支？', '朋友说今天很累，你会怎么安慰？')
STRENGTHS = (0.0, 0.25, 0.5, 1.0)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Base weight digest mismatch')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed().eval()
    originals = {i: base.model.layers[i] for i in (6, 12)}
    candidates = {i: _load_candidate(PAIR / f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    inputs = [closed_think_inputs(tokenizer, q, 'cuda') for q in QUESTIONS]
    teacher = []
    rows = []
    with torch.inference_mode():
        for item in inputs:
            teacher.append(base(**item, use_cache=False).logits[0, -1].float().cpu())
        for strength in STRENGTHS:
            for i in (6, 12):
                base.model.layers[i] = (originals[i] if strength == 0 else
                    SPORALayerBlend(originals[i], candidates[i], strength).eval())
            for q, item, ref in zip(QUESTIONS, inputs, teacher):
                logits = base(**item, use_cache=False).logits[0, -1].float().cpu()
                log_ref = F.log_softmax(ref, dim=-1)
                log_actual = F.log_softmax(logits, dim=-1)
                teacher_top = int(ref.argmax())
                rows.append({'strength': strength, 'question': q,
                             'teacher_top_token': teacher_top,
                             'actual_top_token': int(logits.argmax()),
                             'teacher_top_logprob': float(log_actual[teacher_top]),
                             'kl_teacher_to_actual': float((log_ref.exp() * (log_ref - log_actual)).sum())})
            print(json.dumps({'strength': strength, 'cases': len(QUESTIONS)}), flush=True)
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base weight changed')
    summary = {}
    for strength in STRENGTHS:
        subset = [r for r in rows if r['strength'] == strength]
        summary[str(strength)] = {'top_agreement': sum(r['teacher_top_token'] == r['actual_top_token'] for r in subset),
                                  'mean_kl': sum(r['kl_teacher_to_actual'] for r in subset) / len(subset)}
    report = {'status': 'teacher_forced_blend_development_probe', 'summary': summary, 'rows': rows,
              'base_sha256': WEIGHT_SHA, 'candidate_sha256': {str(i): sha(PAIR / f'layer-{i}.pt') for i in (6, 12)},
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Six authored prompts; no verified independence from earlier development.',
                              'Only the first next-token distribution; no generated answers or semantic grading.',
                              'Memory disabled, no individual state or memory quality measured.',
                              'Blend computes both branches and prohibits KV cache; not a deployment benchmark.']}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
