# Layer-18 progressive replacement with output distillation

## Motivation and sources

Hard substitution after the memory reader previously damaged answers despite a high local hidden-state cosine. [BERT-of-Theseus](https://arxiv.org/abs/2002.02925) progressively replaces modules during task training; its random module replacement for BERT is not the continuous mixture tested here. [Calibrated Progressive Distillation](https://aclanthology.org/2026.findings-acl.335/) studies difficulty-aware curriculum and target mixing for LLM distillation. Our small experiment borrows only the general ideas of staged exposure and matching output distributions; it does not implement either paper's method or establish their results here.

## Method and result

The candidate from the previous 5%-strength run was trained for 160 steps, increasing the blend from 0.1 to 0.3, 0.6 and finally 1.0. The fixed teacher was the paired SPORA layers 6/12 plus original layer 18. Training combined first-token output KL on 36 memory questions and six generic Chinese prompts with first-answer-token cross entropy on memory questions. The memory address and time slot were supplied by the program. After the prior template-loop fix, both variants are included.

On 12 held questions from two entities previously used during model development, the full-replacement candidate scored 3/12 before and 3/12 after, using the historical substring criterion. Under a stricter exact-short-answer criterion it scored 2/12 before and 3/12 after. One prior substring match was a stray character in a malformed response, illustrating why the stricter result matters. There is no convincing improvement in memory answering. For two held general prompts, post-training first-token KL against the paired teacher was 0.030 and 0.037 and the top token matched in each case; no pre-training KL was recorded, and neither continuations nor open-chat performance were measured.

The run took 351.3 seconds with a peak 4,579,364,352 allocated CUDA bytes. The complete answers, training trace, hashes and resource figures are in `runs/deepseek-spora-layer18-curriculum-distill-20261111/result.json`. The candidate stays local. This limited run distills only the first output token, uses a very small set of prompts and does not establish transfer to fresh domains or autonomous language-based memory access. More effective methods require full-sequence output preservation and independently held-out memory and general-language evaluation before selecting any candidate.
