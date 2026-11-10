# Natural-reasoning sampling comparison

The [official model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) recommends temperature 0.5–0.7 (0.6 suggested), no system prompt, repeated evaluation, and starting with `<think>`. It warns that bypassing thinking may hurt performance. Earlier development tests used greedy decoding and a manually closed thinking prefix, so their results should not be read as the model's normal reasoning-mode score.

## Original base

We used the normal chat template, temperature 0.6, top-p 0.95 and a 512-token maximum on two previously inspected questions, with two seeds each. The results are not a benchmark and the budget is below the evaluation budget described in the model card.

* For the post-office question, seed 4830 generated 406 tokens in 110.5 seconds and answered correctly. Seed 4831 reached 512 tokens in 135.6 seconds without a final answer.
* For the warm-water question, both seeds produced final answers after 432 and 436 tokens, taking 100.8 and 103.9 seconds. Both mentioned warm water but added unsupported speculation about the person's preferences or reasons.

Full final answers, seeds, timing and termination states are in `runs/deepseek-official-sampling-probe-20261111.json`; reasoning text was not stored. The earlier 192-token control stopped too early to reveal the first successful result. These four samples do not establish general accuracy or average latency.

## SPORA replacement diagnostic

With the same post-office question, seed 4830 and sampling settings, the layer-6/12 pair ended after 424 tokens in 93.7 seconds but its final answer was unrelated to the question. The layer-6/12/18 triple emitted an end token after 63 tokens in 15.7 seconds without closing the reasoning section or producing a final answer. The original base's matching sample answered correctly. Exact final answers and provenance are in `runs/deepseek-spora-natural-sampling-probe-20261111.json`.

This is one sample per candidate, not an accuracy comparison. To fit the 512-token budget we temporarily extended replacement position tables to 768, with untrained zeros after position 127. Thus the candidate result is confounded by both layer replacement and untrained long positions; memory was not enabled. Nevertheless, the original model's completed samples all used more than 400 tokens, exposing a practical mismatch between normal reasoning and the currently trained 128-position replacements. Further training and evaluation must account for longer generated sequences and sample multiple seeds before drawing general conclusions.
