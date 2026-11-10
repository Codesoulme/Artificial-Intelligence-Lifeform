# SPORA memory strength versus decoder-layer replacement

## Method

Four entities each received an old value and a corrected value in a fresh controlled SPORA state. Twenty-four authored questions covered current, past and later facts. With the same integrated DeepSeek base as reference, we compared three structural candidates (the ordinary two-layer pair, memory-conditioned layer 6 plus ordinary layer 12, and the v2 joint-memory pair) at memory-read residual strengths 0.1, 0.25, 0.5 and 0.75. Each candidate was compared against the original DeepSeek layers with the same memory strength and same per-query address/page.

## Results

| Read strength | Ordinary pair top-1 agreement / KL | Memory-conditioned 6 + ordinary 12 | Joint-memory v2 |
|---:|---:|---:|---:|
| 0.10 | 41.7% / 1.578 | 33.3% / 1.518 | 33.3% / 1.442 |
| 0.25 | 0.0% / 2.998 | 0.0% / 2.888 | 4.2% / 2.445 |
| 0.50 | 0.0% / 4.614 | 0.0% / 3.926 | 8.3% / 2.701 |
| 0.75 | 0.0% / 4.969 | 16.7% / 3.769 | 37.5% / 2.001 |

KL is measured in nats. Lower values mean the candidate distribution is closer to the original model at the same memory strength. The sweep does not show a stable strength setting where layer replacement preserves the teacher's output. Weak memory injection reduces output deviation for the ordinary pair on this set, but it also weakens the memory signal. The joint-memory candidate is closer than the other candidates at strengths 0.25–0.75, but its top-1 agreement remains low.

## Interpretation and boundary

This supports an activation-distribution interaction between native memory injection and the substituted decoder blocks. Simply reducing the memory residual does not resolve it. The next measurement needs generated-answer value accuracy, because teacher-token agreement is only a proxy. Then training should use diverse entities, several independent facts and counterfactual value assignments, keeping a separate held-out entity split.

The 24 questions use four entities and one two-value correction pattern. The metric is next-token agreement with a model reference, not correctness of retrieved facts or helpfulness of responses. No read-strength value is being promoted as a production setting. Exact per-query outputs are in `runs/deepseek-spora-memory-strength-factor-20261111.json`.
