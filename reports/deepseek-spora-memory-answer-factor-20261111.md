# Generated answers under native SPORA memory and decoder substitutions

## Method

Four entities received one old and one corrected drink preference each. The model answered current, past and later questions for each entity, giving 12 prompts at each memory-reader residual strength. We compared the unchanged DeepSeek decoder with the ordinary layer-6/12 pair, a memory-conditioned layer-6 candidate, and the v2 joint-memory pair. Each condition used greedy generation up to eight tokens; correctness required the expected drink word to appear without the conflicting drink word.

## Results

| Read strength | Original DeepSeek | Ordinary pair | Memory-conditioned layer 6 | Joint-memory v2 |
|---:|---:|---:|---:|---:|
| 0.10 | 1/12 | 2/12 | 0/12 | 0/12 |
| 0.25 | 1/12 | 1/12 | 0/12 | 0/12 |
| **0.50** | **12/12** | **0/12** | **0/12** | **0/12** |
| 0.75 | 8/12 | 0/12 | 0/12 | 5/12 |

At the native 0.5 setting, the unchanged integrated model answers all 12 controlled questions correctly. Every replacement variant loses that behavior. Lowering the memory strength does not help: the base model also loses recall when the memory signal is weakened. At 0.75, the joint candidate recovers 5/12, but the original model reaches 8/12.

The ordinary pair often echoes the prompt or emits unrelated fragments. The joint v2 pair answers all four current queries correctly at 0.75 but fails the past and later questions in this small batch. This is a concrete example of the gap between teacher-forced next-token similarity and generated memory use.

## Interpretation

The paired candidates that matched next-token distributions on ordinary development text do not preserve the base model's memory-conditioned answer behavior. Reducing the reader strength suppresses the original memory capability as well, so it is not an acceptable compensation. The next distillation objective should include full generated answer trajectories under active native memory, followed by held-out entities and values. The current candidates should remain experimental and must not replace the working integrated checkpoint.

## Limits

Only four entities, one fixed old/current value pair, three authored temporal forms and eight generated tokens were tested. Exact-word scoring can reject a correct paraphrase or short response. The result is development evidence, not independent memory safety or open-chat acceptance. Per-question responses are preserved in `runs/deepseek-spora-memory-answer-factor-20261111.json`.
