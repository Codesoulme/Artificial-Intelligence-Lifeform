# Joint SPORA block distillation with persistent memory enabled

## Experiment

Layers 6 and 12 were jointly fine-tuned while the existing native SPORA memory reader remained active at layers 5 and 17. The reader, write controller, memory state and DeepSeek weights were frozen. Twelve temporal training questions and nine distinct authored temporal holdouts were used, with one individual, one fact address and two values.

Three objectives were tried from progressively improved two-layer candidates:

| Candidate | Objective and updates | Held top-1 agreement | Held mean KL |
|---|---|---:|---:|
| `joint-memory-distill` | KL + hidden alignment, 100 | 72.39% | 0.25696 nat |
| `joint-memory-distill-v2` | lower-rate, KL-heavy continuation, 120 | 72.39% | **0.24843 nat** |
| `joint-memory-distill-v3` | explicit teacher top-1 cross-entropy + KL/alignment, 120 | 70.15% | 0.44297 nat |

The two-layer candidates that reached 98.06% and 100% top-1 agreement did so with the memory reader disabled. Turning memory on changed the activation distribution enough that this high agreement did not transfer. Memory-conditioned joint tuning reduced average KL substantially from the initial 0.44841 nat, but held top-1 did not improve. Explicitly forcing teacher top-1 labels made both metrics worse.

## Interpretation

This is evidence of a real interaction between the transplanted blocks and memory injection, not evidence that the memory system itself failed. The distillation data are extremely narrow and all share one fact key and the same two values, so the current experiment cannot identify whether the limit comes from block capacity, sequence formatting, memory residual scale, routing, or the objective. No candidate from this round should replace the earlier ordinary-language paired candidate by default.

The next useful experiment should separate memory-site residual scale from block adaptation and evaluate multiple entities/facts with independently authored memory prompts. It should also compare per-layer and joint losses on the same fixed training/holdout split. Merely adding more steps to the current nine-example holdout is not justified.

## Artifacts and limits

Exact outputs are preserved in the three `runs/deepseek-spora-joint-memory-distill*/result.json` files, with per-question results and loss traces. Candidate tensors remain local. DeepSeek's original model files and original SPORA individual are unchanged. This is teacher-token agreement on project-authored temporal questions, not a free-chat or memory-recall acceptance test.
