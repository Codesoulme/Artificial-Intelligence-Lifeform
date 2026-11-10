# One-layer SPORA structure replacement pilot

## Scope

The local DeepSeek-R1-Distill-Qwen-1.5B has 28 Qwen2 decoder layers, hidden width 1536, 12 query heads and 2 key/value heads. SPORA's `CausalBlock` uses full multi-head attention, learned absolute position embeddings and a 4x GELU feed-forward block. Qwen uses grouped-query attention with RoPE and a wider SwiGLU block. Because the internal operations differ, this pilot replaces one middle decoder layer with a randomly initialized SPORA causal block and distills it from the frozen teacher; no DeepSeek model file or original SPORA individual is changed.

The replacement wrapper matches the full-sequence Qwen2 decoder call, casts outputs back to the teacher dtype, and now supports Qwen2's dynamic KV cache for prefill plus one-token incremental decoding. The experiment selected layer 12, used 48 authored Chinese training pairs and 24 project development prompts, and scored 310 next-token positions.

## Distillation comparison

| Candidate | Training | Held hidden cosine | Next-token top-1 agreement | Mean KL |
|---|---:|---:|---:|---:|
| Hidden-state fit | 240 steps | 0.9731 | 87.42% | 0.05098 nat |
| Hidden-state fit | 720 steps | 0.9740 | 86.77% | 0.05072 nat |
| Logit KL from random | 80 steps | 0.9148 | 83.55% | 0.12733 nat |
| Warm-start + logit KL | 160 steps | 0.9725 | **88.71%** | 0.05039 nat |
| Low-rate KL continuation | 160 steps | **0.9733** | 87.74% | **0.04909 nat** |

Longer local hidden fitting improved its held hidden loss but did not improve downstream top-1 agreement. Starting final-logit distillation from the 240-step hidden candidate raised top-1 agreement by 1.29 percentage points. Continuing at a lower learning rate reduced KL further while losing about one percentage point of top-1 agreement. The warm-start candidate is the current choice when exact top-1 agreement is weighted more; the continuation candidate has the lowest KL. Neither is a generally validated replacement.

The 240-step hidden-fit starting candidate SHA-256 is `9b0758b6379c627036ffdc7444c68786d8ac7738db8df2b8bf92c7e6c27f295f`. The 160-step warm-start KL candidate SHA-256 is `4b4c0e370d90c5e3bcbf8d5eab11a82f111cf0193da1d02ad2b36e3f37334307`; the low-rate continuation SHA-256 is `21367b8f08762177d88526998fbb65fa75fe98836daee7f28efff0f985ea01eb`. Candidate tensors remain local because each is about 114 MB; the experiment code and result summaries are versioned, and the original base files remain hash-bound.

## Decision and limits

The result demonstrates that one SPORA-native transformer block can substitute for a middle DeepSeek layer with measurable but nontrivial output deviation after less than three minutes of local distillation. Cached generation completed for two prompts, eight tokens each, using the 88.71% candidate; this is a runtime smoke test only. One short answer begins with a generic uncertainty response, so it does not validate answer quality. Subsequent two-layer and active-memory probes are summarized in `reports/deepseek-spora-multilayer-and-memory-interaction-20261111.md`.

The training material and held scenarios are project-authored and small. Their scores are architecture diagnostics only; they are not independent Chinese-chat acceptance results.
