# SPORA multi-layer and active-memory interaction

## Multi-layer composition

Two independently distilled blocks replace DeepSeek layers 6 and 12. Layer 6 sits directly after the existing memory read site at layer 5; layer 12 is the earlier middle-layer candidate. No base weights are changed.

| Evaluation source | Layer 6 alone | Layer 12 alone | Both layers |
|---|---:|---:|---:|
| `conversation-development-v2`, 24 sequences / 310 positions | 86.77%, KL 0.05953 | 89.03%, KL 0.08451 | **98.06%, KL 0.00986** |
| `conversation-development`, 36 sequences / 458 positions | 83.62%, KL 0.06512 | 76.86%, KL 0.11519 | **100.00%, KL 0.00267** |

The pair's output deviation is much smaller than either single-layer deviation on both project development sets. The likely explanation is error compensation between the separately distilled blocks. That interaction is promising but fragile: the candidates share the same initialization seed and training source, and the evaluations are small project-authored sets. It must be checked on broader independently sourced text and with jointly trained blocks before relying on it.

The paired candidates also completed two 8-token greedy generations with Qwen2's KV cache. This validates the generation path with two replaced layers; the text samples are too short to assess response quality.

## Native memory interaction

The layer-6 block was then evaluated with the existing native SPORA memory reader active at layers 5 and 17. The controller, memory state and reader stayed frozen. On four current/past/later questions (134 scored next-token positions), memory-aware fine-tuning changed agreement with the integrated DeepSeek teacher from 78.36% to 79.10% and reduced mean KL from 0.23355 to 0.17966 nat. The change is small but positive. The trained block began from the no-memory candidate and received 100 updates on 12 authored temporal questions.

This confirms the replacement can run beside real persistent-memory reads. It does not demonstrate accurate memory answers; the measurement is teacher-token agreement, and only one individual's state and one fact address were used.

## Limits and next experiment

The SPORA blocks use learned absolute positions and full multi-head attention, while the DeepSeek teacher uses RoPE and grouped-query attention. Current replacements support cache prefill and single-token continuation up to the trained 128-position range. Memory-aware layer-6 tuning is based on just 12 training and 9 held temporal templates. The next experiment should jointly distill layers 6 and 12 with memory active, broaden the text source and prompt set, and measure full generated answers. These results remain development diagnostics, not open-chat or independent Chinese-language acceptance.
