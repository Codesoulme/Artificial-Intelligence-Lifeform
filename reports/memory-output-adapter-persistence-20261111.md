# Persisted memory output adapter

The memory output path now consists of a small `MemoryOutputAdapter` and a
one-shot logits processor. The adapter maps native finite SPORA value logits to
a bounded vocabulary bias. The processor applies that bias only to the first
generated token; later tokens follow the decoder normally. This avoids the
repetition observed when the same bias was added at every generation step.

On the controlled adapter training probe, the earlier candidate layers
contained the expected value word in 12/16 outputs. The adapter-conditioned
outputs contained it in 16/16. The probe uses two entities, two finite values
and four authored prompt forms; it is not an open-chat result.

The adapter was then stored in a new integrated checkpoint format
`spora-integrated-open-base-v4` alongside the existing individual state,
native reader and candidate replacement-layer metadata. A separate Python
process loaded the checkpoint and verified the adapter configuration, every
adapter tensor, replacement variant and individual state digest. The loaded
checkpoint was used to generate two controlled values:

| Native value | Generated answer | Correct value word |
| ---: | --- | ---: |
| 1 | the corresponding value word | yes |
| 6 | the corresponding value word | yes |

The persisted runtime probe took about 119 seconds on the local GTX 1060,
including model loading and generation. The checkpoint digest is recorded in
`runs/deepseek-spora-memory-output-integrated-v5-20261111/result.json`; the
large checkpoint itself remains local. Base weights remain external and are
checked by SHA-256. The caller still has to provide the routed memory key and
the current implementation is limited to finite value logits.

Old v1/v2/v3 integrated checkpoints remain loadable. Core checkpoint,
integrated decoder, output adapter and persistence tests passed after the v4
extension.
