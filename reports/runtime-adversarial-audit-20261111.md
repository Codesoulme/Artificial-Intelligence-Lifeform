# Isolated runtime adversarial audit

The evaluated runtime loaded the experimental output-v2 candidate, not the
default interactive checkpoint. A temporary, in-memory individual received two
manually addressed observations for one entity: coffee followed by water.
No user checkpoint was modified.

All three authored positive questions were accepted and returned the expected
value (two current water, one previous coffee). All 17 authored negative
questions were rejected: three negations, two requested writes, five questions
about other attributes, two hypotheticals, two mixed requests, one unwritten
entity, and two other domains. The raw rows and checkpoint digest are in
`runs/deepseek-spora-runtime-adversarial-audit-20261111.json`.

This is a small, manually authored test of a single controlled entity. It is
not an independent corpus or evidence of generalized intent understanding,
automatic memory writing, open-ended memory, or real-world reliability. The
guard still relies partly on manually selected beverage keywords, and refusing
an entire mixed request also loses the answerable part. Future evaluation
needs paraphrases and adversarial negatives authored separately from both
training and this audit, multiple known and unknown entities, more predicates,
and per-category false-accept and false-reject measurements.
