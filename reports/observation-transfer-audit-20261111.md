# Observation probe transfer to structural replacements

The existing factorized observation probe was trained on frozen, unmodified
DeepSeek representations. This audit ran the same probe and 20 authored
development examples first on that base, then on the output-v2 experimental
checkpoint with SPORA replacements at layers 6 and 12. Memory reading was
inactive; no predicted observation was written.

The base classified 17/20 as expected; the replaced model classified 16/20.
All six selected positives were correct in both conditions. Among 14 negatives,
the base predicted a value for three and the replaced variant for four. Both
misread a reference to coffee-colored painting, a gift of juice, and a report
of another person's preference. The replaced variant additionally misread a
reported negation. Raw examples, logits, and artifact digests are preserved in
`runs/deepseek-spora-observation-transfer-audit-20261111.json`.

The contrast shows that feature provenance matters, but this small and partly
training-template-overlapping set cannot estimate generalization or prove that
the replacement caused a broad degradation. The larger finding is that the
existing probe is unsafe as an autonomous write trigger even on the original
base: a beverage word is not enough to establish a person's preference. The
third-party report also cannot automatically be treated as a direct source.

Keep the classifier disconnected from persistent writes. Next, develop and
evaluate a version fitted to the replaced representation, with distinct
training, calibration and held-out examples targeting subject attribution,
negation, hypothetical claims, temporal changes, and unseen vocabulary. Test
the composed entity and source routing separately from value prediction;
measure false writes and false rejections on independently authored and real
dialogue before attempting isolated persistence.
