# Composed identity-gate development audit

The default runtime was loaded with a temporary working path and an isolated
state. Four manually addressed known people were assigned finite drink values.
The actual runtime was queried on twelve authored questions: four for those
known people and eight for distinct near-names. The separately trained
whole-utterance identity gate was evaluated as an offline veto on exactly the
same runtime results; it was not installed into the runtime.

| Measure | Default route | With offline gate veto |
| --- | ---: | ---: |
| Known queries routed to the expected key | 2/4 | 2/4 |
| Known queries rejected | 2/4 | 2/4 |
| Unknown queries accepted | 4/8 | 0/8 |

The veto blocked the four default-route false accepts in this development
sample. Both known rejections occurred in the existing runtime before the
additional gate. This experiment scores address selection only; the answer text
is recorded but was not judged. The gate still accepted 2/48 unknown examples
in a separate authored suffix split, so this result does not establish an
acceptable open-world error rate.

The questions were written after the earlier failure was known, only four
people and eight near-name queries were evaluated, and the pretraining overlap
is unknown. No real individual or model files were changed. The runtime still
has the reported name-collision issue, and the experimental gate remains
disconnected. Raw rows, threshold, and hashes are in
`runs/deepseek-spora-identity-gate-composed-audit-20261111.json`.
