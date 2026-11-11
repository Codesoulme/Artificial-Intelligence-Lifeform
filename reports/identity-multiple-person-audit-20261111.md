# Multiple-person memory-read development audit

The default runtime and the experimental whole-utterance identity gate were
evaluated on four single-person questions and eight questions containing both
a known person and a distinct near-name. The audit used a temporary state with
four manually populated addresses; no real user checkpoint was opened.

For this conservative protocol, every multi-person answer is counted as unsafe
because the tested route has not established a reliable mapping from question
target to person. The raw JSON reuses the label `unknown` for these cases; it
means “multi-person / defer under this protocol,” not that every name in the
sentence is unknown.

| Measure | Default runtime | With offline identity veto |
| --- | ---: | ---: |
| Single-person questions routed to expected key | 4/4 | 3/4 |
| Multi-person questions accepted | 0/8 | 0/8 |

The additional gate did not improve these examples and rejected the known
阿哲 question. The test scores keys only, not the generated answer text. It
does not negate the earlier 8/8 near-name false acceptances on single-person
questions, nor does it establish multi-person understanding. These twelve
manually authored cases are development examples; the gate remains disconnected
from the default runtime. Raw results and source hashes are in
`runs/deepseek-spora-identity-multiple-person-audit-20261111.json`.
