# Internal reader and output bias interventions

The four-way experiment independently intervened on the two decoder-layer
memory-reader residuals and on the learned output bias. For reader-off, the
reader returned its input unchanged at both wrapped sites; for bias-off, the
output processor returned its incoming scores unchanged. The runtime still
checked the selected memory status and native value before generation. All
tests used manually written numerical facts in temporary individuals.

In 12 previously examined conditions, each native top value matched the
written value. Exact single-word answer results were:

| Decoder intervention | Correct |
| --- | ---: |
| Both reader and output bias enabled | 9/12 |
| Reader enabled, output bias suppressed | 11/12 |
| Reader bypassed, output bias enabled | 0/12 |
| Reader bypassed, output bias suppressed | 0/12 |

For each of four fixed name/question pairs, the answers in the final row were
identical across the three memory values, while reader-only answers varied
across all three. The bias-only output also varied in some cases but did not
produce an exact target answer. This supports a causal role for the integrated
reader under these specific interventions, not a general proof of usable or
semantically faithful memory.

A follow-up used four known names and two newly authored question forms, still
with the same three values and temporary numerical writes. All 24 conditions
were accepted, routed and had the intended native top value. Exact answer
results were 16/24 for both paths on, 24/24 with only the reader, and 0/24
with both disabled. For every fixed name/question, the reader-only answers
varied with all three values and the neither-path answers did not. The
bias-only condition was not repeated in this follow-up.

These are synthetic, small, related development probes with fixed beverage
symbols and known identities. They do not measure open-ended conversation,
automatic fact acquisition, same-name disambiguation, updates over time or
out-of-domain output quality. Exact word comparison is narrow, and returning
the reader input is an experimental ablation rather than an alternative runtime
architecture. The data justifies further testing of output-bias removal, not
changing default settings yet. Full local results are
`runs/deepseek-spora-reader-bias-factorial-20261112.json` and
`runs/deepseek-spora-reader-bias-fresh-20261112.json`.

Next compare all eight values and independent dialogue forms, including unknown
and temporal cases, before deciding whether to retire or retrain the bias.
