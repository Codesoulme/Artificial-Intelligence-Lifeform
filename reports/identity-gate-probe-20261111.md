# Full-utterance identity gate: development probe

The default locator selected exactly `小林` for both `小林` and `小林子` in
the original challenge. The two projected address similarities were identical
in that context. A cutoff applied only to those projected scores cannot
distinguish those two cases.

A separate linear identity gate was trained on whole-utterance features of the
default replaced model. Known names were shared across the train, calibration
and held splits; suffixes and question templates differed. Threshold selection
used calibration only, seeking zero accepted calibration unknowns.

In the authored held split, the gate accepted 15/16 known examples and 2/48
unknown examples. Both accepted unknowns were `阿明明` under two phrasings.
When replayed on the original ten development questions that exposed the
runtime defect, it accepted 1/2 known questions and 0/8 near-name questions.
The original challenge influenced this experiment and is not a blind test.

These numbers evaluate the gate alone, not the composed runtime. The corpus is
synthetic and covers eight enrolled identities and a small set of suffixes;
it does not address homonyms, aliases, multi-person dialogue, or identity
enrollment. The calibration split is too small to certify a false-accept rate.
The candidate has not been connected to runtime or user memory. Next, assemble
separately authored near-name and normal-name cases, evaluate the full routing
chain, and compare both accepted wrong addresses and lost known recall.

Raw results and digests are in
`runs/deepseek-spora-identity-gate-probe-20261111/result.json` and
`runs/deepseek-spora-identity-gate-original-cases-20261111.json`.
