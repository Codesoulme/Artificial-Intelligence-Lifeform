# Full runtime boundary and output probe

The experimental full-model loader reads the verified BF16 weight file through
ordinary file reads instead of the failing file-backed PyTorch storage path.
With one known and one near-name question, it reproduced the saved runtime's
accepted/reason/key/answer/temporal fields and both recorded gate scores. The
near-name acceptance is a reproduced error, not a successful protection test.
Only two cases were compared; the standard runtime loader remains unchanged.

Four named facts were written by numerical key to a temporary individual.
Each of four known-person questions and four near-name questions was run with
both the current runtime locator and the new boundary head. For both heads,
the known questions were accepted and routed to the expected keys 4/4; exactly
3/4 generated the expected single-word answer. Both heads also routed all four
near-name questions to existing keys, even though the new head selected the
complete unknown name in each case. Better span identification alone does not
prevent the learned address projection from mapping unknown names to known
addresses.

An explicitly enrolled exact-name fingerprint was compared *after* the real
runtime had read memory and generated a response. It would have rejected 3/4
near-name reads with the old locator and 4/4 with the candidate, while keeping
4/4 known cases. This is an offline diagnostic, not a pre-read guard, and
previous tests showed both boundary truncation and poor recall. It does not
solve aliases or names shared by different people.

For the incorrect known answer, a separate controlled rerun showed that the
native memory's top value matched the written value for all four keys; the
problem case selected the correct key and value but generated an unrelated
English response twice. This narrows the observed issue to output use or
generation in this case, but does not establish a causal explanation or prove
the memory works generally.

All examples were authored, the four identities and facts were manually
enrolled, and no real user individual was opened or changed. Numerical labels
and exact single-word answer scoring limit what can be inferred. Full local
results: `runs/deepseek-spora-full-stream-parity-20261112.json`,
`runs/deepseek-spora-boundary-runtime-pilot-20261112.json`, and
`runs/deepseek-spora-output-failure-probe-20261112.json`.

Next work must independently test a pre-read identity check and unknown-name
refusal on varied cases, and investigate the output adapter's failure on the
correct retrieved value. Do not promote the candidate or claim the current
route prevents wrong-person memory access.
