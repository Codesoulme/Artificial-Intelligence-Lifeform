# Reduced-prefix read and CLUENER audit

## Loading diagnosis

Two previous model-loading processes exited with Windows access violations in
`torch_cpu.dll`. With Python fault tracing enabled, the failure was reproduced
in `safetensors.safe_open(..., framework='pt').get_tensor(...)` before CUDA
transfer, both for the embedding and a small layer-normalization tensor from
the DeepSeek weight file. The underlying cause is not established. Ordinary
streaming reads of the small BF16 tensor succeeded. The research-only prefix
loader now validates and reads the BF16 data directly, after checking the full
base-weight digest; the production/full-model loader is unchanged.

The initially tried 19-layer prefix did not expose recorded hidden index 18.
In the installed Transformers version, hidden-state capture records Qwen
decoder layers but skips the two SPORA replacement classes; it also substitutes
the final recorded output with the final normalized output. The experimental
prefix therefore uses 21 physical layers to preserve the two recorded indices
used by the current router. These indices must not be assumed to identify the
same physical layers before and after layer substitution.

On ten previously saved development questions, prefix and full-model locator
spans matched 10/10 and the maximum entity-gate score difference was
0.0000367761. This does not establish equality of the hidden vectors, all
inputs, generated answers, or actual memory reads. Provenance and results are
in `runs/deepseek-spora-prefix-parity-audit-20261111.json`.

## News span evaluation

With the reduced prefix, an additional 40 CLUENER validation texts were
selected by digest, excluding the prior 40 selected texts and exact matches to
the local training text. These are news texts, not memory questions, and the
validation split had already been used for development. One reference name was
manually enrolled per single-person case; this is not a test of autonomous
enrollment or retrieving a stored fact.

| Measure | Result |
| --- | ---: |
| Default locator exact fingerprint match, one annotated name | 3/20 |
| Retrained locator exact fingerprint match, one annotated name | 11/20 |
| Both locators match that fingerprint | 2/20 |
| Default / retrained return one span on multi-name text | 4/10 / 2/10 |
| Default / retrained return one span on text without a name | 3/10 / 5/10 |

The last two rows count spans, not confirmed memory errors. The conjunction
that looked promising on near-name synthetic questions has poor recall in this
small out-of-domain set. Do not promote it to the default memory route. Full
per-example digests and limitations are in
`runs/deepseek-spora-cluener-dual-span-audit-20261111.json`.

Next work should measure a separately held-out conversational dataset, explicit
ambiguity/abstention, known-name recall and wrong-person reads end to end;
do not infer reliable memory from the above span results. Diagnose the original
full-loader failure separately before treating the streaming research loader as
a general fix.
