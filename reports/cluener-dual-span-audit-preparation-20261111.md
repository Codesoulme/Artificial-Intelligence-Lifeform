# Cross-domain dual-locator audit preparation

The retrospective fingerprint conjunction needs evaluation beyond its
handwritten near-name examples. A read-only audit has been prepared using
CLUENER validation text: it chooses the next 20 single-name, 10 multiple-name,
and 10 no-name examples by text digest, excluding the 40 examples used in the
earlier selected audit. It also checks the selected texts against the entire
local locator-training split for exact overlap.

The source-only check selected 20/10/10 examples, with zero exact overlaps
against either the prior selection or local training texts. Two CPU tests pass
for selection deduplication, exclusion, and failure on an insufficient group.
The planned metric compares both predicted name spans with an explicitly
oracle-enrolled name fingerprint on single-name examples. This is a locator
diagnostic, not an end-to-end dialogue or memory retrieval test.

The first real-model inference process exited during loading with exit code 1
and no traceback; no audit result file was produced. Other GPU activity was
present, but the cause is unverified. Do not treat this attempt as a model
result. The task can be rerun later with the same script when resources permit.
Even if it completes, this is news-domain development data, and previous work
used the same validation split. Annotation errors, tokenizer boundary merges,
paraphrase overlap, and base pretraining overlap remain possible.
