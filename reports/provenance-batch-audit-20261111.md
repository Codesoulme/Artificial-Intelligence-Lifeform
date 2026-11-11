# Bounded batched provenance read

`read_provenance_batch` groups finite-symbol queries into bounded batches,
uses the existing neural history and the same provenance decoder, and returns
results in request order. It does not store facts externally or select the
meaning of a natural-language time expression. The default batch is 16 and
the maximum is 64 to limit temporary tensor allocation. Total scan work remains
linear in the number of queried events.

The test creates a fresh controlled state of the protected pre-Chinese
individual using seed 8303, 96 interleaved observations, 16 active addresses
and two history pages. It checks 1,728 key/time pairs plus two unrecorded
times on CPU. All returned dictionaries matched the scalar path for each
batch size, and results were unchanged after portable checkpoint reload in
both the same and an independent Python process. The digest also matched on
the same-process reload. No base model or real conversation was involved.

| Method | Total seconds |
| --- | ---: |
| Scalar | 8.504 |
| Batch 1 | 9.480 |
| Batch 8 | 1.456 |
| Batch 16 | 0.835 |
| Batch 32 | 0.552 |

This one-run CPU comparison measures provenance retrieval only, excluding load,
generation and audit-only extra value queries. It is not comparable to the
earlier report's timings, which included additional queries. It is not a
guarantee for other hardware, thresholds, histories or batch sizes. Batched
floating-point operations can cross decision thresholds on other inputs; GPU
behavior and peak memory have not been assessed. The independent reload is
not a power-loss test. The API is not yet used by the natural-language
pipeline, and equivalence to an existing read is not proof that the read is
semantically correct.

Reproduce with `python scripts/spora_provenance_batch_audit.py` after providing
the protected individual and its dependencies. The script refuses to overwrite
the recorded output. Tests also cover reordered and repeated pairs, two pages,
unrecorded times, validation, state immutability, and checkpoint restore.
