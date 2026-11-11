# Event-key index and longer-history probe

On one controlled history, a confirmed delayed fact remained retrievable after
64 unrelated writes and a second page was allocated. At 128 unrelated writes,
the unindexed effective-history selector abstained because provenance at event
94 was uncertain; it continued to abstain at 256 and 512. Diagnostic replay
showed that event 94 actually belonged to another key, but the target key's
neural historical query classified it as a known value (softmax top 0.380,
margin 0.090). In this specific history, 1 of 130 other-key events was
classified as known for the target, causing the conservative abstention. This
does not establish a population error rate.

An experimental opt-in index stores the numerical key written for each event
as an `int64` tensor in the individual's state. It is exact auxiliary metadata,
not a learned weight or a replacement for native memory values. Only a fresh
controlled state can enable it: an existing history cannot be safely backfilled
from uncertain neural guesses. Writes append the index alongside memory state;
the existing tensor codec and transactional store preserve it. The selector
uses the index to select candidate event positions, then still decodes neural
provenance and historical values. It does not fix incorrect name-to-key
routing, guarantee neural recall, or make tampered valid-looking keys safe.

The following single-trajectory CPU comparison used identical neural state
with and without the index. Times are medians of three unisolated selector
calls on two CPU threads; they exclude generation, I/O and writing:

| Additional writes | Unindexed result / scanned | Indexed result / scanned | Median seconds, unindexed / indexed | Allocated index |
| ---: | --- | --- | --- | ---: |
| 0 | correct / 5 | correct / 3 | 0.0116 / 0.0123 | 512 bytes |
| 64 | correct / 69 | correct / 3 | 0.0385 / 0.0156 | 1,024 bytes |
| 128 | abstained / 133 | correct / 3 | 0.0670 / 0.0113 | 1,536 bytes |
| 256 | abstained / 261 | correct / 3 | 0.1484 / 0.0115 | 2,560 bytes |
| 512 | abstained / 517 | correct / 3 | 0.2340 / 0.0169 | 4,608 bytes |
| 768 | abstained / 773 | correct / 3 | 0.3533 / 0.0115 | 6,656 bytes |

The index still compares keys across the recorded timeline; the number of
expensive neural provenance reads is reduced here, not proven constant for
arbitrary workloads. Six previously used delayed-fact trajectories also
retained their selected event after 64 other-key writes and same-process
save/reload, but are not independent or representative. At 768 writes the
indexed individual's same-process reload matched the selected event and state
digest. Exact-value metadata may have privacy and retention implications; it
is not enabled on the real individual or by the standard runtime.

A full-decoder check repeated the 128-interference trajectory in two temporary
states whose neural tensors were exactly equal. The unindexed runtime
abstained on uncertain provenance; the indexed route selected the correct
historical event and key but its default output adapter generated the wrong
drink. Temporarily suppressing that output bias produced the expected word in
this single case. This is not an end-to-end fix or evidence that suppressing
the bias is generally safe. Details are in
`runs/deepseek-spora-indexed-runtime-audit-20261112.json`.

Local evidence: `runs/deepseek-spora-resolution-interference-audit-20261112.json`,
`runs/deepseek-spora-history-scan-scaling-20261112.json`,
`runs/deepseek-spora-cross-key-history-probe-20261112.json`, and
`runs/deepseek-spora-indexed-history-audit-20261112.json`. Broader names,
same-key dense histories, process crashes, write costs and open-ended dialogue
remain untested by this experiment.
