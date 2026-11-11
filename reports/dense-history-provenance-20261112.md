# Dense same-key history and provenance confidence

The optional event-key index cannot prune a history in which every event
belongs to the same key. In one controlled trajectory alternating two values,
the selector returned the expected previous event through 32 writes, but
abstained at 64, 128 and 256 due to uncertain provenance. The current value
and oracle-addressed previous value were correct at all seven measured
lengths. At 256 writes, same-process save/reload preserved the result.

Inspecting every event in that trajectory separated the failure:

| Writes | Correct historical values | Fully decoded provenance |
| ---: | ---: | ---: |
| 8 | 8/8 | 8/8 |
| 32 | 32/32 | 32/32 |
| 64 | 64/64 | 57/64 |
| 128 | 128/128 | 114/128 |
| 256 | 256/256 | 228/256 |

The uncertain entries had source/action classifications but ambiguous numeric
bits. This is one synthetic trajectory, not a general memory-capacity figure.

The index also exposes a safety distinction: if it says an event belongs to
the requested key but neural history calls the event unknown, the selector now
abstains with `indexed_event_unreadable` instead of silently skipping it. The
legacy unindexed behavior is unchanged because it cannot distinguish absent
events from missed reads. A unit test covers both selection styles.

As a diagnostic only, the bit-ambiguity range was narrowed from 0.2–0.8 to
0.3–0.7. On eight new synthetic numeric keys with varied sources, qualities and
effective times, the original band accepted 482/512 correctly decoded events
and the candidate accepted 512/512; eight further interleaved trajectories
had 512/512 correct under both. In twelve dense direct-write trajectories, the
original selector abstained in all 12; a temporary candidate band selected the
correct previous event in 6 and abstained in 6. These tests use related finite
keys, inputs and prior analysis; they are not an independent deployment gate.

Artificial Gaussian perturbations were then added to provenance tensors, not
to real memory and not as a simulation of actual sleep. At noise RMS equal to
40% of the provenance tensor RMS, the 0.3–0.7 band accepted 9 of 256 event
reads, 3 of which contained incorrect value or provenance; the existing band
accepted 3, none incorrect in this sample. This does not prove any threshold
safe under real disturbances, but demonstrates the risk of simply loosening
confidence. The default threshold remains unchanged.

The expected action in raw-decoding probes was the controller's own action,
so those measurements do not assess whether its decision was correct. No
natural-language facts, real user individual, actual sleep compression or
full-decoder answers were tested. Results are stored under `runs/` in
`deepseek-spora-same-key-density-audit-20261112.json`,
`deepseek-spora-same-key-retention-probe-20261112.json`,
`deepseek-spora-provenance-bits-probe-20261112.json`,
`deepseek-spora-provenance-holdout-probe-20261112.json`,
`deepseek-spora-provenance-interleaved-probe-20261112.json`,
`deepseek-spora-provenance-band-selector-pilot-20261112.json`, and the two
`deepseek-spora-provenance-noise-stress` result files.
