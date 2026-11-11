# Open-base memory routing and provenance scan: development audit

This is development evidence, not an open-world or independent evaluation. The
DeepSeek candidate, individual, gate weights, and full results remain local; the
scripts and selected machine-readable reports are versioned for review. Their
recorded digests identify the required artifacts. No new model weights are
included in this commit.

## Language routing

- Name recalibration retained 18/30 new drink formulations at its strict
  training threshold, versus 25/30 for the previous router. Two token-boundary
  cases could not be represented exactly. A threshold-only calibration also
  exposed a multi-entity false acceptance; tightening it rejected all 30 drink
  formulations. The new router cannot be considered an unqualified upgrade.
- The second entity gate accepted 20/20 numerical, 30/30 drink, and 9/9
  original time formulations in development checks. These and the subsequent
  read-intent gate were adapted to synthetic examples. A later 70-case audit
  nevertheless admitted one non-memory instruction, which motivated the
  separate intent gate; its second iteration retained 19/20 inspected positive
  requests and rejected 110 inspected negatives. These sets and choices were
  repeatedly examined, so the numbers are not independent estimates.
- With the newer time classifier, the composed experiment routed 8/11 positive
  questions correctly, answered 7/11 exactly, and rejected 6/6 non-target
  prompts. The earlier classifier scored 7/11 and 6/11 respectively. This is
  one entity, two scripted facts, closed reasoning, and an eight-token answer
  cap. The experiment still maps temporal classes to fixed event positions 0
  and 1. The classifier and gates are separate local files, not yet integrated
  into the saved individual's complete lifecycle.
- The originally generated temporal classifier result incorrectly labels the
  unbound-person example “小青以前喜欢喝什么？” as temporal abstention. Its
  temporal label should be PAST; identity rejection belongs to the name gate.
  The source now reflects that correction; the original generated result is
  retained unchanged, so its reported grouping must be read with this caveat.
  Another negative instruction is classified as current by the time classifier
  but is rejected by the preceding intent gate in this small composition.

## Neural provenance scan

`scripts/spora_provenance_scan_audit.py` compares a scan of the individual's
neural history against an audit-only record of writes. It uses three target
addresses, repeated and corrected facts, an inference-source event, 58
interference events, an unseen address and an explicit second page. It also
examines addresses present in the interference events. No expected event record
is supplied to the scan; it is used only for scoring. CPU time includes the
value checks, not just provenance retrieval, and is hardware specific.

| Observations | Pages | Probes | Own accepted / total | False accepts / other | First/latest selections correct | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 56 | 8/8 | 0/48 | 14/14 | 0.65 |
| 68 | 2 | 476 | 15/15 | 0/461 | 14/14 | 5.59 |

All accepted own events had the expected stored action and value in this
trajectory. The selection treats `new` and `correct` as candidate belief writes;
it does not resolve valid-time ordering, predecessor-of-current semantics,
conflicting sources, ambiguous language, sealing, or crash recovery. In
particular, first-ever is not necessarily what a person means by “previous”.
Scanning is linear in event count per queried key, so these small timings do not
establish practical scaling. The oracle uses the controller's observed action,
not an independent judgment that each write was correct. This one deterministic
trajectory cannot establish a general absence of false accepts. It would be
unsafe to connect the scanner to conversational answers on this basis.

Three additional seeded trajectories each contained 96 observations across 16
active addresses, two pages, and a mix of direct, reported and inference
sources. Scanning all active addresses plus two absent ones yielded 288/288
accepted own events, 0/4,896 false accepts, and 108/108 correct first/latest
selections. Each run took 17.2–18.3 seconds on this machine. These random
sequences share the same trained individual and are not an adversarial or
independent-model evaluation. They evaluate 18 addresses per run; timings
include additional value queries for the scoring audit.

## Next gate

Run adversarial trajectories and independent individuals, compare against a
held-out audit oracle through page growth and reload, measure misses, false
accepts, time and memory, and distinguish event order from effective time.
Only then consider integrating a bounded, uncertainty-aware selector
into the composed pipeline and checking real multi-turn behavior. Preserve the
original checkpoint and record failures as well as successes.
