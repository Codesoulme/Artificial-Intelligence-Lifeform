# Learned candidate span writer audit

Date: 2026-11-22

## Purpose

The existing writer predicts independent start and end boundaries. A research
candidate instead scored `no-write` versus every complete character span up to the
same span limit. Each candidate used its start vector, end vector, mean vector and
length embedding. The shared character and frozen language features were initialized
from the current writer; only the candidate scoring head was new.

## Results

| Threshold | Read-plan holdout | Writer-transfer holdout |
|---:|---:|---:|
| `2.0` | 0/12 exact, 12 missing cases | 0/12 exact, 12 missing cases |
| `0.0` | 1/12 exact, 11 missing cases | 0/12 exact, 12 missing cases |

At the default threshold the head learned to prefer `no-write` almost everywhere.
Removing the threshold did not recover the spans and introduced additional wrong and
stale values. The training loss decreased, but the independent state metrics did not.

## Decision

Reject this candidate and do not connect it to native generation. The result rules out
a simple replacement of the boundary head as the current fix. The stateful update
copy gate remains the better research direction because it uses the previous working
state and an explicit update transition; its two independent writer holdouts reached
12/12 with zero stale values.

This experiment does not alter the formal checkpoint or the frozen language base.

Artifacts:

- `runs/spora-candidate-span-writer-v2/result.json`
- `runs/spora-candidate-span-writer-v3/result.json`
- `spora/research_candidate_span_writer.py`
- `scripts/audit_spora_candidate_span_writer.py`
