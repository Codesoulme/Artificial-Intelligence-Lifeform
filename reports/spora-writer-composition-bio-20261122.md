# Writer composition and trigger BIO audit

Date: 2026-11-22

## Scope

This audit kept the frozen DeepSeek backbone, formal correction checkpoint, durable
individual and read planner unchanged. It compared two research writers against the
12-case independent read-plan holdout.

## Compositional writer

The candidate expanded training with unseen quantity compositions, additional update
wording and new place and gear values. It started from the current transfer writer
and trained for five epochs.

| Measurement | Current transfer writer | Compositional candidate |
|---|---:|---:|
| Writer holdout state exact | 8/12 reference | 9/12 |
| Missing fields | reference | 2 |
| Wrong fields | reference | 1 |
| Stale changed values | reference | 0 |
| Native automatic + predicted plan exact | 9/12 baseline | 9/12 |

The local writer improvement did not survive the complete native pipeline. The
remaining errors were a missing day and quantity boundary errors. The date-isolation
checks and durable individual digest remained unchanged. This candidate is retained
as research evidence and is not promoted.

## Trigger-gated BIO copier

The second candidate added a field trigger classifier and BIO character tags so a
field would be copied as one contiguous span. It used the same holdout separation and
1,584 training examples. After five epochs it reached `0/12` exact states, with two
stale values. Its spans routinely absorbed surrounding punctuation and clauses, for
example copying an entire update clause as a place or time. The candidate is
rejected; no native read-plan run was attempted.

## Decision

The formal checkpoint and current transfer writer remain unchanged. More generic span
head replacement is not justified by these results. The next research step is a
separate update-aware copy gate: identify the changed field and select a bounded value
candidate using the previous working state, while keeping the existing writer as a
fallback. It will be evaluated as a candidate before any integration.

Artifacts:

- `runs/spora-compositional-writer-v1/result.json`
- `runs/spora-compositional-native-read-plan-v1/result.json`
- `runs/spora-bio-value-writer-v5/result.json`
- `scripts/audit_spora_writer_compositional.py`
- `spora/research_bio_value_writer.py`
- `scripts/audit_spora_bio_value_copy.py`
