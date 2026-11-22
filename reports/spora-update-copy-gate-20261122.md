# Stateful update copy gate audit

Date: 2026-11-22

## Purpose

The current writer loses the first character of some Chinese values and sometimes
misses a changed non-typed value. This experiment added a runtime research gate after
the existing writer. It uses complete typed spans for dates, times, quantities and
directions, then uses a bounded change cue plus the previous working state for place,
person and gear updates. The gate is stateful only within the three turns of one
case; the question turn is explicitly read-only.

The gate is a diagnostic structural component, not a replacement language model and
not a learned checkpoint.

## Independent writer holdouts

| Split | Baseline exact | Gated exact | Baseline stale | Gated stale |
|---|---:|---:|---:|---:|
| Read-plan holdout, 12 cases | 8/12 | 12/12 | 1 | 0 |
| Writer-transfer holdout, 12 cases | 6/12 | 12/12 | 2 | 0 |

The holdouts use different values and paraphrased first and update turns. The gate
does not store any holdout value in a table; it copies bounded spans present in the
current turn.

## Native pipeline

The gate was connected to the automatic state input of the native read-plan audit.

| Measurement | Baseline | Gated |
|---|---:|---:|
| Automatic state + gold plan exact | 10/12 | 12/12 |
| Automatic state + predicted plan exact | 9/12 | 11/12 |
| Predicted plan routing | 12/12 | 12/12 |
| Date-isolation checks | 10/10 | 10/10 |

The remaining `11/12` is a planner formatting/order error, not a memory-state error.
The durable individual digest remained
`f8aa47a395b773f29d2901d6148f74f312ccf336a99378af5f3e4cbefb46796d`.

The 24-turn ordinary-dialogue regression is unchanged: 23/24 generations ended
normally, with 18/24 end-of-sequence outcomes matching the reference and 0/24 exact
answer matches. This gate repairs structured memory writes; it does not create normal
open-ended Chinese conversation.

## Decision

Keep the formal checkpoint unchanged. Retain the gate as a research candidate because
it cleanly separates memory write-boundary errors from the DeepSeek language-base
limitation. Before integration, run broader no-write and spontaneous paraphrase
holdouts, then replace the hand-authored cue layer with a learned typed trigger/copy
interface if the broader audit supports the same behavior.

Artifacts:

- `runs/spora-typed-update-copy-gate-v8/result.json`
- `runs/spora-copy-gate-native-read-plan-v1/result.json`
- `spora/research_update_copy_gate.py`
- `scripts/audit_spora_update_copy_gate.py`
- `scripts/audit_spora_native_read_plan.py`
