# Observation novelty signal

The write path now computes an auditable novelty signal without changing the
existing write controller input or action policy. It records four components:
new fact, value change, quality gain, and effective-time change. A new fact and
a correction receive maximal novelty; an identical repeat at the same time and
quality is quiet. The signal is returned with every sourced observation so it
can be collected for a later controller revision.

The real runtime temporary-checkpoint smoke reported `value_change=1.0` and
`novelty=1.0` for a water-to-coffee correction. The correction was saved and
reloaded with the same cursor, and the real working checkpoint was untouched.
The focused suite passed 22 tests.

This signal is currently diagnostic. It does not override source trust,
confidence, or the existing controller, and it is not evidence that a language
parser has understood novelty. The next training experiment should compare
new facts, corrections, repeats, quality upgrades and repeated noise before
using the signal in a new write controller.
