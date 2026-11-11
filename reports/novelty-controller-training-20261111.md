# Novelty-aware controller candidate

An 11-input `NoveltyWriteControl` was trained on 12,000 synthetic protocol
traces and compared with a newly trained legacy 10-input controller on 4,000
held traces. The target actions were new, correct, support, repeat and defer.

The legacy controller reached 4000/4000 held accuracy; the novelty controller
reached 3999/4000. This does not show that the added feature improves policy
classification because the existing ten features already expose known state,
value equality, quality and freshness. The experiment does validate the
versioned candidate artifact and its training path, but it is insufficient for
adoption.

The novelty controller remains disconnected from all real checkpoints. The
next dataset should come from real runtime trajectories and include repeated
noise, conflicting reports, source changes, quality changes, and corrections;
the held set must be independently authored so an improvement cannot be
explained by replaying the same synthetic rule distribution.
