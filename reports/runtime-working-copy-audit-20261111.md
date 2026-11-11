# Runtime working copy and predicate audit

The manual `--observe` path previously wrote directly over the integrated
bootstrap checkpoint. Atomic replacement protects against partial writes but
does not preserve the previous individual. The entry point now reads its
dedicated working checkpoint if present and otherwise starts from the original
bootstrap. Observations save to the working path before publishing the new
in-memory state. The bootstrap remains available for recovery.

The runtime is still a single-writer prototype: concurrent processes can race
and overwrite one another's working state. It is not wired to the incremental
TensorStore, does not create a versioned checkpoint after every write, and has
not undergone a real process-kill recovery test for this new path. The tests
mock the save call to check destination and failure ordering; they do not
prove disk-level recovery or independent-process behavior.

The beverage guard previously accepted the generic word `偏好`, which also
appears in color or occupation questions. This cue was removed; explicit
beverage or drinking terms remain. It can now reject valid implicit questions,
and it does not reliably identify the subject of a mixed-domain question. This
is a temporary limitation, not a general predicate classifier.

On the existing 16-question negative set, the corrected v2 entity gate and
updated runtime rejected 16/16. Three focused unit checks passed, including
generic color/job preferences, bootstrap preservation and failed-save state
ordering. The prior positive results used the old entity-gate pairing or a
broader cue list and should not be attributed to this version without a rerun.
