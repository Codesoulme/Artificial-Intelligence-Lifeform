# Explicit historical resolution transaction

The experimental chat object now offers a programmatic
`resolve_historical` operation. It requires the caller to explicitly provide
the key, event, value, effective time and confirm/reject decision. It is not
exposed through the default CLI, does not authenticate a caller, and does not
decide whether a fact is true. The existing tensor-backed metadata remains
distinct from learned semantic memory.

The operation obtains the existing working-checkpoint lock, checks the loaded
revision, clones the individual and every state tensor, records the decision
on that private copy, and writes a new integrated checkpoint. Only after the
atomic save returns does it publish the candidate in memory. A failed save or
stale revision leaves the original object unchanged; these conditions are
covered by unit tests. The operation inherits the existing full-checkpoint
save path and does not claim to solve the previously observed intermittent
native-library access violation.

In one real-model temporary test, a prior question initially abstained;
after explicitly confirming the late fact, the expected historical event and
answer were selected, the current answer stayed correct, and the previous
in-memory individual stayed unchanged. The committed temporary integrated
checkpoint was opened in a fresh process; its individual digest, decision and
selected event matched. The child restored only the individual, not the full
decoder. No real user state was opened or modified.

Separate small-model subprocess tests terminate a worker immediately before or
after the atomic file replacement. The resulting checkpoint was respectively
the intact previous state or a loadable committed state containing the
resolution. For those crash-boundary tests the adjudication validation is
stubbed; they test persistence boundaries, not fact verification or model
semantics. They do not establish disk-level guarantees under power loss or
simultaneous independent writers.

Evidence: `runs/deepseek-spora-resolution-transaction-audit-20261112.json`,
`tests/test_runtime_chat_persistence.py`, and
`tests/test_resolution_crash_transaction.py`. More diverse, independently
adjudicated histories and a safe user-facing confirmation workflow are still
needed before using this mechanism for real information.
