# Runtime provenance correction

The unified runtime had loaded the v1 entity gate even though its v4 read-intent
gate was trained and calibrated against the v2 entity gate. Both gate files
passed their own digest checks, but the runtime did not check the cross-file
binding. This error affects the interpretation of the earlier unified audits:
their results describe the v1-entity/v4-intent combination and do not establish
the performance of the intended v2-entity/v4-intent combination. The old
results remain unchanged for traceability.

The runtime now loads the v2 entity gate and verifies the router's base,
tokenizer configuration, structural variant and individual address basis, as
well as the identity and variant bindings of the entity, intent and temporal
gates. One small real-model smoke run after the correction answered three
controlled current/history questions and rejected one non-memory request.
This does not replace the larger holdout: an attempted full rerun exited
without a completed result and must be repeated.

The previous first-token retry compared the answer against the output adapter's
own preference, not independent evidence of factual correctness. It could
reinforce an incorrect token. The default runtime no longer performs that
retry; this is not a claim that answers are now verified.

The integrated runtime also rejects output conditioning for native unknown
memory and for future history positions. Native unknown detection is a model
prediction, not a guarantee that retrieval is correct. The 15 checkpoint
tests pass, including the newly added invalid-time and unknown checks.
