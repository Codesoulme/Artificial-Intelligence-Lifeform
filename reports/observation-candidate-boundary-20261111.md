# Observation candidate boundary

The parser now has an explicit typed boundary before durable SPORA writes:
`spora.observation_candidate.ObservationCandidate`. It carries subject
resolution, source text, optional finite value index, claim type, source,
quality, effective time and confidence. It is intentionally side-effect free.

`write_guard` refuses candidates when the claim is non-affirmative, the subject
or finite value is unresolved, the source is unknown or nonfactual, quality is
missing, effective time is missing or in the future, or confidence is below
the configured floor. A complete candidate produces only a decision object;
the caller still has to invoke the existing guarded `observe_sourced` path.

Four focused tests cover unknown open-vocabulary values, negation, a complete
direct fact, and future timestamps. This closes the contract gap between the
new language parser and the finite internal memory without claiming that the
parser itself has learned source truth or authorization.
