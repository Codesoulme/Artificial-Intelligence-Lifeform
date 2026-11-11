# Fingerprint and predicted-span development audit

The default DeepSeek/SPORA read route was evaluated with a temporary in-memory
individual. An identity fingerprint was explicitly enrolled for each known
name, and a read-only offline veto required the fingerprint of the locator's
predicted text span to match the runtime's predicted address. The default
runtime itself was not changed; the veto was measured after generation.

On the original ten development questions, the baseline accepted 8/8 unknown
near-name queries. The default locator plus fingerprint veto accepted 2/8, both
because `小林子` had been truncated to `小林`. A separate retrained locator plus
the fingerprint veto accepted 0/8 and retained both known-name reads.

On a subsequent twelve-question set with four explicitly enrolled names and
eight near-names, the default route accepted 8/8 unknowns, the default locator
plus fingerprint accepted 3/8, and the retrained locator plus fingerprint
accepted 1/8. The latter failure truncated `小雨晴` to `小雨`. Both locator
versions retained the four known queries in that specific set. These metrics
score routing acceptance and expected keys, not the correctness of answer text.

Retrospectively requiring both locator/fingerprint checks to pass retained
2/2 known queries and rejected 8/8 unknowns on the original set, and retained
4/4 known queries and rejected 8/8 unknowns on the second set. This conjunction
was proposed after observing the component results, and the locators share the
same feature-producing model; their errors can be correlated. Neither set is a
blind or population-representative evaluation, and names were manually bound
in temporary state. This does not demonstrate robust identity or solve
homonyms, aliases, authorization, or dynamic enrollment.

The fingerprint remains deterministic metadata stored in tensors, not learned
semantic identity. No candidate was promoted and no real user checkpoint was
opened or modified. The full results and provenance are in
`runs/deepseek-spora-fingerprint-route-audit-v2-20261111.json`,
`runs/deepseek-spora-fingerprint-fresh-audit-20261111.json`, and
`runs/deepseek-spora-fingerprint-conjunction-20261111.json`.
