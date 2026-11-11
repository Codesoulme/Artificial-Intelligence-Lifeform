# End-to-end candidate audit

The replaced-feature subject locator, retrained finite address projection,
six-class polarity gate, open-vocabulary value span extractor, and typed write
guard were composed in a read-only audit over 182 held synthetic examples.

- 52/52 positive examples had the correct subject address.
- 32/32 positive examples had values that could map to the finite vocabulary.
- 0/130 negative examples were allowed to write.
- 20 positive unknown-value examples were deferred because the finite value
  index was unresolved.
- 0/32 known positive examples were written because their calibrated polarity
  confidence was below the generic 0.9 persistence floor.

The last result is deliberately conservative and identifies the next problem:
the parser needs calibrated joint confidence before it can produce a writable
candidate. Lowering the floor would turn a development classifier threshold
into an unsupported write authorization. The audit performed no state mutation
and no checkpoint write. Full rows and provenance are in
`runs/deepseek-spora-end-to-end-candidate-audit-20261111.json`.
