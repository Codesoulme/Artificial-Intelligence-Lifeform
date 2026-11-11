# End-to-end candidate audit — correction

The original audit used a name-to-key answer table when constructing candidates,
and the reported address-correct count only counted accepted addresses. Its
52/52 claim did not establish correct routing. The audit has been corrected:
candidates use the predicted key, acceptance and correctness are reported
separately, and polarity/value rejection reasons are included in the final
guard. No writes were performed by either version.

The replaced-feature subject locator, retrained finite address projection,
six-class polarity gate, open-vocabulary value span extractor, and typed write
guard were composed in a read-only audit over 182 held synthetic examples.

- In the corrected rerun, 52/52 positive examples had the correct predicted
  subject address, compared against the expected label for scoring only.
- 32 positive examples had ground-truth values in the finite vocabulary;
  overall, 50/52 positive values were extracted exactly.
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
`runs/deepseek-spora-end-to-end-candidate-audit-corrected-v2-20261111.json`.

These are small, synthetic development examples, not an independent safety
evaluation. Direct source and effective time were supplied by the experiment.
Some templates overlap component development data. A probability threshold on
the polarity gate does not validate the joint candidate or prove source trust.
