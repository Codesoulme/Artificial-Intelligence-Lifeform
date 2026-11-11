# Domain-broader intent gate v5 failure

I trained a v5 read-intent candidate from the validated v4 gate with extra
positive templates for color, sport, work and profile queries, plus negative
templates for guessing, writing, editing and recommendations. The goal was to
replace the temporary beverage-domain predicate guard with a learned
domain-independent predicate.

The candidate regressed on the existing held development groups:

- known memory acceptance: `10/20`, versus v4 `20/20`;
- fresh/non-memory and identity rejection remained strong;
- calibration raised the read threshold to `3.535`.

This candidate is not used by the runtime. The result shows that adding more
templates to the same frozen-feature linear gate changes its decision boundary
and loses previously learned memory queries. The next design should use a
separate predicate classifier or a calibrated ensemble with preservation loss,
then evaluate it on held-out domains before replacing v4.
