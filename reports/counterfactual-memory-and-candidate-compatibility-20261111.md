# Counterfactual memory output and candidate compatibility

The previous fresh wording audit showed that native finite-symbol reads were
correct while generated Chinese answers often ignored them. This audit kept a
question fixed, varied only the controlled value, and compared four conditions:

- original base layers without a memory call;
- answer-distilled replacement layers without a memory call;
- original base layers with the same native SPORA reader and value;
- answer-distilled replacement layers with that reader and value.

On four prompt/value pairs, the native query returned the requested value in
all eight memory conditions. The answer-distilled replacement produced the
corresponding word in 4/8 conditions; the original layers produced it in 3/8.
For the paired test where both values had to be expressed correctly, the
answer-distilled replacement passed 2/4 pairs, compared with 1/4 for the
earlier replacement used by the main route probe. Outputs changed when the
memory value changed, but change alone does not prove correct conditioning.
The no-memory outputs also differed between original and replacement layers,
so this is an intervention comparison, not a pure memory-only effect.

The v4 answer-distilled layers had reached 6/6 on their narrow held-entity
probe, which motivated this test. Their feature distribution is not compatible
with the existing name router: the recalibrated router accepted 16/30 fresh
beverage phrases, versus 26/30 for the prior router. A candidate-specific name
threshold calibration then accepted only 5/20 numeric cases and 0/30 beverage
cases under its conservative threshold. The answer layers therefore cannot be
promoted into the main route by swapping files.

This exposes an architectural integration requirement. A replacement layer
must preserve the address feature contract, or the name router, entity gate,
temporal gate and checkpoint binding must all be retrained and evaluated as a
single variant. The current answer-distilled branch improves a narrow output
probe but fails that system-level compatibility check and remains isolated.

All experiments use one protected individual, two names, finite drink values,
greedy generation and short prompts. They do not establish open conversation,
general language ability or autonomous memory. The no-memory and reader-only
conditions are retained to guide the next integration design.
