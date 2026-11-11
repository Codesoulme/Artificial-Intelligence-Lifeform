# Candidate value consistency at the runtime boundary

The finite beverage runtime previously accepted a candidate's value index
without checking that its source text matched the word represented by that
index. For example, a candidate carrying text `汽水` and index 6 (`水`) could
reach `observe_sourced`; an absent text or the unknown symbol index could also
reach that method. This was reproduced with a focused test before the fix.

`SporaChat.observe_candidate` now defers a candidate if its non-null value
index is outside the supported beverage vocabulary or the value text is not an
exact match for that index. This is intentionally limited to this runtime's
finite vocabulary; it does not infer synonyms, verify factuality, or resolve
open-vocabulary memory. The existing unresolved-value behavior is unchanged.
Candidate confidence and the configured confidence floor are also validated
as finite values in [0, 1].

Fifteen focused candidate, persistence, and cross-process lock tests passed.
A real-model smoke with a temporary working checkpoint still returned a
`correct` decision for a matching coffee candidate, saved the temporary state,
and reloaded its cursor. This smoke does not independently verify the recalled
answer or crash recovery. The actual user individual was not modified.

The check does not solve the known person-name collision or validate the
claimed source. Direct callers of the lower-level numeric `observe_sourced`
interface remain responsible for supplying valid numeric observations; this
patch applies to the structured-candidate entry point.
