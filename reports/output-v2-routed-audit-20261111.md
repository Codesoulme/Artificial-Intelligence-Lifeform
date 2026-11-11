# Candidate output head through the routed runtime

This development probe compares the existing output head and the eight-value
candidate with identical model, router and in-memory state. It writes four
interleaved observations for two already-bound names and asks six authored
current, prior and later questions plus four non-target questions. No test
observation was saved to the user's individual.

| Measure | Existing head | Candidate |
| --- | ---: | ---: |
| Correct positive routes | 6/6 | 6/6 |
| Exact positive answers | 4/6 | 6/6 |
| Non-target requests rejected | 4/4 | 4/4 |

This small reused-domain experiment confirms that the previous controlled
improvement can survive the complete routing path in this particular setup.
It is not an independent benchmark, and the adapter has been trained on all
eight drink values. It does not establish reliable automatic learning,
multi-turn conversation, or behavior on other properties. The candidate is
not promoted and its weights remain local.

The runtime now refuses to apply its beverage output head when the selected
native value is outside the eight-value beverage vocabulary, including the
unknown class. This prevents an unsupported finite code being treated as a
drink; native argmax can itself be wrong, and the guard is not confidence
calibration or a general-purpose type system. Four focused runtime tests pass.
