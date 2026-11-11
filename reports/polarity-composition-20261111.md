# Subject, polarity, and value composition

An independent six-class claim gate was added after the subject locator and
before the open-vocabulary value span. Its classes are affirmative,
negation, hypothesis, third-party report, action performed for someone else,
and question/history. It was trained on the replaced representation and its
threshold was selected on a separate calibration split.

On the held synthetic split, the three-way composition accepted 52/52
affirmative observations and 0/130 non-affirmative examples. The value span
probe alone accepted 24/130 negative examples; the subject gate alone reduced
that to 2/130; the polarity gate removed the remaining two negation cases.
Unknown beverage strings remain source spans rather than being mapped to a
fixed vocabulary.

This is a development result, not evidence of general reliability. The
examples are synthetic, the threshold is calibrated on a finite sample, and the
chain still lacks source trust, temporal resolution, correction semantics,
authorization, and address binding in this audit. It has not been connected to
persistent memory writing. Full rows and artifact provenance are in
`runs/deepseek-spora-memory-parse-polarity-composition-20261111.json`.

The next step is to validate the composed parser on independently authored or
licensed Chinese dialogue and then test controlled persistence only for the
subset whose subject, polarity, source and time are all resolved.
