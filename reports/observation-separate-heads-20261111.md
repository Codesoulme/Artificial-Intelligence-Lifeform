# Separated observation heads

The fact head was trained to recognize affirmative statements and the value
head was trained only on affirmative known or unknown values. Negative clauses
were used for fact detection, not as unknown-value labels. The experiment uses
the same replaced-model features and original synthetic split as the previous
candidate, with additional unsupported beverages unique to each split.
Threshold selection required zero observed acceptance of negative and unknown
examples in the calibration set, maximizing correct known positives.

| Held set | Previous candidate | Separated candidate |
| --- | ---: | ---: |
| Known values correct | 8/32 | 27/32 |
| Negative cases accepted | 0/128 | 0/128 |
| Unsupported values accepted | 0/16 | 4/16 |

Examples of unsupported values mapped to a known value included soda classified
as water and apple juice as juice. Such errors change facts and make the new
candidate unsuitable for automatic writing. The prior candidate rejected all
16 unsupported examples, but its threshold was selected without unknown-value
examples, so this result is not proof of robust unknown detection.

This is a small synthetic development comparison, with only four distinct held
unknown beverage names and two shared affirmative templates. Calibration saw
only four unknown examples. Counts are not population estimates. The model
does not handle general entities, predicates, source attribution, or real
dialogue, and no experiment wrote to an individual. Full rows and artifact
digests are in `runs/deepseek-spora-observation-separate-heads-20261111/result.json`.

Further work needs more varied unknown values and natural text, separately
authored challenge sets, and an explicit evaluation of whether a value can be
represented without silently coercing it to a different one.
