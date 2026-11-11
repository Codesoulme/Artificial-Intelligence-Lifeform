# Subject locator retrained after SPORA replacement

The existing name-span locator was trained on original DeepSeek hidden states.
A transfer audit showed that using it unchanged after the SPORA replacements at
layers 6 and 12 reduced single-entity character precision from 0.314 to 0.282
and recall from 0.925 to 0.906 on the selected CLUENER development examples.

The locator was then retrained with the same CLUENER training source and
synthetic name queries, while extracting features after the structural
replacement. On the selected 20 single-name news rows, exact predictions rose
from 0/20 to 10/20 and character precision from 0.281 to 0.701; character
recall was 0.887. On 10 multiple-name rows, exact predictions rose from 0/10
to 5/10, precision was 0.956, and recall was 0.896. The new locator retained
24/24 exact synthetic single-name queries and rejected all six synthetic
no-entity queries in this development run.

The training and evaluation are still small and synthetic/news-domain based.
CLUENER is not a memory-chat benchmark; public text may overlap pretraining,
and the annotations do not establish source trust, polarity, time, or write
authorization. The candidate is saved with the structural feature variant
digest but is not connected to the runtime. Full provenance and rows are in
`runs/deepseek-spora-cluener-locator-retrained-20261111/result.json`.

The next composition test should use this locator with the value-span probe and
explicit source and polarity gates. It must be evaluated on separately
authored or licensed memory-dialogue examples before any persistent write.
