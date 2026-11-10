# Progressive layer-18 answer training

Starting from the answer-distilled layer-18 candidate, we trained its SPORA branch for 140 steps at a fixed blend strength of 0.05 with the layer-6/12 replacements and native memory reader active. The frozen original layer remains in the blend. Training used 18 queries for six entities; the same six held queries for two other entities were evaluated across blend strengths.

| SPORA contribution | Held answers correct |
|---:|---:|
| 0, 0.025, 0.05, 0.075, 0.1 | 5/6 each |
| 0.2 | 4/6 |
| 0.4 | 3/6 |
| 1.0 | 2/6 |

The prior candidate scored 4/6 at 0.1, 2/6 at 0.4 and 1/6 at 1.0 on this same held set. The newly trained candidate therefore tolerated greater contribution in this small comparison, but did not improve on the 5/6 result with no layer-18 contribution. This is not independent confirmation: repeated experiments and the strength sweep reuse the held questions. An unseen evaluation set is required before choosing a strength or claiming generalization. The fixed 0.05 training-strength pre-training result was also 5/6.

The run took 400.5 seconds and reported peak allocated CUDA memory of 4,400,505,344 bytes. Results and source/candidate hashes are in `runs/deepseek-spora-layer18-progressive-answer-20261111/result.json`; the candidate weights remain local. One earlier launch stopped before training because `use_cache` was passed twice; the corrected script finished successfully. The experiment does not test language-based memory routing, varied fact domains, free conversation, or complete conversion of the backbone.
