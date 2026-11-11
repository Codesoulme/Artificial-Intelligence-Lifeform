# Observation candidate on replaced representations

An isolated factorized probe was trained on features from the experimental
checkpoint with layers 6 and 12 replaced. Training, threshold selection and
evaluation used distinct names and sentence templates. The threshold was
chosen on calibration examples to maximize correct positives with zero observed
negative writes. All examples are synthetic and use eight fixed beverage values;
there was no entity resolution, source attribution or memory write.

| Split | Probe | Correct positives | Negative false writes |
| --- | --- | ---: | ---: |
| Training | Previous | 72/72 | 48/192 |
| Training | Candidate | 72/72 | 0/192 |
| Calibration | Previous | 16/16 | 16/48 |
| Calibration | Candidate | 3/16 | 0/48 |
| Held templates | Previous | 32/32 | 32/128 |
| Held templates | Candidate | 8/32 | 0/128 |

The candidate substantially reduces false writes in these examples but loses
most true positives on unseen formulations. Most held positive rejections come
from the value head selecting its abstain class, not the calibrated fact-score
threshold. A zero count on a small constructed set is not a safety guarantee.
There is no unsupported-vocabulary evaluation in this run, and names and
wording were authored together by the developer, so the held set is not an
independent real-world benchmark. The candidate is not connected to the
interactive path.

The next experiment should address value extraction and factuality separately,
retain an explicit unknown-value outcome, and evaluate on independently sourced
dialogue plus disjoint adversarial examples. Establish subject and source
attribution before evaluating any end-to-end write.
