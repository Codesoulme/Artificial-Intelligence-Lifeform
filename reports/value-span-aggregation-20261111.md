# Value span aggregation comparison

The open value span prototype was compared with a learned token attention
pooler and a mean-plus-maximum pooler. Both variants used the same replaced
DeepSeek representation, data generation, boundary labels, and calibration
policy. No persistent individual was changed.

The learned attention pooler collapsed to nearly identical acceptance scores
for every sentence and therefore selected no held positives after calibration.
The mean-plus-maximum variant fitted its training set (87/99 positive spans,
zero negative acceptances) and calibration (12/20 positive spans, zero negative
acceptances), but selected 0/52 held positives at the calibrated threshold.
This is worse than the earlier mean span probe, which selected 50/52 held
positive spans, although that earlier probe had 24/130 held negative
acceptances.

The result indicates that adding pooling complexity without broader, better
annotated data does not improve the observation boundary. Keep the earlier
mean span probe as the current research candidate and leave all candidates
disconnected from persistent writes. The next useful improvement is data and
explicit relation labels, rather than another unvalidated pooling variant.
