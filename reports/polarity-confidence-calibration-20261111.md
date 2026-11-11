# Polarity confidence calibration

The six-class claim gate produced useful ranking separation but low raw
confidence: held affirmative probabilities were approximately 0.34 to 0.40,
so none reached the generic 0.9 write floor. Temperature scaling fitted on the
calibration split produced a temperature of 0.880, but still yielded 0/20
calibration and 0/52 held positive write-confident examples, with zero
negative acceptances.

A separate binary factual-claim gate was also trained on the same replaced
features. At the unchanged 0.9 floor it accepted 0/52 held positives and
0/156 negatives. The result confirms that the problem is not simply the number
of claim classes. The current outputs are development scores, not calibrated
probabilities suitable for persistence.

Keep the generic write floor unchanged and keep both calibration artifacts out
of the write path. The next step requires a larger independently annotated
calibration set, reliability metrics, and a domain-specific acceptance policy
that is validated for false writes before any threshold change.
