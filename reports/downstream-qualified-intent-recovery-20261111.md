# Downstream-qualified intent recovery

The fresh wording audit showed that two legitimate questions were rejected only
by the read-intent threshold even though their entity scores and single-name
spans were strong. The runtime now has a conservative recovery path: a request
below the calibrated intent threshold may continue only when its intent score
is at least `2.0`, the single-entity gate passes, and the span is contiguous.
Name binding and known-memory checks still run afterward. Unknown names and
multi-entity requests therefore do not bypass the later gates.

The same 18-positive/11-negative unified persisted-runtime audit was rerun:

| Metric | Before recovery | With recovery |
| --- | ---: | ---: |
| Positive routes correct | 11/18 | 13/18 |
| Answers containing expected value word | 11/18 | 13/18 |
| Negative requests rejected | 11/11 | 11/11 |

This is a controlled development improvement, not an independent accuracy
claim. The floor was selected from the current narrow gate behavior and must be
recalibrated on broader held-out data before treating it as a production
threshold. The test still uses one entity, two finite beverage facts and
authored questions; writes are supplied by the probe.
