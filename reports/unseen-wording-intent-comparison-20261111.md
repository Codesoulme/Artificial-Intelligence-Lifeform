# Unseen wording intent comparison

This audit used 18 newly authored questions that were not copied from the
earlier 29-case set: 12 single-entity memory questions across current, past
and later wording, plus six non-memory or unsupported requests. The same
persisted checkpoint, entity gate, temporal gate and output adapter were used
for both intent-gate candidates in one process.

| Metric | v2 | v4 candidate |
| --- | ---: | ---: |
| New positive routes | 8/12 | 10/12 |
| Answers containing expected value | 8/12 | 10/12 |
| Negative requests rejected | 6/6 | 6/6 |

The v4 candidate therefore improves on wording that was not copied from the
previous audit, while preserving rejection on this small negative set. It is
still limited to one beverage domain and one bound name. The v4 training
templates were created after earlier development failures, so this is evidence
of improvement rather than an unbiased benchmark. The default runtime remains
v2 until a broader domain and entity holdout is completed.
