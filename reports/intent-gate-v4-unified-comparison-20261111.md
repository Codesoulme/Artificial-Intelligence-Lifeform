# Intent gate v4 comparison in the unified runtime

The conservatively adapted v4 read-intent gate was loaded into the same
persisted runtime used by the default v2 audit. The state, structural
checkpoint, output adapter, entity gate, temporal gate and 29-question set were
kept unchanged.

| Metric | Default v2 | Candidate v4 |
| --- | ---: | ---: |
| Positive routes | 13/18 | 14/18 |
| Answers containing expected value | 13/18 | 14/18 |
| Negative requests rejected | 11/11 | 11/11 |

The candidate v4 gate was initialized from v2 and trained with additional read
and non-read wording after inspecting earlier failures. Its result is therefore
useful engineering evidence, but not an unbiased held-out score. It remains a
candidate rather than the default runtime gate until a new wording set is
authored independently of these failures. Both variants preserve the
downstream entity, known-memory and temporal checks.
