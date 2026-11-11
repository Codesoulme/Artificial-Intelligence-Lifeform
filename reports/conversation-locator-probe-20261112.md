# Controlled conversational name probe

An additional 20 distinct annotated names were selected deterministically from
CLUENER validation texts. The source texts exclude the previous two selected
sets and exact training-text matches. This does **not** establish that the
names themselves are absent from training, base pretraining, or earlier
experiments. Each name was used in one artificial question and in two further
questions about a different, similar name formed by a prefix or suffix. Each
case independently enrolled the reference name as a fingerprint.

| Measure | Default | Retrained | Both required |
| --- | ---: | ---: | ---: |
| Known-name fingerprint match | 10/20 | 15/20 | 9/20 |
| False match on suffix names | 5/20 | 1/20 | 1/20 |
| False match on prefix names | 0/20 | 0/20 | 0/20 |
| Known cases with one predicted span | 19/20 | 19/20 | — |
| Suffix cases with exact full subject span | 5/20 | 14/20 | — |
| Prefix cases with exact full subject span | 6/20 | 12/20 | — |

For known cases, the number of exact full-name spans equals the fingerprint
match count. Most missed known cases were not caused by failure to produce a
single span; the predicted boundaries or characters differed. One truncated
suffix name matched the known-name fingerprint even with both locators.
Combining the locators reduces recall and is not a sufficient safeguard.

This is a 60-question synthetic probe using only one repeated question
template; counts are not population rates. Similar names are defined as
different people by construction. Manual per-case enrollment is an oracle,
and no address projection, actual memory read, generated response, conversation
history, real user data, or production route was tested. The reduced model's
parity with the complete model was checked on ten other questions only.
No candidate was deployed. Full results and hashed source provenance are in
`runs/deepseek-spora-conversation-locator-audit-v2-20261112.json`.

The next experiment should target full-name boundary prediction and explicitly
measure both missed known people and wrong-person reads on separately held-out,
varied dialogue, including natural ambiguity and multiple people. Raising or
lowering a span threshold without measuring both errors would be insufficient.
