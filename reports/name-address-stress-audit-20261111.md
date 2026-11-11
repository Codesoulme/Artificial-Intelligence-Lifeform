# Name-to-address near-name audit

The experimental replaced-feature binding was evaluated without writing memory
on 24 authored queries for eight known names and 16 for eight unbound names.
Unlike its training, which pools the full question, this audit combines the
experimental subject locator with the binding and pools the predicted span.
Expected names and keys were used exclusively for scoring.

At the binding's existing training-derived threshold, 22/24 known queries
mapped to the correct address, two were rejected and none was accepted at an
incorrect address. However, 12/16 unbound near-name queries were accepted at
an existing address. Examples include `小林林` and `林小林` mapped to `小林`'s
key, and `阿宁宁` mapped to `阿宁`'s. This shows that two agreeing memory sites
and a high similarity threshold do not by themselves establish identity.

The prior 52/52 result used a much narrower set with only two known people; it
does not contradict this failure or justify automatic address-based writes.
This candidate is not the default runtime router. That separate route needs its
own near-name audit before any general safety statement. The observed result
does not establish whether the cause is span truncation, feature projection,
or both. The value/address state and user checkpoint were not modified.

The sample is small and manually authored; the result measures this challenge
set, not the error rate in general. Do not promote this binding for writes.
Further work should measure exact predicted span boundaries and compare a
projection trained on name spans with independently chosen names and queries,
keeping development and evaluation partitions separate. Raw rows and artifact
hashes are in `runs/deepseek-spora-name-address-stress-audit-20261111.json`.
