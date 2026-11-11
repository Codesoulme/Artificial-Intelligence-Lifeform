# Chinese memory-routing wording audit and gate experiments

The new 29-case authored probe contains 18 positive questions about one bound
person and 11 negative controls. The formulations were authored after the
original gate training; they share vocabulary and domain with earlier
experiments and are not a representative independent corpus. One negative
case closely resembles prior development material. The probe uses two
programmed facts, native provenance event selection and an eight-token answer
cap. It does not test general conversation or autonomous memory writing.

| Gate | Earlier development positives | New positive routes | New exact answers | New negatives rejected |
| --- | ---: | ---: | ---: | ---: |
| v2 | 19/20 | 11/18 | 7/18 | 11/11 |
| v3, fresh training | 10/20 | not run | not run | not run |
| v4, initialized from v2 | 20/20 | 10/18 | 6/18 | 11/11 |

The extra v3/v4 training templates do not use the exact positive audit
questions, but were designed after inspecting the audit. The v4 result is
therefore a development comparison, not a new independent score. v2 remains
the existing candidate: v3 regressed on the earlier set, while v4 improved that
set but lost two previously accepted fresh cases and gained one. The earlier
set was itself repeatedly inspected and must not be treated as an unbiased
estimate.

For v2, five of seven rejected positive requests were stopped by the read-intent
threshold and two by the single-entity gate. Four of eleven routed positives
failed strict exact answer matching; one generated the conflicting old drink,
one refused, and two produced unrelated or incomplete text. This reveals a
separate weakness in turning the selected neural memory into an answer. The
strict metric also rejects a correct value embedded in a longer sentence, so
individual answer text should be inspected before assigning a semantic score.

The 11 negative controls are too few to justify lowering the threshold. On the
existing calibration examples, a lower threshold would admit some negatives,
including write instructions. None of the gates is an access-control boundary.
No candidate weights are published; their hashes are recorded in the gate
reports. The audit script received additional temporal provenance validation
after the two runs; those runs used the existing local temporal checkpoint but
did not execute that added validation. Scores do not establish reliability of
other names, topics, time expressions or multi-turn interaction.

Next, evaluate memory conditioning separately from language routing using
paired histories that differ only in value and longer, uncapped responses.
Use separately held-out entities and wording, include an unconditioned base
comparison, and measure both correct-value use and contradiction. Preserve the
failed gate variants and do not tune a threshold on this audit set.
