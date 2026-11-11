# Natural dialogue parse audit

The composed subject, polarity, and value-span parser was run read-only on
seven drink-related user turns selected from the local DuLeMon test archive.
All seven were manually labeled as ineligible for an automatic persistent
write because they either lacked a named memory subject, were questions, were
mixed/negative, or did not state a specific stable preference.

The parser accepted 0/7 writes, matching the expected write count. It marked
one first-person utterance, `我爱喝橙汁`, as affirmative and extracted `橙汁`,
but correctly rejected it at the subject gate because no addressable person was
provided. The mixed coffee/milk-tea utterance was classified as negative, and
the question forms were rejected by the claim gate. One salutation containing
“小度” reached the subject-span heuristic, but its question polarity still
blocked persistence.

The sample is only seven selected turns and is not an independent benchmark;
labels describe write eligibility rather than truth. The parser did not write
memory or generate a response. Full rows, component hashes and structural
provenance are in
`runs/deepseek-spora-real-dialogue-parse-audit-20261111.json`.
