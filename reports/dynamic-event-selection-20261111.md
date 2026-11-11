# Dynamic event selection in the composed DeepSeek memory probe

The composed router now has an opt-in dynamic path. For historical questions,
the temporal intent selects `effective_previous` for a past request and
`effective_latest` for a later request. The selector scans neural provenance,
decodes `valid_at`, and supplies the resulting global event time to the
cross-page memory reader. If provenance is incomplete or uncertain, it
abstains. Current memory still uses the current slot.

The actual DeepSeek candidate was evaluated with the existing gates and the
variant temporal classifier. It used the same one-person, two-fact authored
development set as the fixed-route comparison:

| Measure | Dynamic event selection |
| --- | ---: |
| Positive cases | 11 |
| Positive routes correct | 8 |
| Positive exact answers | 7 |
| Negative cases | 6 |
| Negative requests rejected | 6 |

The score did not change from the fixed-position probe. That is expected for a
two-event setup, but the implementation now exercises neural event provenance
and page selection rather than mapping temporal classes directly to event 0 or
1. The recorded rows show selections for both effective-time classes and the
answers remain consistent with the selected facts.

This remains development evidence. The questions, entity, facts, gates and
candidate weights are reused; the output is generated with a short answer cap;
the Chinese text is stored through the local tokenizer and may display poorly
in a console with the wrong encoding. No open-world, multi-person,
independently held-out, autonomous writing, or general date interpretation has
been established. The selector intentionally abstains when its neural
provenance is uncertain, but skipping an unknown event cannot prove that the
history is complete.
