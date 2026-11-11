# Name boundary training pilot

The previous 60-question probe was checked against tokenizer offsets. All 60
gold subject spans were expressible by the token masks; the retrained parent
locator's misses (5 known, 6 suffix, 8 prefix) therefore cannot be attributed
to an unrepresentable token boundary in that probe.

A new experimental linear token head was initialized from that parent and
trained for 200 steps on frozen reduced-prefix features. Training used 32
CLUENER training-split names excluded from the validation-name set, two
question patterns, eight no-person and eight two-person questions. Names and
questions are generated locally; the real individual and runtime were not
modified. This is supervised adaptation of a head, not training the base or
autonomous memory formation. Candidate weights and full metrics are in
`runs/deepseek-spora-boundary-train-pilot-20261112/`.

On the previously examined 20-name development questions, known-name exact
matches rose 15/20 to 18/20, while matching the enrolled known name for 40
constructed near-name questions fell 1/40 to 0/40. These examples were already
used to diagnose the problem, so the results are development evidence only.

The candidate was then compared with the parent on 20 different selected
validation names and different synthetic questions:

| Measure | Parent | Candidate |
| --- | ---: | ---: |
| Correct full known-name span | 13/20 | 14/20 |
| Near-name subject wrongly matches known name | 0/40 | 0/40 |
| Exact full near-name span | 26/40 | 28/40 |
| One span returned, no-person examples | 0/8 | 0/8 |
| One span returned, two-person examples | 1/8 | 0/8 |

The second set has new selected texts and names, but is from a validation split
used throughout development; it is not an untouched population benchmark.
Each group also uses a different synthetic wording. No-person and multi-person
single-span counts are proxies, not actual erroneous memory reads. Near-name
matching compares to the known name using an oracle; enrollment, address
projection, actual reads, output text, and real multi-turn dialogue were not
tested. Reduced-prefix parity with full model was measured only on ten earlier
examples. The candidate remains research-only; no default route was changed.
Detailed results are in `runs/deepseek-spora-boundary-fresh-audit-20261112.json`.

The modest improvement does not justify promotion. Next: construct a held-out
set of varied real dialogue contexts and test disambiguation, refusal, and
actual end-to-end reads; investigate the 6/20 remaining name misses before
choosing more training or a different span model.
