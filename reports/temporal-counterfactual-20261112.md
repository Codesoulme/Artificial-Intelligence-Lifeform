# Temporal memory counterfactual checks

Two known names were each tested with three pairs of manually written drink
values, first old then corrected. The runtime was asked once about the current
value and once about the past value for each pair. With either the original
output bias or that bias suppressed, all 6/6 current and 6/6 past questions
selected the expected key/time class and returned the expected exact word.
Direct native queries also ranked the expected values first; the separate
previous-event selector identified event zero in all six states. When only
one fact had been written, both past questions abstained. The direct history
probe supplies the expected event as an oracle; runtime selection is measured
separately.

In a second experiment, events about another person were inserted between
the target's writes; one scenario included three writes for the target. Across
six states, the independent previous-event selector identified the intended
event in all cases, including event one rather than event zero. Native current
and oracle-addressed past queries ranked the expected values first in 6/6
states. The runtime selected the expected key, time class and previous event
for all 6 current and 6 past questions. Exact past answers were 6/6 in both
output modes; current answers were 4/6 with the output bias and 5/6 with it
suppressed. Thus correct state retrieval and event routing still did not ensure
correct language output.

These are short, authored finite-symbol tests with two names and manual
numerical writes. They do not establish robust long-term temporal reasoning,
natural language learning, general identity safety, restart persistence or
handling of delayed effective times. The event selector decodes neural
provenance and may miss events. The opt-in experimental identity guard was not
used; only known names were queried. Detailed local results are
`runs/deepseek-spora-temporal-counterfactual-audit-20261112.json` and
`runs/deepseek-spora-temporal-interleaved-audit-20261112.json`.

Follow-up should test delayed effective times, incomplete provenance,
unknown and ambiguous subjects, longer histories and restart, and continue
investigating output failures without treating these small results as a
general claim.
