# Counterfactual output and calibration pilot

For each condition, a temporary individual was initialized from a fresh state
with four manually written numerical-key facts. For a fixed name and question,
the target fact was varied among three drink values in separate states. Other
facts remained fixed. All 12 conditions (two names, two question templates,
three values) returned the intended top memory value. The runtime routed all
12, producing exact expected one-word answers in 9/12 with the original output
bias and 11/12 with that bias temporarily suppressed. Output changed with the
written value, but the bias-suppressed path retains the other memory-reader
hooks. This does not isolate the effect of every memory component.

An experimental adapter was then initialized from the original and calibrated
for 80 CPU steps using frozen decoder logits from 12 conditions involving the
other two enrolled names and different training templates. Only the output
adapter was trained; its new weights were saved as a separate research artifact
and not installed. On the 12 conditions already examined above, exact answers
were 9/12 for the parent, 10/12 for the candidate and 11/12 with bias
suppressed. Those cases had informed the investigation and are development,
not independent evaluation.

Five finite values omitted from this calibration were then checked across two
names and two templates (20 conditions). All 20 native top values matched
their writes; parent and candidate each produced 10/20 exact expected answers,
while suppressing the bias produced 18/20. These five values had appeared in
the parent adapter's earlier training. The new adapter did not improve this
check and should not be promoted. These small tests suggest the existing bias
can hurt, but do not establish that removing it is safe for other tasks,
questions, values or historical reads.

All examples use numerical-key writes, pre-existing names and a finite beverage
vocabulary. Exact one-word matching is narrow and can undercount equivalent
phrasing. No natural-language learning, real user individual, broad dialogue,
or persistent model changes were tested. Detailed local results and adapter
provenance are in:

- `runs/deepseek-spora-counterfactual-output-audit-20261112.json`
- `runs/deepseek-spora-output-recalibration-pilot-20261112/result.json`
- `runs/deepseek-spora-output-unseen-values-audit-20261112.json`

Next work should compare output paths with varied unseen questions, values and
history, and independently intervene on reader state and output bias to
separate genuine memory dependence from a decoder's prior associations.
