# Pre-read check and output-bias ablation

An opt-in experimental identity guard now runs after address projection but
before the runtime checks memory status, selects an event, reads a value, or
generates. It compares the extracted name's explicit fingerprint with the
projected key. The default command-line route does not pass this guard. Its
fingerprints in this test were manually attached to a temporary individual;
there is no automatic enrollment or production deployment.

In 16 authored questions, four known cases reached the expected key and three
gave the expected answer. All eight near-name unknown cases were rejected by
the new check before the instrumented memory-status or generation calls. The
two multiple-person and two no-person cases were rejected earlier by the
intent gate, so they do not validate the new check. Eight questions repeat the
previous development pilot. This does not demonstrate robust identity: a unit
test explicitly demonstrates that a name truncated to an enrolled name can pass;
aliases, same-name people and enrollment errors remain unresolved. A guard
rejection also does not establish a general security boundary for all future
execution paths.

The wrong-answer case was investigated with the first-token logits processor.
For the expected value, the target token initially ranked first, but the
learned memory-output bias reduced its score by approximately 3.41 and moved it
to third. On a controlled rerun, suppressing only this output bias changed that
case from the unrelated English response to the expected value. Across four
hand-authored cases, exact answers went from 3/4 with the bias to 4/4 without
it. The other integrated memory-reader hooks remained active. These results
identify a harmful intervention in this case, but do not show that disabling
the output adapter generally improves answers or that the answer is independent
of memory. No weights or default generation settings were changed.

Local detailed results: `runs/deepseek-spora-preread-guard-pilot-20261112.json`,
`runs/deepseek-spora-output-logits-probe-20261112.json`, and
`runs/deepseek-spora-output-bias-ablation-20261112.json`. Follow-up should test
the pre-read guard against genuinely varied natural questions and truncation,
and evaluate output conditioning with counterfactual values before retraining
or deciding whether to change the adapter.
