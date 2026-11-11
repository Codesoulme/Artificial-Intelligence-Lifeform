# Delayed fact ordering and conservative abstention

Six short synthetic trajectories used the exact individual recovered from the
verified integrated checkpoint. A direct observation of an earlier effective
time arrived after a newer observation; other-person facts were interleaved.
The independent test oracle assumed each direct observation was true. This is
an experimental assumption: in practice the late claim can be uncertain.

Before the change, the write controller deferred every late observation and
preserved the current value in 6/6 cases. The deferred value remained readable
at its known observation-time address in 6/6. However the effective-previous
selector omitted the deferred fact in all six cases: it abstained in two and
returned an older, incorrect candidate in four. Earlier selector tests used
the controller's accepted actions as the oracle, so they did not reveal this
omission.

The selector now conservatively abstains when a deferred direct or reported
observation could affect the requested effective-time view. It does not treat
the deferred claim as true or resolve the conflict. Repeating the same six
cases produced six abstentions, while preserving correct current values and
direct late-history reads; same-process save/reload reproduced the observations
in 6/6. Effective-latest selection also abstains in these cases, a deliberate
availability cost. Observation-order selection and deferred inference sources
are unchanged. Neural unknown reads can still hide events, so this is not a
complete safety guarantee.

On the real temporary chat route, four selected delayed-fact cases all gave
the expected current answer and abstained on the past question before calling
generation. This is an integration check using known names and manual numeric
writes, not natural-language fact acquisition or an identity-safety test.
The changed selector is used by the existing historical route; current reads do
not go through it. There is no automatic workflow to resolve deferred facts yet.

Full local evidence is in `runs/deepseek-spora-delayed-fact-audit-20261112.json`
(before), `runs/deepseek-spora-delayed-fact-audit-v2-20261112.json` (after),
and `runs/deepseek-spora-delayed-runtime-guard-audit-20261112.json`. Reload was
in the same process, not a crash test. Longer histories, delayed duplicates,
conflicting sources and multi-person ambiguity need separate evaluation.
