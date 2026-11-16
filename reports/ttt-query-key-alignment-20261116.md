# Query-Key Alignment Diagnostic

> **Deprecated artifact.** This diagnostic serialized fixed `2048 x 2048`
> matrices. The current DeepSeek down projection consumes `8960`-wide inputs,
> so the matrix artifact must not be loaded into the current reader. The
> compatible follow-ups are the sample-basis probes in
> `reports/ttt-query-basis-read-20261116.md` and
> `reports/ttt-query-basis-shared-20261116.md`.

The fast-weight reader writes next-state targets using keys captured at value
positions in fact sentences. Query generation uses a different decoder state at
the end of the question. In the interference probe, those spaces were nearly
unrelated: layer 6 key cosine between different people was as high as 0.933,
and the query-to-key diagnostic found near-zero direct query/key cosine and
poor nearest-person agreement.

A frozen-base ridge map was fit from the final question state to the write key
using four authored training people. It reached mean train cosine 0.992 at
layer 6 and 0.994 at layer 12. On four held people it reached 0.935 and 0.853,
respectively. This is a useful feasibility signal: a learned query bridge may
let the fast-weight lookup address a person-specific state. It is not yet an
implementation of reading. The fit uses oracle facts, person associations and
fixed questions; the map was evaluated on a tiny synthetic split and does not
reject unknown people.

The diagnostic does not alter the base checkpoint, adapter, or saved fast
weights. Full fitted weights, source hashes and limitations are in
`runs/deepseek-spora-ttt-query-key-alignment-20261116.json`. The next experiment
should insert the bridge only into the fast-weight delta path, compare ordinary
and protected writes, and measure exact answers, unknown rejection, correction
retention, reload parity and memory/latency cost.
