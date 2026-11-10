# Native memory causal read and generation probes

## Causal defect and correction

When no explicit memory address was supplied, `NativeSPORAMemoryReader` previously derived one query from the last hidden token and reused it at every position. Appending a token could therefore change the memory result at earlier positions, violating autoregressive causality and potentially making cached and uncached execution diverge. Explicit-address calls did not have this defect. The native reader now derives an address per position and scores history pages independently per position. This intentionally changes automatic-address behavior; it does not train a language-to-memory router or establish retrieval accuracy. The explicit-address path remains equivalent to the mapped legacy reader in the existing regression test, including multiple pages.

A new regression reproduces the original causal failure and verifies that earlier outputs remain unchanged when later tokens are appended. The targeted suite passed 46 tests. On two short prompts with a synthetic current/history state and no supplied address, cached and uncached generation produced identical token sequences for the integrated original layers, the layer-6/12 pair and the 6/12/18 triple. See `runs/deepseek-spora-auto-address-causal-probe-20261111.json` for the full outputs and hashes. Two prompts cannot establish numerical parity or behavior across all inputs.

## Explicit memory and dialogue samples

Before the causal fix, an explicit-address probe compared cached and uncached generation with a single entity, one current fact and one earlier fact. All three layer configurations generated identical token sequences under both execution modes and returned the expected short value for both questions. This does not test automatic routing. The code change does not alter the explicitly addressed algorithm, as checked in the regression suite.

The same probe sampled two unrelated Chinese questions without a memory state, at up to 48 generated tokens. The paired candidate gave an inaccurate rain explanation that included geological activity. Several answers hit the token limit, and the triple candidate confused comforting a friend with comforting oneself. These examples expose possible language-quality degradation but are too few for a comparative score or open-chat acceptance. The explicit-address probe, raw text, resource usage and provenance are recorded in `runs/deepseek-spora-integrated-cache-dialogue-probe-20261111.json`.

Both probes use a manually closed reasoning prefix, a small synthetic memory and at most 128 positions. They leave broader multi-turn conversation, autonomous recall, and the larger structural conversion unresolved.
