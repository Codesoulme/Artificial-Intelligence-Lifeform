# Research review for the current SPORA direction

The review used primary arXiv abstracts and project descriptions available on
2026-11-11. It informs design decisions; it is not a claim that the cited
systems have been reproduced.

## Findings

**Titans: Learning to Memorize at Test Time**

<https://arxiv.org/abs/2501.00663> describes a neural long-term memory that
learns from historical context while attention handles the current context.
The useful idea for SPORA is a write signal based on novelty or surprise,
combined with a bounded memory update. The memory should not receive every
token equally. SPORA already has explicit write control and page history; the
next improvement is to expose a measured novelty feature to that controller.

**Infini-attention**

<https://arxiv.org/abs/2404.07143> combines masked local attention with a
compressive long-term attention state in one block and keeps memory and
computation bounded. This supports the current separation between the base
decoder's local processing and SPORA's persistent sites. The practical lesson
is to distinguish current-page reads from compressed historical reads and to
bound the number of pages read per token.

**MEMORYLLM**

<https://arxiv.org/abs/2402.04624> uses a fixed-size self-updatable memory pool
inside the latent space and evaluates repeated updates. The relevant idea is a
fixed-capacity learned memory pool; the risk is silent overwriting and identity
drift. SPORA's explicit provenance, correction counters, source gates and
checkpoint revision checks are needed around any such update.

**Test-time training and state evolution**

Test-time training research treats an evolving state as separate from the
frozen or slowly updated base model. That separation fits SPORA's current
implementation: the DeepSeek weights and structural replacements are checked
by provenance, while the individual state is durable and revisioned. Direct
online gradient updates to the DeepSeek weights are not appropriate for the
current local runtime because they would mix language adaptation, identity and
crash recovery in one uncontrolled mutation.

## Decisions for SPORA

1. Add novelty/surprise as an explicit write feature and test it independently
   before connecting it to `observe_sourced`.
2. Keep current and historical memory reads bounded and separate. Do not merge
   all history into a growing prompt or an unbounded attention cache.
3. Keep the base model, structural replacement parameters, individual memory,
   provenance and parser confidence as separate persisted components.
4. Keep open-vocabulary text candidates outside the finite neural memory until
   a value representation and a reliable source/time decision exist.
5. Do not use test-time gradients on the DeepSeek base as a substitute for the
   durable individual memory.

The immediate implementation target is the novelty feature and its evaluation
against repeated, corrected and unrelated observations. It should be measured
for write precision, correction recall, and resistance to repeated noise before
any confidence-floor change.
