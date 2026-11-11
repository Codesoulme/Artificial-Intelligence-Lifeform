# Research notes and verification path

Recent arXiv work that informed this iteration:

- **Route to Remember: Adaptive Memory Selection for LLM Agents**
  (`arXiv:2611.05636`): weight memory and text memory are complementary; a
  verify-then-regenerate router lets each handle the queries it serves best.
- **Can LLM Agents Organize Their Own Memory?** (`arXiv:2611.11929`): typed
  linked records, evidence links, and separate temporal/relational/recent
  views reduce the cost of revising and retrieving memory.
- **MIRIX-2.0** (`arXiv:2611.12068`): typed memory agents, provenance
  hypergraphs, explicit retrieval budgets and consolidation outside the write
  path.
- **Decision Context** (`arXiv:2611.08644`): behavioral memories work better
  as explicit directives activated at decision points.
- **What Is This Memory Useful For?** (`arXiv:2611.03930`): memory quality
  should be measured by task-conditioned paired utility, not a task-blind
  importance score.

The current implementation adopts the first idea in a constrained form. The
output adapter generates once, checks whether its first generated token agrees
with the adapter's highest memory-supported token, and retries with a stronger
memory gain on conflict. The retry is recorded in the response metadata.
Unit tests cover one-shot behavior, gain scaling and batch validation.

The new wording audit did not change from `10/12` positive routes and value
words with or without the retry. This is useful negative evidence: output
verification is not the present bottleneck. The next work should target typed
memory predicates, provenance-backed domain routing and paired utility
evaluation, while preserving the current rejection gates.
