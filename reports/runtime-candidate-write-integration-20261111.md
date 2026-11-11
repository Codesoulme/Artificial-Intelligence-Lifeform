# Candidate write integration

The structured observation boundary is now connected to
`SporaChat.observe_candidate`. The entry validates the candidate against the
individual key/value ranges, current cursor, claim type, source, quality,
effective time and confidence. A rejected candidate returns a defer decision
with reasons and never calls the persistence method. A fully resolved direct
candidate delegates to the existing locked, atomic `observe_sourced` path.

The focused runtime and candidate tests now total 11 passing cases, including
an open-vocabulary value (`汽水`) that is explicitly deferred and a complete
finite fact that is delegated. No real user checkpoint was changed by these
tests; the parser artifacts remain research candidates until their provenance,
source and temporal gates are composed in the live model path.
