# Explicit identity fingerprint prototype

The near-name audits showed that neural similarity alone can map a distinct
name to a known person's key. This prototype adds an optional, fixed-capacity
`IdentityFingerprints` module to a SPORA individual. Explicit enrollment binds
one normalized complete name to one key; a deterministic SHA-256 fingerprint
is stored in model buffers. Lookup of an unbound full name returns no key.
Reusing a key or binding the same name twice is rejected. A name and its
near-name can occupy different keys when both are explicitly enrolled.

This is an exact-match check implemented using tensors, not learned neural
memory or proof of a person's real identity. It does not solve homonyms,
aliases, identity enrollment, authorization, or incomplete span extraction:
if `小林子` is truncated to `小林` upstream, this module will match `小林`.
Fingerprints of low-entropy names can be guessed by enumeration; they should
not be treated as encryption or as secret personal identifiers. It is not
connected to the default runtime or the existing user individual.

Five focused tests cover near-names, conflicts, Unicode normalization, corrupt
state, ordinary checkpoint and transactional tensor-store round trips. A mock
integrated checkpoint also preserved the module. The existing integrated
checkpoint and durable suites ran 25 tests successfully. These checks do not
establish behavior on the real DeepSeek model, crash recovery during a write,
or automatic identity routing. Next evaluate a complete subject-span mechanism
and route agreement on a protected temporary individual before any promotion.
