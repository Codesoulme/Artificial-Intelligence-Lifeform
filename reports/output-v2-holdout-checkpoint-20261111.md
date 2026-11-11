# Output head phrase holdout and isolated checkpoint

The eight-value output candidate was compared with the existing head on two
names not used for output-head training, three newly authored Chinese question
templates, and four finite drink values. The SPORA address was supplied by
the probe. The native read was correct in 24/24 cases. Under exact answer
matching, the existing head scored 11/24 and the candidate 24/24. Correct
longer answers would count as failures under this metric.

The candidate was saved as a separate local integrated checkpoint. A new
Python interpreter reloaded it, verified the individual-state digest, cursor,
and every adapter tensor against the saved candidate, and checked the base and
structural provenance. The original integrated checkpoint remained unchanged.
In a subsequent small routed probe using this candidate checkpoint, three
controlled current/history questions produced the expected words and one
non-memory request was rejected.

The candidate file is local, not committed to Git. Its SHA-256 and the source
hash are in the machine-readable checkpoint report. The default runtime still
points to the original checkpoint and existing working state; no user memories
were migrated or replaced. The data consists of finite, authored beverage
questions and does not demonstrate open-domain chat, unknown-value handling,
stable multi-turn learning or robustness to other wordings. Further testing
must include diverse independently authored questions and memory corrections.
