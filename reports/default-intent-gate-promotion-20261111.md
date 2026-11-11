# Default intent gate promotion

The unified runtime now loads the v4 read-intent gate by default. v2 remains
available in its original checkpoint for regression comparison. The promotion
is based on two development comparisons with the same persisted checkpoint and
entity/temporal/output components:

- Earlier 29-case wording: v4 routed 14/18 positive requests and rejected 11/11 negatives.
- Newly authored 18-case wording: v4 routed 10/12 positive requests and rejected 6/6 negatives.

The v4 gate was adapted after inspecting earlier failures, so these numbers are
engineering evidence rather than an unbiased benchmark. The runtime retains a
downstream-qualified intent recovery floor and all later entity, address and
known-memory checks. A future holdout spanning additional domains, names and
request types can demote v4 if its rejection or generalization degrades.
