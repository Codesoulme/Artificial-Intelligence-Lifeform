# Memory-domain guard audit

The promoted v4 gate plus downstream recovery initially admitted three varied
non-memory questions because their entity and intent scores were high. The
runtime now requires a finite-domain memory predicate before routing into the
current beverage memory path. The guard accepts the trained beverage cues
`饮品`, `饮料`, `喝`, `偏好` and `饮用`; all later address and known-memory
checks remain active. This is a temporary domain guard while broader predicate
training is unavailable, not a general language solution.

Regression results:

| Audit | Positive result | Negative result |
| --- | ---: | ---: |
| 16 varied non-memory requests | not applicable | 16/16 rejected |
| 12 newly authored beverage questions | 10/12 routes and value words | 6/6 rejected |

The guard removed false accepts for color, sport, work and other ordinary
requests without reducing the new beverage holdout score. It also makes the
current prototype's domain boundary explicit. The next architectural step is
to replace this finite cue list with a separately trained, held-out predicate
router covering multiple memory domains and aliases.
