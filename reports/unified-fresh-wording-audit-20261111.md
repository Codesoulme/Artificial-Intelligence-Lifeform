# Unified persisted runtime fresh wording audit

The complete fresh authored set was run through the unified runtime entry
point, including the persisted v4 structural checkpoint, native reader,
learned gates, effective-time selector and first-token output adapter. The
probe reset the state and wrote two controlled values for the bound name before
asking 18 positive and 11 negative questions.

| Metric | Unified persisted runtime |
| --- | ---: |
| Positive questions | 18 |
| Correct routes | 11/18 |
| Answers containing the expected value word | 11/18 |
| Negative requests rejected | 11/11 |

The earlier route-only fresh audit had 11/18 correct routes and 7/18 strict
exact answers. The new 11/18 word-containment score is not directly comparable
to strict exact matching, but it is a useful intervention result: the route
decisions did not improve, while the persisted output path supplied the
expected value word in four additional cases. This isolates the next work
area: broadening natural-language routing without weakening rejection of
non-memory requests.

The test remains narrow. Questions share one beverage domain, one bound entity
and two controlled facts; the writes are supplied by the test; word containment
is only a coarse answer metric; and the gates were developed on related
authored data. It does not establish open-domain conversation, autonomous
observation or general Chinese language ability.
