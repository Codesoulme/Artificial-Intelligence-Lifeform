# Default runtime near-name collision

The default runtime's own bootstrap and gate configuration were loaded with a
temporary working path. A fresh isolated individual was manually given one
known fact: the bound entity 小林 likes water. Two known-name questions were
accepted with the expected address. All eight questions for four distinct,
unbound near-names were also accepted and routed to 小林's address.

The follow-up locator probe reconstructed the selected character spans from
tokenizer offsets. In six of eight negative questions the entire near-name
was selected (小林林, 林小林, or 小林娜); the address projection nevertheless
resolved it to 小林's address. The two 小林子 questions selected only 小林,
showing a separate span-boundary failure. This evidence rules out the claim
that the collisions are solely due to name truncation, but does not identify
all causes of the projection failure.

This is a ten-question, manually authored challenge with one known memory,
not a population estimate. The configured default working individual was not
opened or modified; only an in-memory temporary state was used. Neither the
observations nor the result demonstrate that arbitrary names or facts are
correctly understood. The current default read route must be treated as
experimental and may attribute one person's memory to a different person.

Next, evaluate and improve both the locator and identity binding on separate
near-name partitions; use the full route, including unknown-name rejection and
ordinary known-name recall, to measure changes. Do not infer that a heuristic
passes simply because it blocks this single name family. Raw results:
`runs/deepseek-spora-default-near-name-audit-20261111.json` and
`runs/deepseek-spora-default-near-name-spans-20261111.json`.
