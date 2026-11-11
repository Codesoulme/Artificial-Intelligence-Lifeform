# Composed subject and value parsing

The retrained subject locator and the open-vocabulary value span probe were
run together on the held synthetic split after the SPORA replacements. The
subject locator was trained on the replaced representation; the value probe
preserves the source text span instead of mapping it to a fixed beverage class.
No persistent memory was changed.

The value probe alone accepted 24/130 negative examples. Requiring a contiguous
single subject span reduced this to 2/130 while retaining 52/52 positive
examples. The two remaining false accepts were:

- `小梅从来不喝牛奶。`
- `小梅从来不喝酸奶。`

The composition correctly rejected the examples where one person was buying a
drink for someone else or reporting another person's preference. This is a
useful architectural result: subject routing and value extraction remove
different error classes and should remain separate gates.

The result is still a small synthetic test. The subject locator comes from a
CLUENER news-domain development source, while the value probe uses authored
templates. Neither component models source trust, polarity, time, correction,
or authorization. The two remaining false accepts show that the composed path
is not safe for automatic writes. Full rows and provenance are in
`runs/deepseek-spora-memory-parse-composition-20261111.json`.
