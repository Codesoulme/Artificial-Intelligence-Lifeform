# Open vocabulary value span probe

Finite value classification silently maps an unseen value to the closest known
class. This experiment instead predicts whether a sentence contains a factual
observation and extracts the corresponding token span from the sentence. It
uses the DeepSeek representation after SPORA replacements at layers 6 and 12.
The experiment is isolated and performs no memory writes.

The first version extracted 50/52 held positive values, but accepted 29/130
negative sentences. After adding varied training forms for negation,
hypotheticals, changed preferences, and actions performed for other people,
the candidate still extracted 50/52 held positive values while false acceptance
fell to 24/130. Calibration had 13/20 positive values and zero negative
acceptances. The held results are the meaningful check; the calibration result
is only used to select the threshold.

The value span itself remains open vocabulary, so strings such as “汽水” and
“橙汁” are preserved instead of being forced into “水” or “果汁”. However,
fact detection is still not reliable enough for automatic persistence. The
training and held data are synthetic, and subject attribution, source trust,
temporal meaning, authorization, and correction are not modeled. The full
rows and artifact provenance are in
`runs/deepseek-spora-value-span-probe-v3-20261111/result.json`.

The next iteration should use independently authored or licensed dialogue with
explicit annotations for fact, subject, source, polarity, time and value span.
The span extractor can then become the value component of a guarded write path;
it must remain disconnected until the complete route reaches an independently
measured false-write target.
