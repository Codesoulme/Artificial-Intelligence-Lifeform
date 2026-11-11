# Explicit resolution of a delayed observation

An opt-in `HistoricalResolutions` module records bounded explicit confirm/reject
decisions as exact integer tensors inside an individual checkpoint. They are
structured metadata, not learned weights or an automatic truth judgment. A
caller supplies the key, event, effective time and value; the module checks
that neural provenance still identifies a deferred direct/report observation
and that the native historical value agrees. The selector rechecks this
evidence each time; missing or changed evidence keeps the conservative
abstention. A confirmed observation may enter effective-time selection, and
an explicitly rejected one is excluded. Same-name identity, authorization,
confidence calibration and automatic confirmation are not solved.

On six previously used delayed-fact development trajectories, explicit
confirmation selected the expected previous event in 6/6; explicit rejection
selected the expected alternative, or abstained for insufficient history, in
6/6. Current and original historical top reads were unchanged. Ordinary
individual save and same-process reload reproduced all 12 results. These are
human/oracle decisions on known finite facts, not autonomous memory learning.

One full-model temporary-individual experiment confirmed a previously deferred
fact: before confirmation the historical question abstained; afterward the
correct previous event and exact answer were produced, the current memory
tensors did not change, and the individual portion of a temporary integrated
checkpoint restored the resolution and selection. This does not demonstrate
full decoder reload in a fresh process, concurrent commits, crash recovery,
natural-language adjudication, or reliable generalization. The new module
mutates in place and is not exposed through the standard user entry point;
callers must use a private copy and guarded transaction before any real use.

An earlier full run crashed in `c10.dll` while copying replacement tensors for
saving. The same copy on isolated CPU blocks succeeded once and isolated GPU
blocks succeeded three times; a full retry hit a separate CUDA out-of-memory
error during load. A research loader variant skips the two original layers
that the verified integrated checkpoint subsequently replaces, and checks no
meta parameters remain before inference. One run using that variant completed
the temporary integrated save. This does not prove the prior access violation
fixed or establish general loader equivalence; default loading is unchanged.

Detailed local results are
`runs/deepseek-spora-delayed-resolution-pilot-v2-20261112.json` and
`runs/deepseek-spora-resolution-runtime-pilot-20261112.json`.
