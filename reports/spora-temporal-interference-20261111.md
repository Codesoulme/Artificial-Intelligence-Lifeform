# Temporal memory interference probe

## Question

Does correcting a fact later in a history page damage the old value stored at its earlier event time, and does that change at page boundaries?

## Method

The probe loads the protected pretrained SPORA individual and its trained write controller, creates a fresh controlled state, then writes one initial fact, inserts 1–127 additional accepted events, and corrects the original fact. Two sequences are used: events on distinct keys and repeated changes to the same key. The tested gaps are 1, 2, 4, 8, 16, 32, 63, 64 and 127. At page capacity the probe uses the existing explicit `grow` operation. It measures cosine between the retrieved history vector and the target value embedding before and after correction, raw vector drift, and nearest-value top-1 accuracy.

## Results

- 18/18 old-value reads selected the correct target embedding at top-1 after correction.
- 18/18 new-value reads selected the correct target embedding at top-1.
- Worst old-value cosine after correction: 0.9879.
- Worst raw vector drift: 16.1%, in the dense repeated-same-key sequence.
- The distinct-key sequence stayed at approximately 1.00 old-target cosine; its largest vector drift was about 5% near the page boundary.
- Corrections accepted by the trained controller were required for each case; the protected source checkpoint and loaded individual digest were checked before and after the run.

The exact per-case records are in `runs/spora-temporal-interference-v2-20261111.json`.

## Interpretation and limits

The previous identity-matrix unit fixture was a poor proxy for the learned temporal address geometry. The real-individual probe shows that RLS writes preserve the tested value ranking despite measurable drift. No change to the write rule is justified by these 18 cases alone.

This is not an end-to-end language evaluation. The probe uses one fixed individual, one seed, a narrow key/value vocabulary and authored write schedules. It does not test noisy statements, natural-language time routing, query refusal, sleep consolidation, or adversarially similar facts. High similarity in the native memory vector does not prove the integrated DeepSeek language model recalls the old fact correctly.
