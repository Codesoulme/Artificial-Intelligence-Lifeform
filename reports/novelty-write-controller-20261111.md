# Versioned novelty write controller

The write-control architecture now has an explicit `NoveltyWriteControl`
variant with an 11th input containing the auditable observation novelty score.
The existing `WriteControl` remains unchanged, so historical checkpoints keep
their original 10-input behavior. State serialization records the controller
kind and restores the correct class.

`NoveltyWriteControl.from_legacy` creates a migration baseline by copying all
legacy weights and zeroing the novelty column. Tests confirm bit-for-bit equal
outputs when the novelty input is zero, round-trip restoration, finite input
validation, and a real observation using the extra feature.

The new controller has not been trained or connected to the user's individual.
The migration baseline is intentionally behavior-preserving; the next step is
to train and evaluate a replacement on labeled new/correction/repeat/noise
trajectories before adopting it in any checkpoint.
