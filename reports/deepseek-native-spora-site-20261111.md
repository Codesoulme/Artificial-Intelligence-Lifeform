# DeepSeek native SPORA memory-site pilot

## Question

Can the open-base memory reader at DeepSeek decoder layers be replaced by SPORA's own `DeltaMemorySite.read` while preserving the already learned memory behavior and the base model's output on a small temporal-memory task?

## Change

`NativeSPORAMemoryReader` installs two native SPORA `DeltaMemorySite` modules at the existing integrated read sites (layers 5 and 17). Its fast matrices are the SPORA individual's current/history memory matrices. The existing trained reader's output projection and gate are mapped into the native sites; the native slow expert outputs are zero-initialized so this mapping starts as an identity-preserving conversion. The integrated checkpoint format advances to v2 and continues to load v1 checkpoints.

This replaces the memory read unit, not DeepSeek's Transformer blocks, attention, tokenizer, or base weights. Memory writes and temporal routing are still performed by the existing SPORA state/controller path and the separate experimental router.

The next integration slice adds `IntegratedSPORARuntime`, which owns the
decoder, controlled SPORA individual and evolving state. A sourced observation
is proposed through SPORA's existing controller and native `TemporalSite`
(a `DeltaMemorySite` subclass). If a checkpoint path is configured, the
candidate state is atomically saved before it becomes the runtime's live state;
a failed save leaves the prior state active. Runtime reload restores the native
reader, write controller, provenance and memory pages from one integrated
checkpoint.

## Evidence

| Measure | Legacy reader | Native SPORA site |
|---|---:|---:|
| Temporal route accuracy, 9 held prompts | 9/9 | 9/9 |
| Final answer accuracy | 8/9 | 8/9 |
| Held-prompt outputs that changed | — | 0/9 |
| End-to-end run time | 138.1 s | 98.6 s |
| Peak allocated GPU memory | 3,599,055,872 bytes | 3,600,349,696 bytes |

The run-time difference is not treated as a speedup: it is one alternating experiment per implementation, and GPU contention and thermal conditions were not controlled. Both runs use one name, one memory address, two drink values, 12 synthetic router examples and 9 authored holdout prompts. The one wrong answer is a base-model refusal to “截至目前，小林的饮品偏好是什么？”, even though the router chose the correct current-memory page.

Unit tests verify one-page and multi-page output equivalence after weight mapping, checkpoint save/load, v1 checkpoint compatibility, temporal-slot integration, write-failure rollback, new-fact/correction provenance, and reload of both events. The focused suite passed 24 tests after adding the runtime transaction test.

The correction test also surfaced a remaining memory limitation: two event-time addresses written into the same RLS history page can overlap, so a later correction can shift the earlier address's retrieved vector. Provenance retains the distinct `new` and `correct` labels, but this test does not establish interference-free old-value recall. Page-boundary/capacity and address-separation effects need a dedicated evaluation before claiming robust correction behavior.

The full 751-test suite exposed unrelated local-state failures: P114 disk-story tests hit their memory guard with about 2.46 GB of physical memory available, and a P28 test could not find its required isolated PyArrow 19.0.1 vendor. It also caught a stale read-mask benchmark unpacking the pre-slot context tuple; that benchmark was updated and its test now passes.

## Provenance and limits

- DeepSeek weight SHA-256: `58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945`
- Original SPORA individual SHA-256: `6269d7749148c95ed4653928f61642c8d93098defed649baab3f807c71494db9`
- Native integrated checkpoint SHA-256: `bf5f036bbadac54bb03314d66400bab6ce02caccf5559fe36b19d6f66454a4db`
- The 9-prompt result is development evidence only; it does not establish open chat, autonomous temporal interpretation, or unrestricted online learning.
- The native SPORA slow experts are mapped from a zero-output initialization; they have not yet been trained to improve language behavior.
- The native reader checkpoint is local under `runs/deepseek-temporal-intent-router-native-spora-site-20261110/`.
