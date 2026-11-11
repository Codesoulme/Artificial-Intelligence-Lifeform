# Incremental memory with the v4 output adapter

The existing transactional TensorStore can now bind to a structural v4
checkpoint containing the memory output adapter. The bootstrap remains
immutable. Its complete file hash binds the store to the exact adapter and
structural weights; the in-process decoder signature also detects ordinary
adapter parameter and configuration changes before reads or writes.

An isolated real-model probe loaded the existing DeepSeek/SPORA v4 bootstrap,
wrote one controlled observation into a temporary store and read its state back
in the same process. The state digest matched, the cursor changed from 2 to 3,
the transaction wrote 205,200 logical tensor bytes, the output adapter was
present, and the bootstrap file hash remained unchanged. The temporary state
was discarded; no personal memory was changed.

Eight small-model durable tests passed. These include the v4 adapter roundtrip,
adapter-mutation detection, stale-writer rejection and process termination
before/after commit. The termination cases use a v3 small-model bootstrap,
so they are not proof of a real-model v4 crash recovery. The chat entry still
uses its full working checkpoint rather than this incremental store; switching
it requires an explicit migration and separate end-to-end testing. Disk usage,
concurrent loading and generation quality were not measured by this probe.
