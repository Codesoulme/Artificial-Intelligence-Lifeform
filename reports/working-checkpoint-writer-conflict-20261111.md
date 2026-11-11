# Full working checkpoint writer conflicts

The interactive entry previously loaded a working checkpoint into memory and
then replaced the file on each explicit observation without checking whether
another process had updated it. Two running instances could therefore lose an
observation even though each individual write used atomic file replacement.

The entry now remembers the SHA-256 of the working file it loaded. Before
publishing an observation it takes a nonblocking OS lock on a stable sidecar,
checks the current file hash against its loaded version, and raises
`RevisionConflict` if the file changed or appeared since load. The check and
atomic save occur under the same cooperative lock. State is published in
memory only after save returns; after a committed save, failure to hash it
prevents later writes until the process reloads. Loading also checks for a
working-file change during model setup. The immutable bootstrap is unchanged.

The lock tests use separate processes to verify exclusion, release on normal
exit, and OS lock release after terminating a lock-holding process. Focused
runtime tests cover a stale working file and injected save failure. A separate
real-model experiment wrote one controlled correction to a temporary working
checkpoint and reloaded it in another process: cursor, file hash and state
digest matched, and answers for both current and previous values were exact.
The candidate source checkpoint remained unchanged.

This is a cooperative single-writer mechanism. A program that ignores the lock
can still race, and a process crash during an actual large checkpoint write or
a storage/power failure was not tested here. SHA-256 checks add full-file I/O;
this path still rewrites the full checkpoint rather than using TensorStore.
The two-process model test is one synthetic correction, not open-world chat or
automatic learning. A lock sidecar may remain as a small file after exit; it is
not itself evidence that a writer is running.
