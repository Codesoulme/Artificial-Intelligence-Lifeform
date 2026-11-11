# Candidate correction across a normal restart

This probe loads the isolated output-v2 integrated checkpoint in one process,
records the existing native value at a manually specified key, and writes a
controlled correction to value 7 into a temporary working checkpoint. A second
independent process loads the temporary working checkpoint and asks both the
current and the pre-correction questions through the actual route and decoder.

The controller reported `correct`; the cursor advanced from 2 to 3. The
second process restored the same cursor, individual-state digest and working
file digest. Its current answer exactly matched the new value and its previous
answer exactly matched the native value measured before the correction. The
source candidate checkpoint's SHA-256 remained unchanged. The temporary
working copy was removed after the experiment and no user working individual
was modified.

This is a single synthetic correction and two authored questions. The write
uses a manually specified finite key and value; it does not demonstrate that
the model extracts facts from ordinary conversation. Both processes exit
normally: this is not a crash, power-loss, concurrent-writer, or large-history
test. The default runtime is not switched to the candidate, and this result
does not establish unrestricted dialogue or memory reliability.
