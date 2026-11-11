# Runtime candidate persistence smoke

The real DeepSeek/SPORA runtime loaded its protected integrated checkpoint with
the working path redirected to a temporary directory. A complete direct
candidate changed one finite value from water (index 6) to coffee (index 1).
The write controller returned `correct` with confidence 0.999992, the temporary
checkpoint was saved, and a fresh base model reloaded it with cursor 3, matching
the in-memory state. The actual user working checkpoint was not touched.

This validates the path from `ObservationCandidate` through the candidate guard,
existing SPORA write controller, atomic integrated checkpoint save, and fresh
reload. It does not validate natural-language parsing, open-vocabulary mapping,
source truth, or general correction accuracy; the candidate was constructed by
the test. The raw smoke output is recorded in the script run and the test is
repeatable with a temporary checkpoint.
