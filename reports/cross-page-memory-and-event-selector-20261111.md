# Cross-page memory and chronological selector audit

The decoder integration previously supplied a history page slot of `1` for
every historical event. That was correct only for the first page. The fix now
uses `1 + event_time // time_count`, while the neural address still uses the
page-local time `event_time % time_count`. It validates that the event is
recorded and that the page exists. Current memory continues to use slot `0`.

The actual DeepSeek candidate was loaded on the available GTX 1060 and tested
with a constructed two-page state. The same entity was written at event 0
with value 1 and event 64 with value 6; 63 unrelated events filled the first
page. The native memory returned both expected values, and the integrated
decoder produced the corresponding value for both explicit historical reads.
The two rows took about 146 seconds end to end, including model loading and
generation, with a peak allocation of about 3.72 GB. This is a structural
cross-page check, not an autonomous time-understanding score.

The finite-symbol selector then scanned a separate 68-event, two-page state
containing interleaved addresses, a repeated value, corrections, a deferred
inference source, and a write carrying an earlier `valid_at`. It selected
first, immediately previous, and latest accepted belief writes for 15/15
defined cases and returned the expected value for 12/12 cases. A portable
reload produced no selection mismatches.

The selector deliberately does not rank by `valid_at`: observation order and
effective time are different concepts. It abstains for incomplete provenance,
uncertain numeric provenance, an exceeded scan budget, or insufficient
candidates. `unknown_event` is skipped, so an apparently successful result
does not prove that no hidden event exists. The oracle scores the write
controller's actions, which are not independent evidence that the facts were
true. Natural language, multiple independently trained individuals, abrupt
crash recovery, and open-world behavior remain untested.

The follow-up selector adds `effective_first`, `effective_latest`, and
`effective_previous`. It decodes the exact numeric `valid_at` field and, for
duplicate effective times, keeps the latest observation at that time. In the
same 68-event experiment, 15/15 effective-time selections matched the audit
oracle, including an intentionally supplied correction with `valid_at=0` that
the controller deferred. This result is still finite-symbol development
evidence; it does not establish that a natural-language date was understood or
that every hidden event was found.
