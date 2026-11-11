# Runtime observation persistence

The unified local entry point now supports guarded fact writes through the same
integrated checkpoint used for answers. `--observe KEY VALUE` calls
`observe_sourced`, preserving source and quality checks, then atomically saves
the updated individual, native reader state, structural replacements and
memory output adapter back to the local v4 checkpoint. A failed write or save
does not replace the previous checkpoint because it uses the existing atomic
checkpoint path.

The interactive question path remains read-only unless an explicit observation
command is supplied. Inference and simulation sources continue to be subject
to the internal nonfactual-source guard. This closes the persistence path for
the local prototype; natural-language extraction of facts is still separate
and has not been enabled automatically.
