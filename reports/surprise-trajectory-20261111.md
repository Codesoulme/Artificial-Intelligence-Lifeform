# Novelty trajectory validation

The first probe used a randomly initialized small SPORA model and could not
reliably read its own finite values. That result is retained as a test
limitation: novelty derived from an untrained readout is not meaningful.

A second smoke used the real DeepSeek/SPORA integrated checkpoint and a
temporary working file. Three sequential direct candidates produced:

- first correction: `correct`, novelty `1.0`;
- same value repeated: `repeat`, current write `false`, novelty `0.25` from
  time advancement only;
- second correction: `correct`, novelty `1.0`.

The temporary checkpoint reloaded successfully and the real working checkpoint
was untouched. This validates that the novelty signal follows actual durable
runtime state. It remains diagnostic and is not yet an input to a new write
controller.
