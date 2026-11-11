# Output adapter value coverage

The original adapter was trained only on values 1 and 6. A controlled probe
with one name, one question and eight possible values found the native SPORA
query correct for all eight, while the original output path returned the wrong
word for value 7 (expected 酸奶; returned 水). The two trained values were
correct, and five of six values not used in adapter training were correct.
This is one prompt, not a general accuracy estimate.

A separate v2 candidate was trained across all eight finite values; it does
not replace the original adapter or individual. On two names absent from this
adapter's training and one new question template, each of the eight controlled
values was queried with a caller-supplied address. The original adapter gave
9/16 exact matches, and v2 gave 16/16. The candidate has seen all evaluated
value classes and both candidates use a previously trained language reader;
the test does not isolate which component carries the lexical mapping. It does
not cover new domains, automatic addressing, free conversation, or reliable
recognition of unrecorded facts. All observations used temporary states.

The candidate weight file remains local; its digest and training output are in
the experiment result. Promotion requires independent phrase and entity tests,
testing of uncertain/unknown evidence, a complete route comparison, and a new
versioned integrated checkpoint. Existing individual state is preserved.

The checkpoint loader was also tightened: v4 must actually contain its output
adapter, legacy formats cannot silently contain newer components, and the
adapter's input value count must match the individual. Sixteen checkpoint tests
and eight incremental-store tests passed.
