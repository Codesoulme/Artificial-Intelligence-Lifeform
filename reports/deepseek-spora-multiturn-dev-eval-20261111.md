# Multi-turn development comparison and prefix control

## Setup

The previously recorded `conversation-development-v2.json` contains eight development scenarios of three turns each, reserved from training. We ran all 24 turns on the original DeepSeek backbone, the SPORA layer-6/12 pair, and the layer-6/12/18 triple. The memory reader was disabled: the purpose was to examine language behavior with conversation history, not persistent memory. Each model received its own previous generated replies, so later input histories differ. Decoding was greedy, limited to 48 new tokens per turn and 256 total positions. To permit longer histories, candidate position tables were zero-extended after position 127; those new positions were not trained. A closing `</think>` was manually added before every answer as in earlier diagnostics. This is an artificial prefix and is not the model's normal reasoning mode.

All 72 turns ran. Seven original, three paired and two triple answers reached the token limit; three original, one paired and zero triple prompts exceeded 128 input tokens. Truncation and differing histories limit direct comparison. The run took 586.7 seconds, with peak allocated CUDA memory 3,932,108,800 bytes. Complete prompts and replies are in `runs/deepseek-spora-multiturn-dev-eval-20261111.json`.

## Observations from reading the replies

* In the two-person lending scenario, the paired model correctly identified who lent the magazine but then repeated that answer when asked who held the pen. The triple model gave an incorrect first answer. The original got the pen holder wrong on turn two but answered the return recipient on turn three.
* For an egg-free breakfast, the original and triple explicitly included eggs. The paired model's second response suggested eggs after the user said there was no bread, violating the prior constraint.
* For the revised appointment, the original produced unrelated network advice and the paired model repeated self-introductions; the triple repeated the initial appointment but did not retain the subsequent change.
* In the unknown-count scenario the original correctly computed seven minus two on the third turn, while the pair and triple did not. In the water/coffee scenario the triple identified warm water initially but invented or contradicted the reason on the next turn.

These examples show meaningful failures under this diagnostic setting; they do not establish a full-chat score or isolate each failure to one layer. The development set is small and has already been inspected.

## Prefix control

Because the original backbone also performed unexpectedly poorly, a separate probe used two first-turn questions with the original model and compared the same forced-closed prefix against the normal chat template. With the forced prefix, the answers were unrelated refusals. With the normal template, both runs consumed the full 192-token budget without closing the reasoning section or producing a final answer; therefore this control cannot establish whether the normal template would answer correctly with more time. It does establish that the 48-token forced-prefix comparison is not a direct measurement of the model's normal reasoning-mode behavior. The control records final answers and termination state, not the generated reasoning, in `runs/deepseek-think-prefix-control-probe-20261111.json`.

Next comparisons should use a separately held evaluation set, record both reasoning-mode completion and latency, and compare the same bounded settings without presenting development results as evidence of persistent memory or autonomous recall.
