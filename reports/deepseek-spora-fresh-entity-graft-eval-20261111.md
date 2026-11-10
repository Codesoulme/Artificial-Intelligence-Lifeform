# Fresh-entity comparison of the layer-18 graft

Four new names, previously unused name-to-address assignments, new old/new beverage combinations and new question wording were used to compare fixed candidates. Each of the four systems received the same program-supplied address and time slot, and the integrated SPORA memory reader remained enabled in all systems.

| System | Expected value present, conflicting value absent | Exact short answer |
|---|---:|---:|
| Original DeepSeek layers with integrated memory | 6/8 | 4/8 |
| SPORA layers 6 and 12 | 6/8 | 6/8 |
| SPORA layers 6 and 12, 10% layer-18 blend | 6/8 | 6/8 |
| SPORA layers 6 and 12, 100% layer-18 blend | 3/8 | 3/8 |

Both paired variants failed the same two questions under the exact criterion. The 10% blend preserves this small sample's answer count but provides no observed improvement over the pair. Full replacement still degrades results. These are eight short questions in one finite beverage domain, not evidence of general language ability, autonomous routing, or open chat. The memory was programmatically written and addressed; the first comparison is an integrated base, not an unmodified standalone model. Full generated answers, candidate digests, and runtime are in `runs/deepseek-spora-fresh-entity-graft-eval-20261111.json`.

## Coverage correction

While preparing this probe, `_rows` in the earlier answer-distillation script was found to append outside its template loop. Earlier results recorded the actual evaluated counts correctly, but only the last variant for each intent was used, despite two variants being listed in the code. The loop is now fixed and a regression test checks both variants. This change alters the data produced by rerunning that script: old checkpoints and reports still describe the original, narrower experiment and should not be treated as if both variants were tested. Future runs using the corrected helper are new experiments, not exact reproductions of those old runs.
