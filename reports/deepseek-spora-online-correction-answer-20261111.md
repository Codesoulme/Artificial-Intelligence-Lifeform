# Online correction with two SPORA decoder blocks

## Result

Two DeepSeek decoder layers (6 and 12) were replaced by answer-supervised SPORA `CausalBlock` candidates. The native SPORA memory reader remained active at layers 5 and 17, and the existing write controller handled three sequential events: an initial fact, a correction, then a second online correction. The test compared the unchanged integrated base with the replacement pair on current, past and intermediate-value questions after the writes.

| Entity | Base answers correct | SPORA pair answers correct |
|---|---:|---:|
| 阿远: 果汁→水→茶 | 3/7 | **7/7** |
| 小梅: 咖啡→豆浆→茶 | 0/7 | **7/7** |

The SPORA pair answered both old historical values, both first-correction values and both newest values correctly for each entity. Provenance reads retained `new`, `correct`, `correct` for the three write events. These two entity/value assignments were held out from answer-distillation training; the pair had separately reached 11/12 on the broader v4 held-template test and 12/12 on the earlier v3 split.

## What this establishes

This is direct runtime evidence that two SPORA-native transformer blocks can run with SPORA's persistent temporal memory and guarded online writes, then answer from the old and corrected event pages after another update. The result is substantially stronger than prior teacher-forced logit agreement: the generated answers match memory values through a live three-event sequence.

## Limits

The test covers two entities, one key per entity, three drink facts and seven short generated questions per entity. It supplies the entity address and temporal slot directly; language parsing/routing is not under test. It is not open-chat acceptance, and most of DeepSeek's 28-layer backbone remains unchanged. More names, unrelated facts, multi-turn user phrasing, seeds and value assignments remain necessary.

Per-answer text, write provenance and digests are recorded in `runs/deepseek-spora-online-correction-answer-20261111.json` and `runs/deepseek-spora-online-correction-answer-entity4-20261111.json`.
