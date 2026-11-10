# Memory-answer distillation for paired SPORA layer replacements

## Result

Joint answer supervision recovered native-memory answer behavior after replacing DeepSeek layers 6 and 12 with SPORA `CausalBlock` modules. On two held entities, six held temporal question forms and two old-to-new fact assignments absent from training, the candidate answered all 12 questions with the exact stored drink word. The unmodified DeepSeek model answered 8/12 under the same memory state, prompts and generation budget.

| Evaluation | Base | SPORA pair before answer tuning | SPORA pair after tuning |
|---|---:|---:|---:|
| Held entities, familiar held templates and trained value-pair pattern | 6/6 | 6/6 | 6/6 |
| Held entities, six unseen templates and held-out directed old→new pairs | 8/12 | — | **12/12** |

The strict held pairs were 小岚: 可乐→茶 and 阿明: 酸奶→牛奶. Training used six different entities and six different directed old-to-new value pairs. All eight drink words occurred in training somewhere, so the held-out condition tests new compositions of values, entity bindings and temporal prompts rather than unseen output vocabulary.

The candidate is a pair of randomly initialized SPORA blocks distilled and then answer-supervised while the native SPORA memory reader remains active at layers 5 and 17. Only the two replacement blocks were trainable. The base language model, memory reader, write controller and persistent-memory format stayed frozen. The relevant candidate hashes are recorded in `runs/deepseek-spora-memory-answer-distill-v3-20261111/result.json`.

## Recovery note

The first answer-supervised run successfully wrote both candidate checkpoints but stopped while serializing its report because of a missing import. The saved pair was evaluated in a separate process and reached 6/6 on the familiar held templates; its training-step trace was lost. The subsequent expanded-template and novel-value runs completed normally and have full reports.

## Limits

This is a narrow positive result, not evidence of open-ended chat. It covers two held entities, 12 short questions and a finite set of eight drinks. The runtime is given the entity address and temporal slot; it does not infer them from raw language in this test. Exact-word scoring and an eight-token generation limit can reject correct longer paraphrases. A further replication should rotate held entities, keys, values, seeds and phrasing families, then test memory writing and correction in a multi-turn generated conversation.
