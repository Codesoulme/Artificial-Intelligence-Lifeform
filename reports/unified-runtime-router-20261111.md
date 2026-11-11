# Unified routed runtime probe

`deepseek_spora_runtime_chat.py` is now the local route-and-answer entry point
for the persisted v4 checkpoint. It loads the DeepSeek base, the structural
replacement layers, the native SPORA reader and the memory output adapter. A
question is processed by the learned read-intent gate, single-entity gate,
temporal gate and name-to-address router. Past and later requests then call the
neural effective-time selector. Accepted requests pass native memory logits to
the persisted output adapter; ambiguous or unsupported requests abstain.

The controlled probe wrote two values for one bound name and tested current,
past, later and non-memory questions:

| Question type | Result |
| --- | --- |
| Current preference | routed and correct |
| Earlier preference | routed; dynamic event selected; correct |
| Later preference | routed; dynamic event selected; correct |
| Non-memory encouragement request | rejected by the read-intent gate |

The recorded rows show learned address indices `[6, 6]`, effective event
selection for historical requests, and the corresponding generated words.
This is an integration smoke test with one entity and finite drink values. The
probe supplies the writes and therefore does not test autonomous observation;
the gates were trained on narrow authored data; no open-world Chinese dialogue
claim follows. The entry point remains experimental and should abstain when
its evidence gates fail.
