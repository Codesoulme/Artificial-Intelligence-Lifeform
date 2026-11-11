# Joint identity and address classifier

The earlier binary identity gate could accept a sentence without proving that
the default route chose the corresponding address. This experiment trained a
single linear head over whole-utterance features to output either a specific
SPORA key or an unknown class. Eight names and their keys were supervised;
no existing state or model weights were modified. The candidate is not used by
the default runtime.

Training, calibration, and held wording/negative transformations were
distinct, but the eight enrolled names were necessarily shared. A threshold was
chosen solely on calibration examples to reject all observed unknowns and
avoid accepted wrong known keys. At that threshold:

| Split | Known correct | Known rejected | Unknown accepted |
| --- | ---: | ---: | ---: |
| Train | 16/16 | 0/16 | 0/64 |
| Calibration | 3/8 | 5/8 | 0/16 |
| Held | 12/16 | 4/16 | 5/48 |

No known sample was accepted at the wrong known address, but five unknown
near-names were assigned existing keys. This makes the candidate unsuitable
for automatic identity routing or persistent writes. The synthetic data and
small calibration set do not estimate real-world performance; no subject
enrollment, multi-person resolution, source validation or answer quality was
evaluated. Raw rows, checkpoint hash and provenance are in
`runs/deepseek-spora-joint-identity-probe-20261111/result.json`.

The first run exited without diagnostic output and produced no result. A later
run used smaller feature-extraction batches and completed; the cause of the
first exit is unknown. Future work should collect separately authored cases
and investigate identity representations and controlled enrollment rather than
choosing a threshold from the reported held results.
