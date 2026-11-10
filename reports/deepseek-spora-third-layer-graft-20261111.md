# Progressive graft at DeepSeek layer 18

## Question

Can another DeepSeek layer immediately after the native SPORA memory read at layer 17 be replaced without losing the paired layers 6/12 memory-answer behavior?

## Experiments

The layer-18 SPORA block was first distilled to match hidden activations after layers 6/12 and the active memory read at layer 17. Its held-answer accuracy fell from 5/6 to 0/6 after hard replacement. A second round directly trained the layer-18 block with answer cross-entropy; this raised accuracy only to 1/6.

We then kept the original layer-18 output and blended in the SPORA block output at strengths 0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4 and 1.0. Strength was selected by teacher-forced answer NLL on the six training entities, then generated answers were measured on the two held entities.

| SPORA output contribution | Training answer NLL | Held generated answers |
|---:|---:|---:|
| 0.00 | 0.16050 | 5/6 |
| 0.01 | 0.15930 | 5/6 |
| **0.025** | **0.15863** | **5/6** |
| 0.05 | 0.15879 | 5/6 |
| 0.10 | 0.16508 | 4/6 |
| 0.20 | 0.20000 | 4/6 |
| 0.40 | 0.36537 | 2/6 |
| 1.00 | 0.88821 | 1/6 |

## Interpretation

The local hidden fit reaches 0.995 cosine similarity, yet hard layer substitution destroys almost all held answers. A residual blend with a 2.5% SPORA contribution gives the best training NLL and preserves the existing 5/6 held behavior. This supports staged grafting with measured contribution strength. It does not yet show a held-answer improvement, and the third block's contribution remains small; more varied facts and users are needed before increasing it.

The blend path disables KV caching while it evaluates both the original and SPORA branches. The existing fully replaced layer-6/layer-12 pair continues to use its validated KV-cache path. Exact sweep outputs are in `runs/deepseek-spora-layer18-blend-20261111.json`; the layer-18 candidate tensor is local at `runs/deepseek-spora-postmemory-layer18-20261111/layer-18.pt`.

## Limits

Only two held entities, one finite beverage domain, and six generated answers. The entity address and temporal page are supplied by the test. This is an incremental architecture experiment, not open-chat validation or a fully converted backbone.
