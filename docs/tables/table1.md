| Model | GoEmotions | UTE | Seeds | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|---|---|
| M1 | No | No | 1 | 0.6533 | 0.4519 | 0.6374 |
| M2 | Yes | No | 1 | 0.6467 | 0.4671 | 0.6362 |
| M3 | No | Yes | 1 | 0.6402 | 0.4638 | 0.6337 |
| M4 | Yes | Yes | 1 | 0.6398 | 0.4863 | 0.6360 |

| Emotion | M1 F1 | M2 F1 | M3 F1 | M4 F1 | Support |
|---|---|---|---|---|---|
| anger | 0.469 | 0.460 | 0.472 | 0.501 | 345 |
| disgust | 0.198 | 0.324 | 0.243 | 0.288 | 68 |
| fear | 0.174 | 0.156 | 0.191 | 0.306 | 50 |
| joy | 0.600 | 0.607 | 0.600 | 0.595 | 402 |
| neutral | 0.801 | 0.791 | 0.782 | 0.779 | 1256 |
| sadness | 0.358 | 0.379 | 0.381 | 0.383 | 208 |
| surprise | 0.563 | 0.553 | 0.576 | 0.551 | 281 |

| Comparison | Effect | Macro-F1 delta |
|---|---|---|
| M1 → M2 | fine-grained transfer | +1.53 |
| M1 → M3 | sequential context | +1.19 |
| M3 → M4 | transfer on top of context | +2.25 |
