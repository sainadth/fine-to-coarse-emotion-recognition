# Stage-II recipe sweep (M4, seed 42)

Selection by dev macro-F1; test shown for the record only. Source: kaggle_output/sweep/20260930-181659.
`speaker_prefix` and `long_focal_prefix` did not run (Kaggle mounted a stale source upload).

| Variant | Dev macro-F1 | Test macro-F1 | Test acc. | Test weighted-F1 |
|---|---|---|---|---|
| head_lr3e4 | 0.4917 | 0.4593 | 0.6303 | 0.6233 |
| base | 0.4858 | 0.4863 | 0.6398 | 0.6360 |
| focal1 | 0.4853 | 0.4654 | 0.6234 | 0.6205 |
| backbone_lr2e5 | 0.4776 | 0.4580 | 0.6337 | 0.6278 |
| speaker | 0.4769 | 0.4655 | 0.6414 | 0.6336 |
| stage1_first | 0.4769 | 0.4839 | 0.6391 | 0.6363 |
| ute4 | 0.4695 | 0.4706 | 0.6349 | 0.6286 |
| long_focal | 0.4689 | 0.4591 | 0.6268 | 0.6168 |
| focal2 | 0.4681 | 0.4591 | 0.6195 | 0.6167 |
| long | 0.4647 | 0.4634 | 0.6130 | 0.6151 |
| smooth | 0.4646 | 0.4779 | 0.6414 | 0.6343 |
| long_focal_speaker | 0.4623 | 0.4573 | 0.6184 | 0.6168 |
| weighted_ce | 0.4512 | 0.4388 | 0.5674 | 0.5891 |
| long_weighted | 0.4438 | 0.4547 | 0.5939 | 0.6040 |
| focal2_weighted | 0.4404 | 0.4258 | 0.5525 | 0.5742 |
