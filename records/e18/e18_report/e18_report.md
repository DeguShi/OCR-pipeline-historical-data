Held-out WER (%) on 7, 10, 12; clean default 18.74. In brackets: share of the shift's damage recovered (default - objective) / (default - clean).

| shift | default | conf | lex | conflex | stat | oracle |
|---|---|---|---|---|---|---|
| bleed_through | 19.21 | 19.21 (+0%) | 19.21 (+0%) | 19.21 (+0%) | 20.47 (-267%) | 19.53 (-67%) |
| faded_ink | 20.16 | 19.69 (+33%) | 19.06 (+78%) | 19.06 (+78%) | 20.00 (+11%) | 19.37 (+56%) |
| jpeg_noise | 90.08 | 68.66 (+30%) | 70.08 (+28%) | 70.08 (+28%) | 90.08 (+0%) | 77.95 (+17%) |
| low_contrast_cast | 19.84 | 18.74 (+100%) | 19.53 (+29%) | 19.53 (+29%) | 21.57 (-157%) | 20.16 (-29%) |
| low_res_blur | 28.98 | 27.72 (+12%) | 23.31 (+55%) | 23.31 (+55%) | 25.35 (+35%) | 28.03 (+9%) |
| uneven_light | 21.57 | 19.21 (+83%) | 19.06 (+89%) | 19.06 (+89%) | 24.41 (-100%) | 19.06 (+89%) |

Spearman(objective, greedy CER) over candidates on the sample page (1 = perfect ranking):

| shift | n | conf | lex | conflex | stat |
|---|---|---|---|---|---|
| bleed_through | 47 | 0.69 | 0.95 | 0.94 | 0.76 |
| faded_ink | 47 | 0.92 | 0.94 | 0.95 | 0.62 |
| jpeg_noise | 60 | 0.91 | 0.97 | 0.98 | 0.56 |
| low_contrast_cast | 47 | 0.88 | 0.94 | 0.95 | 0.56 |
| low_res_blur | 47 | 0.97 | 0.95 | 0.97 | 0.60 |
| uneven_light | 47 | 0.89 | 0.96 | 0.96 | 0.59 |

Chosen chains:

- bleed_through: conf up2 seg[contrast+remove_rules] rec[]; lex up2 seg[contrast+remove_rules] rec[]; conflex up2 seg[contrast+remove_rules] rec[]; stat up2 seg[illumination+contrast+remove_rules] rec[]; oracle up2 seg[clahe+remove_rules] rec[]
- faded_ink: conf up2 seg[illumination+contrast+remove_rules] rec[clahe]; lex up2 seg[clahe+remove_rules] rec[contrast]; conflex up2 seg[clahe+remove_rules] rec[contrast]; stat up1 seg[contrast+remove_rules] rec[contrast]; oracle up2 seg[clahe+remove_rules] rec[]
- jpeg_noise: conf up2 seg[clahe+remove_rules] rec[denoise]; lex up2 seg[] rec[denoise]; conflex up2 seg[] rec[denoise]; stat up2 seg[contrast+remove_rules] rec[]; oracle up1 seg[contrast+remove_rules] rec[denoise]
- low_contrast_cast: conf up3 seg[contrast+remove_rules] rec[contrast]; lex up2 seg[clahe+remove_rules] rec[channel]; conflex up2 seg[clahe+remove_rules] rec[channel]; stat up1 seg[contrast+remove_rules] rec[gamma]; oracle up2 seg[clahe+remove_rules] rec[]
- low_res_blur: conf up2 seg[illumination+remove_rules] rec[contrast]; lex up2 seg[clahe+remove_rules] rec[contrast]; conflex up2 seg[clahe+remove_rules] rec[contrast]; stat up2 seg[clahe+remove_rules] rec[sharpen]; oracle up2 seg[contrast+remove_rules] rec[clahe]
- uneven_light: conf up2 seg[illumination+remove_rules] rec[gamma]; lex up2 seg[clahe+remove_rules] rec[gamma]; conflex up2 seg[clahe+remove_rules] rec[gamma]; stat up1 seg[contrast+remove_rules] rec[gamma]; oracle up2 seg[clahe+remove_rules] rec[gamma]
