# Leave-One-Scene-Out Stability

| model       | effect                               | layer_name | full_estimate | loo_min  | loo_max  |
| ----------- | ------------------------------------ | ---------- | ------------- | -------- | -------- |
| V-JEPA2-G   | hue_090_tdc_best_layer               | block_30   | 0.783978      | 0.781137 | 0.797331 |
| V-JEPA2-G   | brightness_0p45_tdc_best_layer       | block_30   | 0.954174      | 0.953452 | 0.961289 |
| V-JEPA2-G   | selective_recolor_180_tdc_best_layer | block_30   | 0.712209      | 0.708422 | 0.719711 |
| V-JEPA2-G   | pao_k4_block_00                      | block_00   | 0.401341      | 0.390728 | 0.408426 |
| V-JEPA2-G   | pao_k4_final                         | final      | 0.059580      | 0.056887 | 0.060405 |
| VideoMAE2-G | hue_090_tdc_best_layer               | block_30   | 0.747444      | 0.744129 | 0.755278 |
| VideoMAE2-G | brightness_0p45_tdc_best_layer       | final      | 0.964616      | 0.964158 | 0.965943 |
| VideoMAE2-G | selective_recolor_180_tdc_best_layer | final      | 0.782934      | 0.780010 | 0.794094 |
| VideoMAE2-G | pao_k4_block_00                      | block_00   | 0.347758      | 0.336302 | 0.353852 |
| VideoMAE2-G | pao_k4_final                         | final      | 0.076547      | 0.071786 | 0.077611 |
