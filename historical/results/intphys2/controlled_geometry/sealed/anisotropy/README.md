# Anisotropy Diagnostic

This is a post-freeze diagnostic and does not alter the primary metrics.

| model       | layer_name | population                     | mean_direction_norm | effective_rank_entropy | energy_top_1 | energy_top_2 | energy_top_4 | energy_top_8 |
| ----------- | ---------- | ------------------------------ | ------------------- | ---------------------- | ------------ | ------------ | ------------ | ------------ |
| V-JEPA2-G   | block_00   | unrelated_clean                | 0.033083            | 2.647231               | 0.666626     | 0.908656     | 0.973533     | 0.996638     |
| V-JEPA2-G   | block_00   | shuffled_hue_090               | 0.550031            | 3.340299               | 0.499857     | 0.781416     | 0.981003     | 0.997541     |
| V-JEPA2-G   | block_00   | shuffled_brightness_0p45       | 1.385023            | 1.598551               | 0.886097     | 0.967668     | 0.992058     | 0.998952     |
| V-JEPA2-G   | block_00   | shuffled_selective_recolor_180 | 0.440623            | 2.651951               | 0.623286     | 0.922941     | 0.975694     | 0.996601     |
| V-JEPA2-G   | block_10   | unrelated_clean                | 0.002312            | 3.762882               | 0.642342     | 0.787321     | 0.903984     | 0.969379     |
| V-JEPA2-G   | block_10   | shuffled_hue_090               | 0.052445            | 4.542736               | 0.565908     | 0.731622     | 0.876958     | 0.964375     |
| V-JEPA2-G   | block_10   | shuffled_brightness_0p45       | 0.144191            | 2.485604               | 0.793404     | 0.872308     | 0.940324     | 0.980561     |
| V-JEPA2-G   | block_10   | shuffled_selective_recolor_180 | 0.044425            | 4.058636               | 0.600009     | 0.772264     | 0.906345     | 0.965516     |
| V-JEPA2-G   | block_20   | unrelated_clean                | 0.002025            | 5.451431               | 0.533464     | 0.664703     | 0.826077     | 0.927408     |
| V-JEPA2-G   | block_20   | shuffled_hue_090               | 0.044113            | 6.595295               | 0.448232     | 0.584563     | 0.804230     | 0.920145     |
| V-JEPA2-G   | block_20   | shuffled_brightness_0p45       | 0.110834            | 4.133036               | 0.635129     | 0.745450     | 0.880850     | 0.949807     |
| V-JEPA2-G   | block_20   | shuffled_selective_recolor_180 | 0.036070            | 6.045189               | 0.496822     | 0.612772     | 0.814805     | 0.922895     |
| V-JEPA2-G   | block_30   | unrelated_clean                | 0.004638            | 3.590304               | 0.691694     | 0.795526     | 0.883216     | 0.953315     |
| V-JEPA2-G   | block_30   | shuffled_hue_090               | 0.043802            | 3.655179               | 0.695223     | 0.782496     | 0.876240     | 0.947972     |
| V-JEPA2-G   | block_30   | shuffled_brightness_0p45       | 0.197637            | 2.726171               | 0.774008     | 0.856755     | 0.919623     | 0.968016     |
| V-JEPA2-G   | block_30   | shuffled_selective_recolor_180 | 0.045024            | 3.823512               | 0.675067     | 0.764508     | 0.875730     | 0.953128     |
| V-JEPA2-G   | final      | unrelated_clean                | 0.011593            | 9.008464               | 0.371955     | 0.491484     | 0.658435     | 0.851673     |
| V-JEPA2-G   | final      | shuffled_hue_090               | 0.140823            | 9.721128               | 0.334337     | 0.455569     | 0.641056     | 0.840865     |
| V-JEPA2-G   | final      | shuffled_brightness_0p45       | 0.152216            | 9.339545               | 0.365247     | 0.479879     | 0.641367     | 0.829661     |
| V-JEPA2-G   | final      | shuffled_selective_recolor_180 | 0.118400            | 9.330885               | 0.352605     | 0.474810     | 0.657888     | 0.843326     |
| VideoMAE2-G | block_00   | unrelated_clean                | 0.002280            | 2.513934               | 0.645373     | 0.939143     | 0.976782     | 0.992800     |
| VideoMAE2-G | block_00   | shuffled_hue_090               | 0.107115            | 3.107169               | 0.472506     | 0.843634     | 0.982002     | 0.994553     |
| VideoMAE2-G | block_00   | shuffled_brightness_0p45       | 0.185195            | 1.666257               | 0.861516     | 0.975656     | 0.990650     | 0.997152     |
| VideoMAE2-G | block_00   | shuffled_selective_recolor_180 | 0.099900            | 2.445119               | 0.593115     | 0.958539     | 0.981007     | 0.993475     |
| VideoMAE2-G | block_10   | unrelated_clean                | 0.005070            | 7.346176               | 0.326237     | 0.558263     | 0.783405     | 0.922512     |
| VideoMAE2-G | block_10   | shuffled_hue_090               | 0.084764            | 7.920851               | 0.290155     | 0.498649     | 0.769764     | 0.918349     |
| VideoMAE2-G | block_10   | shuffled_brightness_0p45       | 0.128261            | 6.057605               | 0.438871     | 0.641561     | 0.822882     | 0.937536     |
| VideoMAE2-G | block_10   | shuffled_selective_recolor_180 | 0.069121            | 7.488533               | 0.316744     | 0.522878     | 0.792254     | 0.921985     |
| VideoMAE2-G | block_20   | unrelated_clean                | 0.005258            | 9.139711               | 0.293218     | 0.481092     | 0.694316     | 0.871955     |
| VideoMAE2-G | block_20   | shuffled_hue_090               | 0.077627            | 10.234718              | 0.247109     | 0.419607     | 0.631060     | 0.856054     |
| VideoMAE2-G | block_20   | shuffled_brightness_0p45       | 0.128350            | 8.509103               | 0.308693     | 0.514185     | 0.721717     | 0.889111     |
| VideoMAE2-G | block_20   | shuffled_selective_recolor_180 | 0.070816            | 10.147515              | 0.251148     | 0.423988     | 0.651897     | 0.848873     |
| VideoMAE2-G | block_30   | unrelated_clean                | 0.006738            | 9.004948               | 0.282366     | 0.485706     | 0.704300     | 0.880974     |
| VideoMAE2-G | block_30   | shuffled_hue_090               | 0.064301            | 9.967116               | 0.262189     | 0.431111     | 0.645674     | 0.863677     |
| VideoMAE2-G | block_30   | shuffled_brightness_0p45       | 0.113227            | 8.886434               | 0.284298     | 0.484321     | 0.708484     | 0.883479     |
| VideoMAE2-G | block_30   | shuffled_selective_recolor_180 | 0.059905            | 9.785348               | 0.257439     | 0.443823     | 0.658979     | 0.863967     |
| VideoMAE2-G | final      | unrelated_clean                | 0.006111            | 7.509547               | 0.295671     | 0.564587     | 0.772878     | 0.921158     |
| VideoMAE2-G | final      | shuffled_hue_090               | 0.044201            | 7.418958               | 0.328503     | 0.572274     | 0.772870     | 0.917660     |
| VideoMAE2-G | final      | shuffled_brightness_0p45       | 0.101998            | 7.949428               | 0.293565     | 0.526859     | 0.745053     | 0.922186     |
| VideoMAE2-G | final      | shuffled_selective_recolor_180 | 0.060224            | 7.751808               | 0.305899     | 0.534631     | 0.758454     | 0.918792     |
