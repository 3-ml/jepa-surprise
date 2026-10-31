# Static-content diagnostic

This diagnostic scores every complete 16-frame segment in 10 deterministically selected Physion++ Mass videos with V-JEPA2-L, measures consecutive RGB change, and compares each segment with repeated first-, middle-, and last-frame controls.

`segment_surprise_pixel_summary.csv` is the complete 273-segment table. `static_baseline_candidates.csv` contains repeated-frame residuals and the selected near-static cases. `selected_videos.csv` records the sample. `summary.json` and `summary_key_tables.txt` provide aggregates. The two PNG files visualize residuals and the residual-motion relationship.

Raw residual and motion correlate at 0.596. Six videos contain a high-error near-static candidate below mean RGB change 0.01; all six have nonpositive original-minus-static residuals with mean -0.0350. This is a diagnostic, not a population false-positive estimate.
