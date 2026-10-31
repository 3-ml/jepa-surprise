# WMPAI appendix figures: generating code

Code for seven candidate appendix figures of the WMPAI paper. Every figure is computed from existing records. No model was run and no new experiment was done. At the original code snapshot, figures were not committed or integrated. The later submitted manuscript used selected figures; this release contains generating code and stored numeric records, not newly rendered figures.

Run the commands below from `scripts/wmpai2026/alex/appendix_figures/`. Public path adaptations are documented in `release/DEPENDENCIES.md` at repository root; the code is not byte-identical to the original source snapshot. The final256 score input defaults to the bundled record, and the omitted dose cache must be supplied explicitly.

## Inputs

| Figures | Input | Source |
| --- | --- | --- |
| 1, 2, 3 | `release/final256/original/scores/all-scores.json` (repository-relative default) (3072 clips, 256 scenes, sha256 `055700d438bf765d381c9a142bdddaf2624f519bc2850428f9109e38ca309bde`) | `results/27-19-38-wmpai-vasi-final256-complete/final256-completion-metadata.zip` on `origin/wmpai/vasi-experiments`, extracted as data only |
| 5, 7 | `layer_features.npz`, `samples.json`, `analysis_split.json` in `${JEPA_MULTIDOSE_CACHE}` (path set at the top of `dose_common.py`) | multi-dose V-JEPA2-G cache, read only, `allow_pickle=False` |
| 4 | three mean-MSE tables hard-coded in `fig4_pusht_waterfall.py` | PushT decomposition table in the paper |
| 8 | none | schematic |

## Figures and commands

Run from the figure-code directory specified above.

| Figure | Content | Command |
| --- | --- | --- |
| 1 | crossover plot: lines by trajectory and by physical validity, solid support vs gap | `python scripts/fig1_crossover.py` |
| 2 | native error per tubelet, stop minus fall, 95% scene bootstrap | `python -I scripts/fig2_tubelet.py` |
| 3 | score distributions with the frozen threshold t* | `python scripts/fig3_threshold.py` |
| 1, 3 | acceptance checks, `figs/kubric_figs_provenance.json` | `python scripts/provenance.py` |
| 4 | PushT waterfall of S_h - S_o | `python scripts/fig4_pusht_waterfall.py` |
| 5 | per-factor PCA trajectories of dose displacements (exploratory) | `python -I scripts/fig5_trajectories.py` |
| 7 | cosine similarity between mean displacement directions (exploratory, descriptive) | `python -I scripts/fig7_cosines.py` |
| 5, 7 | provenance, `figs/dose_figs_provenance.json` | `python -I scripts/dose_provenance.py` |
| 8 | score pipeline and crossed 2x2 design schematic | `python scripts/fig8_schematic.py` |

Figure 6 (patch-error heat maps) is not included because no token-level error maps were stored.

## Checks built into the scripts

- Figures 1 and 3: the twelve lawful and inconsistent means, the four margins (clean solid -0.006734, clean gap +0.006908, persistent solid -0.007839, persistent gap +0.007495), the false-alarm and miss rates at t* = 0.4858863055706024, and the per-world AUROC values are compared with the paper values; `provenance.py` reports pass or fail.
- Figure 4: the three-term residual of S_h - S_o is checked against the paper contrast values.
- Figures 5 and 7: straightness and effective rank at the final layer are checked against the paper (hue 0.692 and 9.83, brightness 0.907 and 5.43, recolor 0.553 and 13.72).

## Status

Figures 5 and 7 are exploratory and were computed after the sealed test. The PCA in figure 5 is a two-dimensional projection and shows only part of the variance, as printed on each axis. Figure 7 is a new descriptive summary, not one of the frozen paper metrics.

Dependencies: Python 3.10, numpy 2.1.2, matplotlib 3.10.7.
