# Paper figures

These metadata-stripped PDFs visually match the assets used by the active anonymous paper:

- `figure4_intphys_scene_crossfit.pdf` shows the corrected sealed IntPhys2 five-fold scene cross-fit readouts and paired global scene-bootstrap differences.
- `figure2_controlled_geometry_compact.pdf` shows relative displacement magnitude, mean resultant length above shuffled correspondence, and cross-fitted physical projection into the appearance subspace.
- `figure3_lewm_construct_controls.pdf` shows the TwoRoom and PushT action and appearance controls.
- `appendix_exploratory_trajectories.pdf` shows straightness, effective rank, and endpoint, all-dose, and local-tangent physical projections for the post-sealed exploratory V-JEPA2-G dose grid.

`scripts/make_paper_figures.py` rebuilds all four figures from the bundled tables. The builder reproduces Figures 1 and 3 pixel-for-pixel in the tested Python 3.12 environment. Figures 2 and 4 reproduce the same data, axes, intervals, and dimensions; their archived copies retain the paper's final adjusted `encoder depth` label placement. Metadata stripping does not change any rasterized page.
