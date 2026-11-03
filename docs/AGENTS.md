# Paper website

This is the independent `gh-pages` branch of `3-ml/jepa-surprise`. Read `README.md`. Keep every site file under `docs/`; Pages serves this folder. Do not merge site edits into scientific `main` or touch its immutable release tags/manifests.

Use primary paper metadata at https://arxiv.org/abs/2611.01815 and pinned scientific records for claims. Preserve the author order, workshop-specific venue attribution, equal contribution and study limitations. No experimental run is authorized by website maintenance. Paper images are scientific evidence; do not generate or modify their plotted data.

Follow the accepted main manuscript's plain wording: descriptive headings, concrete changed inputs and measured outputs, no teaser slogans or dense abstract phrasing. Keep longer explanations and additional published figures in native expandable sections so the main reading path stays concise.

Keep the site static and dependency-free at runtime, with relative asset paths, keyboard access, reduced-motion support and a working no-JavaScript reading path. Run `tests/browser_check.py` against a local preview and then the public URL; visually inspect desktop and phone screenshots. Store temporary evidence outside the published directory. When changing CSS, update its HTML query key to the first 12 SHA-256 characters; the browser check verifies it so returning visitors receive the matching styles.
