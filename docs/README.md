# Paper project page

Live site: https://3-ml.github.io/jepa-surprise/

Paper: https://arxiv.org/abs/2611.01815

## Layout and publishing

All website content, assets and tests are under `docs/` on the **`gh-pages` branch**. GitHub Pages publishes this branch's `/docs` folder. `.nojekyll` makes this a plain static site: no framework, build service, external fonts, analytics or CDN runtime dependencies.

The default scientific `main` branch and its versioned release manifests/tags are independent. Do not merge website commits into `main` or regenerate a scientific manifest for a page edit. Update the site on this branch and push normally; GitHub's built-in Pages deployment publishes it. No repository-root index or custom Actions workflow is required.

Preview from the repository root:

```sh
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

The deployment artifact is exactly this directory. Keep relative asset URLs compatible with a GitHub project-site subpath.

## Content

- `index.html`: paper metadata, on-demand official abstract, concise study summaries, expandable result explanations and figures, resource links and BibTeX.
- `style.css`, `site.js`: responsive layout, the support/trajectory illustration and citation copying.
- `assets/`: paper-derived figures, illustrative frame strips, social preview and favicon. See `assets/PROVENANCE.md`.
- `tests/browser_check.py`: browser acceptance checks. Requires Playwright with Chromium already installed. With the local preview running:

```sh
uv run --no-project --with playwright python docs/tests/browser_check.py http://127.0.0.1:8000/
```

Pass a different base URL to check the public deployment. Tests create and close their own browser, do not require accounts and do not modify the scientific release.

The expandable results use native `details` elements and work without JavaScript. Wide figures and the Physion results table scroll within focusable regions, keeping labels readable on phones without widening the page.

Match title, authors, abstract, arXiv link and citation against the announced paper when revising. Keep the broad readout/geometry/native-error scope. Do not present development-set illustration frames as the independently scored final-test population, compare native score scales across models, or turn post-hoc readouts into claims of untouched-test performance.
