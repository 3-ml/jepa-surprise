# Content review: paper site

Reviewed `index.html`, `site.js`, `style.css`, asset provenance and displayed scientific figures against arXiv:2611.01815v1, its saved primary PDF, and the pinned v1 TeX source. Read-only review; no site or public-repository source files changed.

## Finding

- **Citation omits the accepted workshop venue.** `index.html`'s BibTeX entry labels the work only as `journal={arXiv preprint arXiv:2611.01815}`. The primary arXiv record says “Accepted at the World Models in Physical AI Workshop, NeurIPS 2026,” consistent with the page's venue banner. Include the workshop attribution in the citation (for example, a `note` field) while retaining the arXiv identifier. This avoids losing the paper's actual venue when readers copy the citation.

## Checks

Title, author order, equal-contribution marker, affiliations, abstract, paper link and CC BY 4.0 attribution match the v1 record/source. The findings and figure captions reviewed match the paper; the development-set frame example is explicitly distinguished from final256 aggregate results. Figure provenance is documented and scoped so the paper license is not applied to repository code or external inputs. Static accessibility includes meaningful image alternatives, keyboard-focus styling, a skip link, reduced-motion handling, and a readable no-JavaScript path. No other actionable content issue found. Browser/runtime acceptance was not rerun as the parent is handling browser tests.
