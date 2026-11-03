'use strict';

const worlds = {
  solid: {
    stop: {label: 'Lawful', className: 'lawful', alt: 'With solid support, the sphere stops on the platform: a lawful continuation.'},
    fall: {label: 'Violation', className: 'violation', alt: 'With solid support, the sphere falls through the platform: a physical violation.'},
  },
  gap: {
    stop: {label: 'Violation', className: 'violation', alt: 'With a gap in the platform, the sphere stops unsupported: a physical violation.'},
    fall: {label: 'Lawful', className: 'lawful', alt: 'With a gap in the platform, the sphere falls through the opening: a lawful continuation.'},
  },
};

document.querySelector('.support-controls').hidden = false;
for (const button of document.querySelectorAll('[data-world]')) {
  button.addEventListener('click', () => {
    const world = button.dataset.world;
    for (const control of document.querySelectorAll('[data-world]')) {
      control.setAttribute('aria-pressed', String(control === button));
    }
    for (const trajectory of ['stop', 'fall']) {
      const state = worlds[world][trajectory];
      const label = document.getElementById(`${trajectory}-validity`);
      const frames = document.getElementById(`${trajectory}-frames`);
      label.textContent = state.label;
      label.className = `validity ${state.className}`;
      frames.src = new URL(`./assets/${world}-${trajectory}.webp`, document.baseURI).href;
      frames.alt = state.alt;
    }
  });
}

const copyButton = document.getElementById('copy-citation');
copyButton.hidden = false;
copyButton.addEventListener('click', async () => {
  const citation = document.getElementById('bibtex');
  const status = document.getElementById('copy-status');
  try {
    await navigator.clipboard.writeText(citation.textContent.trim() + '\n');
    status.textContent = 'BibTeX copied.';
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(citation);
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = 'Copy was blocked by your browser. The citation is selected for manual copying.';
  }
});
