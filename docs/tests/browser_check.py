"""Browser acceptance for the static project page; owns and closes its browser."""
from pathlib import Path
from urllib.parse import urlsplit
import argparse
import json
import tempfile

from playwright.sync_api import sync_playwright, expect

TITLE = 'When Does Latent Prediction Error Support Physical Surprise Claims in Video World Models?'
STUDIES = ['readout-details', 'geometry-details', 'color-details', 'physion-details', 'support-details']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base_url')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    base = args.base_url.rstrip('/') + '/'
    if args.output:
        output = args.output
        output.mkdir(parents=True, exist_ok=True)
    else:
        scratch = Path.home() / 'tmp'
        scratch.mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix='jepa-pages-browser-', dir=scratch))
    findings, errors, failed, external = [], [], [], []
    host = urlsplit(base).netloc
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=['--no-sandbox'])
        try:
            for name, width, height in [('desktop', 1440, 1100), ('phone', 390, 844), ('small-phone', 320, 568), ('tablet', 820, 1180)]:
                context = browser.new_context(viewport={'width': width, 'height': height}, permissions=['clipboard-read', 'clipboard-write'])
                page = context.new_page()
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.on('response', lambda response: failed.append({'url': response.url, 'status': response.status}) if response.status >= 400 else None)
                page.on('request', lambda request: external.append(request.url) if urlsplit(request.url).netloc != host else None)
                response = page.goto(base, wait_until='networkidle', timeout=45000)
                assert response.status == 200
                expect(page.locator('h1')).to_have_text(TITLE)
                assert page.title() == TITLE
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), name
                expect(page.get_by_role('link', name='Paper', exact=True)).to_have_attribute('href', 'https://arxiv.org/abs/2611.01815')
                expect(page.get_by_role('link', name='Code & results')).to_have_attribute('href', 'https://github.com/3-ml/jepa-surprise')
                expect(page.locator('.authors a').filter(has_text='Vasil Vasilev')).to_have_attribute('href', 'https://vasi.boo/')
                assert page.locator('meta[name="citation_author"]').evaluate_all('(nodes) => nodes.map(n => n.content)') == ['Anton Hristov', 'Aleksandar Slavov', 'Vasil Vasilev']
                # Load even off-screen lazy images before testing their dimensions/screenshot.
                assert page.locator('img').evaluate_all('''async images => {
                    await Promise.all(images.map(image => {image.loading = 'eager'; return image.decode();}));
                    return images.every(image => image.naturalWidth > 0 && image.alt.length > 0 && Number(image.getAttribute('width')) === image.naturalWidth && Number(image.getAttribute('height')) === image.naturalHeight);
                }''')
                expect(page.locator('h2')).to_have_text(['What we evaluate', 'Physical-label classification', 'Appearance and physical changes across layers', 'Action and appearance sensitivity', 'Physical validity with and without support', 'Code and experiment records', 'Citation'])
                expect(page.locator('.study-more')).to_have_count(len(STUDIES))
                for identifier in STUDIES:
                    detail = page.locator('#' + identifier)
                    assert not detail.evaluate('(node) => node.open')
                    detail.locator('summary').focus()
                    page.keyboard.press('Enter')
                    expect(detail).to_have_attribute('open', '')
                assert page.locator('#physion-details tbody tr').count() == 3
                expect(page.locator('#physion-details tbody tr').first).to_have_text('Persistent minus original-0.00312 [-0.00385, -0.00236]-0.00270 [-0.00337, -0.00203]')
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), (name, 'expanded')
                for region in page.locator('.figure-scroll').all():
                    assert region.evaluate('(node) => node.getBoundingClientRect().right <= innerWidth + 1')
                    if width < 720:
                        assert region.evaluate('(node) => node.scrollWidth > node.clientWidth')
                        region.focus()
                        page.keyboard.press('ArrowRight')
                        page.wait_for_function('(node) => node.scrollLeft > 0', arg=region.element_handle())
                        region.evaluate('(node) => node.scrollLeft = 0')
                page.screenshot(path=str(output / (name + '-expanded.png')), full_page=True)
                for identifier in STUDIES:
                    detail = page.locator('#' + identifier)
                    detail.locator('summary').focus()
                    page.keyboard.press('Enter')
                    assert not detail.evaluate('(node) => node.open')
                assert page.locator('a[href^="#"]').evaluate_all('(nodes) => nodes.every(n => !!document.getElementById(n.hash.slice(1)))')
                expect(page.locator('#stop-validity')).to_have_text('Lawful')
                expect(page.locator('#fall-validity')).to_have_text('Violation')
                gap = page.get_by_role('button', name='Platform with a gap', exact=True)
                gap.focus()
                page.keyboard.press('Enter')
                expect(gap).to_have_attribute('aria-pressed', 'true')
                expect(page.locator('#stop-validity')).to_have_text('Violation')
                expect(page.locator('#fall-validity')).to_have_text('Lawful')
                expect(page.locator('#stop-frames')).to_have_attribute('src', base + 'assets/gap-stop.webp')
                expect(page.locator('#fall-frames')).to_have_attribute('src', base + 'assets/gap-fall.webp')
                assert page.locator('#stop-frames').evaluate('(image) => image.decode().then(() => image.naturalWidth > 0)')
                solid = page.get_by_role('button', name='Solid platform', exact=True)
                solid.focus()
                page.keyboard.press('Enter')
                expect(page.locator('#stop-validity')).to_have_text('Lawful')
                expect(page.locator('#fall-validity')).to_have_text('Violation')
                page.locator('#abstract > summary').click()
                expect(page.locator('#abstract')).to_have_attribute('open', '')
                assert 'A fixed error threshold also incorrectly flags many possible events as impossible.' in page.locator('#abstract p').inner_text()
                page.locator('#abstract > summary').click()
                page.get_by_role('button', name='Copy BibTeX', exact=True).click()
                expect(page.get_by_role('status')).to_have_text('BibTeX copied.')
                copied = page.evaluate('navigator.clipboard.readText()')
                assert copied == page.locator('#bibtex').inner_text().strip() + '\n'
                assert 'arXiv:2611.01815' in copied
                # Keep a clean first viewport and a full-page view, without test feedback.
                page.evaluate("document.getElementById('copy-status').textContent = ''")
                page.evaluate('window.scrollTo({top: 0, behavior: "instant"})')
                page.screenshot(path=str(output / (name + '-first.png')))
                page.screenshot(path=str(output / (name + '-full.png')), full_page=True)
                findings.append({'viewport': name, 'overflow': False, 'figures_loaded': True, 'support_keyboard_toggle': True, 'abstract': True, 'clipboard_exact': True, 'expandable_details': len(STUDIES), 'contained_figure_scrolling': True})
                context.close()
            context = browser.new_context(java_script_enabled=False, viewport={'width': 390, 'height': 844})
            page = context.new_page()
            page.goto(base, wait_until='networkidle', timeout=45000)
            expect(page.get_by_role('button', name='Platform with a gap', exact=True)).not_to_be_visible()
            expect(page.get_by_role('button', name='Copy BibTeX', exact=True)).not_to_be_visible()
            page.locator('#abstract > summary').click()
            expect(page.locator('#abstract')).to_have_attribute('open', '')
            assert '2611.01815' in page.locator('#bibtex').inner_text()
            for identifier in STUDIES:
                detail = page.locator('#' + identifier)
                detail.locator('summary').focus()
                page.keyboard.press('Enter')
                expect(detail).to_have_attribute('open', '')
            expect(page.locator('#physion-details table')).to_be_visible()
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            context.close()
            context = browser.new_context(reduced_motion='reduce')
            page = context.new_page()
            page.goto(base, wait_until='networkidle', timeout=45000)
            assert page.evaluate('getComputedStyle(document.documentElement).scrollBehavior') == 'auto'
            context.close()
            context = browser.new_context()
            page = context.new_page()
            page.add_init_script("navigator.clipboard.writeText = async () => { throw new DOMException('Test permission denial', 'NotAllowedError'); }")
            page.goto(base, wait_until='networkidle', timeout=45000)
            page.get_by_role('button', name='Copy BibTeX', exact=True).click()
            expect(page.get_by_role('status')).to_contain_text('selected for manual copying')
            assert 'arXiv:2611.01815' in page.evaluate('window.getSelection().toString()')
            context.close()
            assert not errors, errors
            assert not failed, failed
            assert not external, external
            report = {'base_url': base, 'checks': findings, 'no_javascript': 'readable, expandable studies work, no dead controls', 'reduced_motion': True, 'clipboard_denied': 'manual selection', 'browser_errors': errors, 'failed_requests': failed, 'external_runtime_requests': external}
            (output / 'browser-check.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report, indent=2))
        finally:
            browser.close()


if __name__ == '__main__':
    main()
