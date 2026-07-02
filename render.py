import base64
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

INDEX_HTML = Path(__file__).parent / "static" / "index.html"


def _png(page: Page, card: dict) -> bytes:
    data_url = page.evaluate("(c) => renderImage(c).toDataURL('image/png')", card)
    return base64.b64decode(data_url.split(",", 1)[1])


def render_cards(cards: list[dict], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.goto(INDEX_HTML.as_uri(), wait_until="domcontentloaded")
        for i, card in enumerate(cards, 1):
            path = out_dir / f"card-{i}.png"
            path.write_bytes(_png(page, card))
            paths.append(path)
        browser.close()
    return paths
