import pytest

import render


class _FakeBrowser:
    def __init__(self):
        self.closed = False

    def new_page(self):
        return _FakePage()

    def close(self):
        self.closed = True


class _FakePage:
    def goto(self, *args, **kwargs):
        pass


class _FakePlaywright:
    def __init__(self, browser):
        self.chromium = _FakeChromium(browser)


class _FakeChromium:
    def __init__(self, browser):
        self._browser = browser

    def launch(self):
        return self._browser


class _FakeContext:
    def __init__(self, browser):
        self._browser = browser

    def __enter__(self):
        return _FakePlaywright(self._browser)

    def __exit__(self, *args):
        return False


def test_render_cards_closes_browser_on_error(monkeypatch, tmp_path):
    browser = _FakeBrowser()
    monkeypatch.setattr(render, "sync_playwright", lambda: _FakeContext(browser))

    def boom(page, card):
        raise RuntimeError("render failed")

    monkeypatch.setattr(render, "_png", boom)

    with pytest.raises(RuntimeError, match="render failed"):
        render.render_cards([{"text": "x"}], tmp_path)
    assert browser.closed
