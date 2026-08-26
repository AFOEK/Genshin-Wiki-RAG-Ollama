from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from urllib.parse import urlencode
import time
from dataclasses import dataclass, field
from DrissionPage import Chromium, ChromiumOptions

log = logging.getLogger(__name__)

@dataclass
class BrowserResponse:
    url: str
    status_code: int
    text: str
    headers: dict[str, str]=field(default_factory=dict)
    @property
    def content(self) -> bytes:
        return self.text.encode("utf-8", errors="replace")

def build_url(url: str, params: dict | None = None) -> str:
    if not params:
        return url

    query = urlencode(params, doseq=True)
    return f"{url}{'&' if '?' in url else '?'}{query}"

def looks_like_cloudflare_challenge(title: str, html: str) -> bool:
    title=(title or "").casefold()
    html=(html or "").casefold()

    markers=(
        "just a moment",
        "performing security verification",
        "verify you are human",
        "checking your browser",
        "challenges.cloudflare.com",
        "/cdn-cgi/challenge-platform/",
        "cf-chl-",
    )

    return any(marker in title or marker in html for marker in markers)

class BrowserFallback:
    def __init__(self, *, browser_path: str | None = None, user_data_path: str = "data/browser/fallback", port: int = 9333):
        self.browser_path = browser_path
        self.user_data_path = str(Path(user_data_path).resolve())
        self.port = int(port)

        self._lock = threading.Lock()
        self._browser = None
        self._tab = None

    def _start(self):
        if self._browser is not None:
            return

        opts = ChromiumOptions()
        if self.browser_path:
            opts.set_browser_path(self.browser_path)

        opts.set_local_port(self.port)
        opts.set_user_data_path(self.user_data_path)
        self._browser = Chromium(opts)
        self._tab = self._browser.latest_tab

        log.info("[BROWSER] started at port: %d profile: %s", self.port, self.user_data_path)

    @property
    def tab(self):
        self._start()
        return self._tab

    def wait_for_manual_clearance(self, url: str, *, timeout_s: float=180.0, poll_s: float=2.0) -> bool:
        with self._lock:
            self._start()
            log.warning("[BROWSER] Cloudflare challenge detected; opening browser url=%s", url)
            self._tab.get(url)
            print("\nCloudflare verification requires attention.\nComplete the verification in the opened Chromium window.\nThe crawler will resume automatically after the challenge clears.\n", flush=True)
            deadline=time.monotonic()+timeout_s
            while time.monotonic()<deadline:
                title=self._tab.title or ""
                html=self._tab.html or ""
                if not looks_like_cloudflare_challenge(title,html):
                    log.info("[BROWSER] challenge cleared title=%r url=%s", title, self._tab.url,)
                    return True

                time.sleep(poll_s)

            log.error("[BROWSER] challenge clearance timed out after %.0fs url=%s", timeout_s, url,)
            return False

    def get_response(self, url: str) -> BrowserResponse:
        with self._lock:
            self._start()
            ok=self._tab.get(url)
            if ok is False:
                raise RuntimeError(f"Browser navigation failed: {url}")

            html=self._tab.html or ""
            final_url=str(self._tab.url or url)

            if looks_like_cloudflare_challenge(self._tab.title or "", html):
                raise RuntimeError(f"Cloudflare challenge returned again: {final_url}")

            return BrowserResponse(
                url=final_url,
                status_code=200,
                text=html,
                headers={
                    "Content-Type":"text/html; charset=utf-8",
                },
            )

    def get_json(self, url: str, *, params: dict | None = None):
        full_url=build_url(url, params)
        with self._lock:
            self._start()
            result=self._tab.get(full_url)
            status=getattr(result, "status", None)
            if status is not None and int(status) >= 400:
                raise RuntimeError(f"Browser request HTTP {status}: {full_url}")

            if looks_like_cloudflare_challenge(self._tab.title or "", self._tab.html or ""):
                raise RuntimeError(f"Cloudflare challenge returned again: {full_url}")

            try:
                return self._tab.json
            except Exception:
                # Some JSON endpoints are rendered as body text.
                body=self._tab.ele("tag:body")
                if not body:
                    raise

                return json.loads(body.text)

    def get_html(self, url: str) -> str:
        with self._lock:
            self._start()

            result=self._tab.get(url)
            status=getattr(result, "status", None)

            if status is not None and int(status) >= 400:
                raise RuntimeError(f"Browser request HTTP {status}: {url}")

            html=self._tab.html or ""

            if looks_like_cloudflare_challenge(self._tab.title or "", html):
                raise RuntimeError(f"Cloudflare challenge returned again: {url}")

            return html

    def close(self):
        if self._browser is not None:
            try:
                self._browser.quit()
            finally:
                self._browser=None
                self._tab=None
