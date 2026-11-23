"""Polite HTTP access to SEC EDGAR (User-Agent + rate limit + retries)."""
from __future__ import annotations

import time
from pathlib import Path

import requests

from src import config

_session: requests.Session | None = None


def session() -> requests.Session:
    global _session
    if _session is None:
        if not config.SEC_USER_AGENT:
            raise RuntimeError(
                "Set SEC_USER_AGENT, e.g. export SEC_USER_AGENT='Your Name your@email.com' "
                "(SEC blocks anonymous automated requests)."
            )
        _session = requests.Session()
        _session.headers.update(
            {"User-Agent": config.SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
        )
    return _session


def get(url: str, retries: int = 5, stream: bool = False) -> requests.Response:
    for attempt in range(retries):
        time.sleep(config.SEC_SLEEP)
        r = session().get(url, timeout=120, stream=stream)
        if r.status_code == 200:
            return r
        if r.status_code in (403, 429, 500, 502, 503):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
    raise RuntimeError(f"Failed after {retries} retries: {url} (last status {r.status_code})")


def download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    """Stream a (large) file to disk; skip if it already exists."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {dest.name}")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with get(url, stream=True) as r, open(tmp, "wb") as f:
        for block in r.iter_content(chunk):
            f.write(block)
    tmp.rename(dest)
    print(f"  downloaded: {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest
