"""Download the auto sources into data/raw and record what was fetched.

The user agent names this project honestly. Sites that refuse it (ssa.gov,
cbo.gov and bls.gov run bot protection) are listed as manual sources instead;
do not work around a refusal by imitating a browser.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from .. import DATA
from .sources import SOURCES, Source

USER_AGENT = "us-social-security-model/0.1 (+https://github.com/markwroberts0/us-social-security)"
MANIFEST = DATA / "raw" / "MANIFEST.json"
POLITE_DELAY_S = 1.0


def local_path(src: Source) -> Path:
    base = DATA / ("manual" if src.access == "manual" else "raw")
    return base / src.path


def _load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def _save_manifest(manifest: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(dict(sorted(manifest.items())), indent=2) + "\n")


class NotData(Exception):
    """The server answered with a web page (an error, a key prompt) instead of the file."""


def _download(url: str, dest: Path) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = resp.read()
        ctype = resp.headers.get("Content-Type", "")
    if dest.suffix not in (".html", ".htm") and (
        "text/html" in ctype or body[:200].lstrip().lower().startswith((b"<!doctype html", b"<html"))
    ):
        raise NotData(f"got an HTML page ({ctype or 'no content type'}), not the file")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(body)
    tmp.replace(dest)
    return body


def fetch(keys: list[str] | None = None, refresh: bool = False) -> list[str]:
    """Fetch auto sources. Returns the keys that failed."""
    manifest = _load_manifest()
    failed: list[str] = []
    last_hit: dict[str, float] = {}
    for src in SOURCES:
        if src.access != "auto" or (keys and src.key not in keys):
            continue
        dest = local_path(src)
        if dest.exists() and not refresh:
            continue
        host = urlparse(src.url).netloc
        wait = POLITE_DELAY_S - (time.monotonic() - last_hit.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        try:
            body = _download(src.url, dest)
        except (urllib.error.URLError, TimeoutError, NotData) as exc:
            print(f"  FAILED {src.key}: {exc}")
            failed.append(src.key)
            continue
        finally:
            last_hit[host] = time.monotonic()
        manifest[src.key] = {
            "url": src.url,
            "path": str(dest.relative_to(DATA)),
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "retrieved": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }
        _save_manifest(manifest)
        print(f"  fetched {src.key} ({len(body):,} bytes)")
    return failed


def status() -> list[Source]:
    """Print which sources are present; return the missing manual ones."""
    missing_manual: list[Source] = []
    for src in SOURCES:
        present = local_path(src).exists()
        mark = "ok " if present else "-- "
        print(f"{mark}{src.access:6} {src.key:34} {src.path}")
        if not present and src.access == "manual":
            missing_manual.append(src)
    if missing_manual:
        print("\nManual downloads (these sites refuse scripted requests):")
        for src in missing_manual:
            print(f"  {src.title}\n    from {src.landing or src.url}\n    save as data/manual/{src.path}")
    return missing_manual
