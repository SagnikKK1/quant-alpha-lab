"""Download client for data.binance.vision (USDT-M futures bulk dumps).

Design rules:
- Ingestion is faithful-to-source: placeholder bars for delisted contracts are
  downloaded as-is and dealt with in curation (`delistings.py`), so the raw
  cache always mirrors the vendor.
- Every zip is verified against its sibling ``.CHECKSUM`` (SHA256) before use.
- A 404 is a normal outcome (contract not listed that month), returned as
  ``None`` — only non-404 HTTP errors raise.
- Downloads are idempotent: an existing raw file that passes checksum is not
  re-fetched.
"""

from __future__ import annotations

import hashlib
import logging
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import requests

from alphalab.config import BINANCE_VISION_BASE, RAW_DIR, UM_FUTURES_PREFIX

log = logging.getLogger(__name__)

_S3_LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def monthly_kline_url(symbol: str, interval: str, month: str) -> str:
    """month is 'YYYY-MM'."""
    return (
        f"{BINANCE_VISION_BASE}/{UM_FUTURES_PREFIX}/monthly/klines/"
        f"{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    )


def monthly_funding_url(symbol: str, month: str) -> str:
    """Funding rates are published monthly-only (verified §12.1)."""
    return (
        f"{BINANCE_VISION_BASE}/{UM_FUTURES_PREFIX}/monthly/fundingRate/"
        f"{symbol}/{symbol}-fundingRate-{month}.zip"
    )


def daily_metrics_url(symbol: str, day: str) -> str:
    """day is 'YYYY-MM-DD'. Metrics (OI, long/short ratios) are daily-only."""
    return (
        f"{BINANCE_VISION_BASE}/{UM_FUTURES_PREFIX}/daily/metrics/"
        f"{symbol}/{symbol}-metrics-{day}.zip"
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, session: requests.Session, retries: int = 3) -> bytes | None:
    """GET with retries. Returns None on 404, raises on other errors."""
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=60)
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
            continue
        if resp.status_code == 404:
            return None
        if resp.status_code >= 500 and attempt < retries - 1:
            time.sleep(2**attempt)
            continue
        resp.raise_for_status()
        return resp.content
    return None  # unreachable, keeps type-checkers happy


def download_verified(
    url: str,
    dest: Path | None = None,
    session: requests.Session | None = None,
) -> Path | None:
    """Download ``url`` and its .CHECKSUM, verify SHA256, cache under RAW_DIR.

    Returns the local path, or None if the remote file doesn't exist (404).
    Raises ValueError on checksum mismatch (the partial file is removed).
    """
    session = session or requests.Session()
    if dest is None:
        rel = url.removeprefix(BINANCE_VISION_BASE + "/")
        dest = RAW_DIR / rel
    dest.parent.mkdir(parents=True, exist_ok=True)

    checksum_body = _fetch(url + ".CHECKSUM", session)
    if checksum_body is None:
        return None
    expected = checksum_body.decode().split()[0].strip().lower()

    if dest.exists() and _sha256(dest) == expected:
        return dest

    body = _fetch(url, session)
    if body is None:
        return None
    dest.write_bytes(body)
    actual = _sha256(dest)
    if actual != expected:
        dest.unlink(missing_ok=True)
        raise ValueError(f"checksum mismatch for {url}: expected {expected}, got {actual}")
    return dest


def read_zip_csv(path: Path) -> bytes:
    """Return the raw CSV bytes from a single-file bulk zip."""
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if len(names) != 1:
            raise ValueError(f"{path} contains {len(names)} members, expected 1")
        return zf.read(names[0])


def list_symbols(session: requests.Session | None = None) -> list[str]:
    """Enumerate all symbol directories (including delisted) via S3 listing."""
    session = session or requests.Session()
    prefix = f"{UM_FUTURES_PREFIX}/monthly/klines/"
    symbols: list[str] = []
    marker = ""
    while True:
        url = f"{_S3_LIST_URL}?delimiter=/&prefix={prefix}"
        if marker:
            url += f"&marker={marker}"
        resp = session.get(url, timeout=60)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        prefixes = [
            el.find(f"{_S3_NS}Prefix").text
            for el in root.iter(f"{_S3_NS}CommonPrefixes")
        ]
        symbols.extend(p.removeprefix(prefix).rstrip("/") for p in prefixes)
        if root.findtext(f"{_S3_NS}IsTruncated") != "true":
            break
        marker = root.findtext(f"{_S3_NS}NextMarker") or prefixes[-1]
    return symbols


def month_range(start: str, end: str) -> list[str]:
    """Inclusive list of 'YYYY-MM' strings from start to end."""
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return months
