"""Descarga histórico multi-timeframe (1h/4h/1d) de Hyperliquid a state/research_data/.

Universo: top N cripto por volumen + oro (PAXG) + principales mercados
tradicionales del dex `xyz` (índices, acciones, petróleo, metales).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

URL = "https://api.hyperliquid.xyz/info"
OUT = Path(__file__).resolve().parent.parent / "state" / "research_data"
MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


def post(payload: dict, tries: int = 7):
    delay = 5
    for _ in range(tries):
        r = requests.post(URL, json=payload, timeout=30)
        if r.status_code == 200:
            return r.json()
        time.sleep(delay)
        delay = min(delay * 2, 60)
    r.raise_for_status()


def universe(n_crypto: int = 20, n_trad: int = 10) -> list[str]:
    meta, ctxs = post({"type": "metaAndAssetCtxs"})
    rows = [(a["name"], float(c.get("dayNtlVlm") or 0)) for a, c in zip(meta["universe"], ctxs) if not a.get("isDelisted")]
    crypto = [n for n, _ in sorted(rows, key=lambda r: -r[1])[:n_crypto]]
    if "PAXG" not in crypto:
        crypto.append("PAXG")
    meta, ctxs = post({"type": "metaAndAssetCtxs", "dex": "xyz"})
    rows = [(a["name"], float(c.get("dayNtlVlm") or 0)) for a, c in zip(meta["universe"], ctxs) if not a.get("isDelisted")]
    trad = [n for n, _ in sorted(rows, key=lambda r: -r[1])[:n_trad]]
    for must in ("xyz:GOLD", "xyz:SILVER", "xyz:SP500", "xyz:XYZ100"):
        if must in {n for n, _ in rows} and must not in trad:
            trad.append(must)
    return crypto + trad


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    coins = sys.argv[1:] or universe()
    (OUT / "universe.json").write_text(json.dumps(coins))
    now = int(time.time() * 1000)
    for coin in coins:
        for iv in ("1h", "4h", "1d"):
            f = OUT / f"{coin.replace(':', '_')}_{iv}.json"
            if f.exists() and time.time() - f.stat().st_mtime < 12 * 3600:
                continue
            rows = post({"type": "candleSnapshot", "req": {"coin": coin, "interval": iv, "startTime": now - MS[iv] * 5000, "endTime": now}})
            f.write_text(json.dumps(rows))
            print(f"{coin:14s} {iv} {len(rows):5d}", flush=True)
            time.sleep(5)
    print("DESCARGA_OK", flush=True)


if __name__ == "__main__":
    main()
