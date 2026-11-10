"""Descarga el funding horario REAL de Hyperliquid para los periodos en que
las operaciones del backtest estuvieron abiertas.

La API devuelve 500 filas por petición, así que solo se piden los tramos
necesarios (por activo, uniendo operaciones consecutivas). Reanudable: guarda
cada activo en state/research_data/funding/COIN.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab  # noqa: E402
from download import post  # noqa: E402

OUT = lab.DATA / "funding"
HL_START = pd.Timestamp("2023-05-12", tz="UTC")


def fetch_span(coin: str, start_ms: int, end_ms: int) -> list[list]:
    rows: list[list] = []
    cur = start_ms
    while cur < end_ms:
        page = post({"type": "fundingHistory", "coin": coin, "startTime": cur, "endTime": end_ms})
        time.sleep(2.4)
        if not page:
            break
        rows += [[int(r["time"]), float(r["fundingRate"])] for r in page]
        if len(page) < 500:
            break
        cur = int(page[-1]["time"]) + 1
    return rows


def fetch_for_trades(trades: list[lab.Trade]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    by_coin: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for t in trades:
        if t.t_out > HL_START:
            by_coin.setdefault(t.coin, []).append((max(t.t_in, HL_START), t.t_out))
    total = len(by_coin)
    for n, (coin, spans) in enumerate(sorted(by_coin.items()), 1):
        f = OUT / f"{coin.replace(':', '_')}.json"
        blob = json.loads(f.read_text()) if f.exists() else {"rows": [], "spans": []}
        have = [(pd.Timestamp(a, unit="ms", tz="UTC"), pd.Timestamp(b, unit="ms", tz="UTC")) for a, b in blob["spans"]]
        # une operaciones separadas por menos de 10 días en un solo tramo
        merged: list[list[pd.Timestamp]] = []
        for a, b in sorted(spans):
            a, b = a - pd.Timedelta(hours=2), b + pd.Timedelta(hours=2)
            if merged and a - merged[-1][1] < pd.Timedelta(days=10):
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        new = 0
        for a, b in merged:
            if any(x <= a and b <= y for x, y in have):
                continue
            rows = fetch_span(coin, int(a.timestamp() * 1000), int(b.timestamp() * 1000))
            blob["rows"] += rows
            blob["spans"].append([int(a.timestamp() * 1000), int(b.timestamp() * 1000)])
            new += len(rows)
            f.write_text(json.dumps(blob))
        print(f"[{n}/{total}] {coin:10s} tramos {len(merged):3d} filas nuevas {new}", flush=True)
    print("FUNDING_OK", flush=True)
