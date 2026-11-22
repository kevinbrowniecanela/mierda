"""Descarga datos frescos para el backtest del sistema `strategy/`.

Para los N activos de mayor volumen: velas 15m, 1h, 4h y 1d (las últimas 5.000 de cada una; el límite de la
API de Hyperliquid), metadatos (szDecimals) y funding horario real de los últimos ~210 días.
Reanudable: omite lo ya descargado en las últimas 12 horas.

    python research/strategy_data.py 20
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download import post  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "state" / "research_data" / "bt"
MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
FUND_DAYS = 212


def fresh(f: Path, hours: float = 12) -> bool:
    return f.exists() and time.time() - f.stat().st_mtime < hours * 3600


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    OUT.mkdir(parents=True, exist_ok=True)
    meta, ctxs = post({"type": "metaAndAssetCtxs"})
    rows = sorted(((float(c.get("dayNtlVlm") or 0), a) for a, c in zip(meta["universe"], ctxs) if not a.get("isDelisted")), key=lambda r: -r[0])
    coins = [a["name"] for _, a in rows[:n]]
    (OUT / "meta.json").write_text(json.dumps({a["name"]: {"szDecimals": a["szDecimals"], "maxLeverage": a.get("maxLeverage", 1)} for a in meta["universe"]}))
    (OUT / "coins.json").write_text(json.dumps(coins))
    now = int(time.time() * 1000)
    for k, coin in enumerate(coins, 1):
        for tf in ("15m", "1h", "4h", "1d"):
            f = OUT / f"{coin}_{tf}.json"
            if fresh(f):
                continue
            raw = post({"type": "candleSnapshot", "req": {"coin": coin, "interval": tf, "startTime": now - MS[tf] * 5000, "endTime": now}})
            f.write_text(json.dumps(raw))
            time.sleep(3.5)
        f = OUT / f"{coin}_funding.json"
        if not fresh(f):
            out, cur = [], now - FUND_DAYS * 86_400_000
            while cur < now:
                page = post({"type": "fundingHistory", "coin": coin, "startTime": cur, "endTime": now})
                time.sleep(2.4)
                if not page:
                    break
                out += [[int(r["time"]), float(r["fundingRate"])] for r in page]
                if len(page) < 500:
                    break
                cur = int(page[-1]["time"]) + 1
            f.write_text(json.dumps(out))
        print(f"[{k}/{len(coins)}] {coin} listo", flush=True)
    print("DATOS_OK", flush=True)


if __name__ == "__main__":
    main()
