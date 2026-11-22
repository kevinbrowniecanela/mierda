"""Histórico diario ajustado (splits y dividendos) de acciones de EE. UU. desde Yahoo.

Limitación importante: Yahoo solo sirve empresas que HOY cotizan, así que el
universo tiene sesgo de supervivencia (favorece a las estrategias de compra).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent.parent / "state" / "research_data" / "stocks"
TICKERS = """AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO ORCL ADBE CRM AMD INTC CSCO QCOM TXN IBM NOW INTU MU AMAT LRCX KLAC
NFLX DIS CMCSA T VZ TMUS PEP KO WMT COST HD LOW MCD SBUX NKE TGT PG CL MDLZ MO PM
JPM BAC WFC C GS MS AXP BLK SCHW USB PNC V MA PYPL COF BRK-B
UNH JNJ PFE MRK ABBV LLY BMY AMGN GILD CVS MDT ABT TMO DHR ISRG ZTS REGN VRTX
XOM CVX COP SLB EOG OXY MPC PSX
CAT DE BA HON GE LMT RTX UPS FDX UNP MMM EMR ETN
NEE DUK SO D AEP
LIN APD SHW FCX NEM DOW
AMT PLD EQIX SPG O
F GM UBER ABNB BKNG MAR EBAY ROST TJX
SPY QQQ IWM GLD TLT""".split()


def fetch(sym: str) -> dict | None:
    for attempt in range(4):
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                         params={"period1": 1262304000, "period2": int(time.time()), "interval": "1d", "events": "div|split"},
                         headers={"User-Agent": "Mozilla/5.0 HyperAegis-research"}, timeout=30)
        if r.status_code == 200:
            res = r.json()["chart"]["result"]
            return res[0] if res else None
        time.sleep(2 * (attempt + 1))
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for sym in sys.argv[1:] or TICKERS:
        f = OUT / f"{sym}.json"
        if f.exists() and time.time() - f.stat().st_mtime < 24 * 3600:
            continue
        res = fetch(sym)
        if not res or not res.get("timestamp"):
            print(f"{sym:7s} SIN DATOS", flush=True)
            continue
        q = res["indicators"]["quote"][0]
        adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose") or q["close"]
        rows = []
        for i, t in enumerate(res["timestamp"]):
            o, h, l, c, v, a = q["open"][i], q["high"][i], q["low"][i], q["close"][i], q["volume"][i], adj[i]
            if None in (o, h, l, c, a) or not c:
                continue
            k = a / c
            rows.append([t, o * k, h * k, l * k, a, v or 0, c])
        f.write_text(json.dumps(rows))
        print(f"{sym:7s} {len(rows):5d} velas", flush=True)
        time.sleep(0.6)
    print("DESCARGA_OK", flush=True)


if __name__ == "__main__":
    main()
