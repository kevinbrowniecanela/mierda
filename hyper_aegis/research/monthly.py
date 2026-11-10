"""Prueba de arranque con capital pequeño: retorno MENSUAL del conjunto diario.

    python research/monthly.py --eur 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab  # noqa: E402


def monthly_returns(curve: list[tuple[pd.Timestamp, float]]) -> pd.Series:
    s = pd.Series([e for _, e in curve], index=pd.DatetimeIndex([t for t, _ in curve]))
    s = s.groupby(level=0).last().resample("ME").last().ffill()
    return s.pct_change().dropna() * 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eur", type=float, default=200)
    ap.add_argument("--eurusd", type=float, default=0.89342, help="EUR por 1 USD (BCE)")
    ap.add_argument("--strategy", default="ensemble_diario")
    args = ap.parse_args()
    start = args.eur / args.eurusd
    universe = json.loads((lab.DATA / "universe.json").read_text())
    crypto = [c for c in universe if ":" not in c and c != "PAXG"]
    res = lab.evaluate(crypto, "1d", args.strategy)
    trades = res["trades"]
    print(f"Capital inicial: {args.eur:.0f} EUR = {start:.2f} USDC | estrategia {args.strategy} (1d, {len(crypto)} criptos)")
    print(f"Operaciones: {len(trades)} | expectativa IS {res['IS']['exp']:+.3f}R / OOS {res['OOS']['exp']:+.3f}R\n")
    print(f"{'riesgo':>6} {'pos':>3} {'final €':>9} {'anual%':>7} {'maxDD%':>7} {'mes mediano%':>12} {'mes medio%':>10} {'peor mes%':>9} {'mejor mes%':>10} {'meses +%':>8} {'ops':>4}")
    keep = None
    for risk in (0.01, 0.02, 0.03):
        for max_pos in (3, 5):
            p = lab.portfolio(trades, risk=risk, max_pos=max_pos, start=start)
            m = monthly_returns(p["curve"])
            print(f"{risk:>6.0%} {max_pos:>3} {p['final'] * args.eurusd:>9.0f} {p['cagr']:>7.1f} {p['maxdd']:>7.1f} "
                  f"{m.median():>12.2f} {m.mean():>10.2f} {m.min():>9.1f} {m.max():>10.1f} {(m > 0).mean() * 100:>8.0f} {p['taken']:>4}")
            if risk == 0.02 and max_pos == 3:
                keep = (p, m)
    p, m = keep
    print("\nDetalle con 2 % de riesgo y 3 posiciones:")
    by_year = (1 + m / 100).groupby(m.index.year).prod() - 1
    print("  Retorno por año:", ", ".join(f"{y}: {v * 100:+.0f}%" for y, v in by_year.items()))
    print("  Últimos 12 meses:", ", ".join(f"{t:%b%y} {v:+.1f}%" for t, v in m.tail(12).items()))
    flat = (m == 0).mean() * 100
    print(f"  Meses sin ningún cierre: {flat:.0f}% | meses negativos: {(m < 0).mean() * 100:.0f}%")
    q = np.percentile(m, [10, 25, 50, 75, 90])
    print(f"  Distribución mensual (p10/p25/p50/p75/p90): {q[0]:+.1f} / {q[1]:+.1f} / {q[2]:+.1f} / {q[3]:+.1f} / {q[4]:+.1f} %")
    out = lab.ROOT / "state" / "monthly_200.json"
    out.write_text(json.dumps({"curve": [(str(t), e) for t, e in p["curve"]], "monthly": {str(k.date()): v for k, v in m.items()}}))


if __name__ == "__main__":
    main()
