"""Seguimiento del estudio de acciones: ¿hay ventaja real o solo deriva alcista?

1. Largos solamente (los cortos pierden en acciones).
2. NULA: entradas largas AL AZAR con la misma salida (stop 2 ATR, trailing 3 ATR).
   La ventaja real de una señal = su expectativa - la del azar.
3. Control sin sesgo de supervivencia: ETFs (SPY, QQQ, IWM, GLD, TLT).
4. Cartera frente a comprar y mantener SPY, con distintos riesgos.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab  # noqa: E402
import stocks_study as ss  # noqa: E402
import validate as v  # noqa: E402
from quant.strategies import STRATEGIES  # noqa: E402


def long_only(fn):
    def make(df):
        spec = fn(df)
        sig = np.asarray(spec["sig"]).copy()
        sig[sig < 0] = 0
        spec["sig"] = sig
        return spec
    return make


def random_long(p: float, seed: int):
    def make(df):
        rng = np.random.default_rng(seed + len(df))
        return dict(sig=(rng.random(len(df)) < p).astype(int), stop_k=2.0, exit="chandelier", trail_k=3.0)
    return make


def run(frames, fn, elig=None, since=None):
    out = []
    for c, df in frames.items():
        if len(df) < 280:
            continue
        mask = elig[c].reindex(df.index).fillna(False).to_numpy() if elig is not None and c in elig else None
        out += lab.simulate(c, df, fn(df), "1d", mask)
    return [t for t in out if since is None or t.t_in >= since]


def fmt(tr, label):
    rs = np.array([t.r for t in tr])
    st = lab.stats(rs)
    return f"   {label:42s} n={len(rs):5d} exp={rs.mean():+.3f}R t={st['t']:+.2f} gan={st['win']:4.1f}%"


def main() -> None:
    stocks = ss.load_stocks()
    ss.cost_mode("stocks")
    elig = ss.stock_elig(stocks)
    names = ["donchian55_tendencia", "donchian20_tendencia", "pullback_rsi40", "squeeze_breakout",
             "meanrev_rsi2", "volspike_chandelier", "ensemble_diario"]
    biased = {c: d for c, d in stocks.items() if c not in ss.BENCH}
    etfs = {c: stocks[c] for c in ("SPY", "QQQ", "IWM", "GLD", "TLT") if c in stocks}

    print("=== 1 y 2. LARGOS vs ENTRADAS AL AZAR (acciones top 40, 2010-2026, misma salida) ===")
    null = []
    for seed in range(5):
        null += run(biased, random_long(0.02, seed * 101), elig)
    base = np.mean([t.r for t in null])
    print(fmt(null, "AZAR largo (5 sorteos) — la deriva alcista"))
    res = {}
    for n in names:
        tr = run(biased, long_only(STRATEGIES[n]), elig)
        res[n] = tr
        e = np.mean([t.r for t in tr])
        print(fmt(tr, f"{n} (solo largos)") + f"  ventaja sobre azar {e - base:+.3f}R")

    print("\n=== 3. Control SIN sesgo de supervivencia: ETFs (SPY, QQQ, IWM, GLD, TLT), solo largos ===")
    null_e = []
    for seed in range(20):
        null_e += run(etfs, random_long(0.02, seed * 53))
    base_e = np.mean([t.r for t in null_e])
    print(fmt(null_e, "AZAR largo (20 sorteos)"))
    for n in names:
        if n == "volspike_chandelier":
            continue
        tr = run(etfs, long_only(STRATEGIES[n]))
        if len(tr) >= 20:
            print(fmt(tr, f"{n} (solo largos)") + f"  ventaja {np.mean([t.r for t in tr]) - base_e:+.3f}R")

    print("\n=== 4. Cartera (solo largos) frente a comprar y mantener, 2010-2026 ===")
    spy = stocks["SPY"]["close"]
    for n in ("donchian55_tendencia", "pullback_rsi40", "ensemble_diario"):
        tr = res[n]
        spy_w = spy[spy.index >= min(t.t_in for t in tr)]
        c, d = ss.cagr_dd(spy_w)
        print(f"   {n}: (SPY mismo periodo: {c:+.1f} % anual, caída {d:.0f} %)")
        for risk, mp in ((0.01, 5), (0.02, 5), (0.02, 10), (0.03, 10)):
            p = lab.portfolio(tr, risk=risk, max_pos=mp, start=224)
            print(f"      riesgo {risk:.0%}, {mp:>2} posiciones: {p['cagr']:+5.1f} % anual, caída máx. {p['maxdd']:4.1f} %, {p['taken']} operaciones")
    print("\nNota: el sesgo de supervivencia infla los largos en acciones; el control con ETFs y el azar lo acotan.")


if __name__ == "__main__":
    main()
