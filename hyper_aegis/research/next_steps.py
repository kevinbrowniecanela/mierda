"""¿Por dónde seguir? Riesgo, aportaciones periódicas, referencia y variantes.

Todo sobre el universo honesto (top 20 punto-en-el-tiempo con deslistados,
era Hyperliquid) y con funding real donde está descargado.

    python research/next_steps.py --monthly-eur 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab  # noqa: E402
import validate as v  # noqa: E402
from monthly import monthly_returns  # noqa: E402
from quant.strategies import ema, s_ensemble  # noqa: E402

EUR = v.EUR


def btc_regime(data) -> pd.Series:
    btc = data["BTC"]["close"]
    return btc > ema(btc, 200)


def with_regime(data, long_only: bool = False, use_btc: bool = True):
    """Conjunto diario filtrado: largos solo con BTC sobre su EMA200; cortos solo por debajo."""
    bull = btc_regime(data)

    def make(df: pd.DataFrame) -> dict:
        spec = s_ensemble(df)
        sig = np.asarray(spec["sig"]).copy()
        if use_btc:
            b = bull.reindex(df.index).ffill().fillna(False).to_numpy()
            sig[(sig > 0) & ~b] = 0
            sig[(sig < 0) & b] = 0
        if long_only:
            sig[sig < 0] = 0
        spec["sig"] = sig
        return spec
    return make


def hold_stats(close: pd.Series) -> tuple[float, float]:
    c = close[close.index >= v.HL_START]
    years = (c.index[-1] - c.index[0]).days / 365.25
    return ((c.iloc[-1] / c.iloc[0]) ** (1 / years) - 1) * 100, float(((c.cummax() - c) / c.cummax()).max() * 100)


def contributions(monthly_pct: np.ndarray, start: float, add: float, years: int, sims: int = 5000, seed: int = 5):
    """Capital final con aportación mensual, remuestreando los retornos mensuales observados."""
    rng = np.random.default_rng(seed)
    n = years * 12
    draws = rng.choice(monthly_pct / 100, size=(sims, n), replace=True)
    eq = np.full(sims, start)
    for m in range(n):
        eq = eq * (1 + draws[:, m]) + add
    return np.percentile(eq, [10, 50, 90])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--monthly-eur", type=float, default=100)
    args = ap.parse_args()
    data, _ = v.load_all()
    elig = v.point_in_time(data)
    lab.USE_REAL_FUNDING = bool(lab.load_funding())
    base = v.run(data, "ensemble_diario", elig)

    print("=== 1. ¿Subir el riesgo mejora el retorno anual? (sistema actual, universo honesto) ===")
    print(f"   {'riesgo':>6} {'anual%':>7} {'maxDD%':>7} {'peor mes%':>9} {'anual/DD':>8}")
    for risk in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08):
        p = lab.portfolio(base, risk=risk, max_pos=3, start=224)
        m = monthly_returns(p["curve"])
        print(f"   {risk:>6.1%} {p['cagr']:>7.1f} {p['maxdd']:>7.1f} {m.min():>9.1f} {p['cagr'] / p['maxdd'] if p['maxdd'] else 0:>8.2f}")

    print("\n=== 2. Referencia: comprar y mantener desde mayo de 2023 (sin apalancamiento) ===")
    for coin in ("BTC", "ETH", "SOL", "PAXG"):
        if coin in data:
            cagr, dd = hold_stats(data[coin]["close"])
            print(f"   {coin:5s} anual {cagr:+6.1f}%  caída máxima {dd:4.1f}%")

    print("\n=== 3. Variantes razonadas del sistema (hipótesis, no ajustadas a los datos) ===")
    variants = {
        "Actual: largos y cortos": base,
        "Con filtro de régimen BTC (EMA200)": v.run(data, with_regime(data), elig),
        "Solo largos + filtro BTC": v.run(data, with_regime(data, long_only=True), elig),
        "Solo largos, sin filtro": v.run(data, with_regime(data, long_only=True, use_btc=False), elig),
    }
    for label, tr in variants.items():
        print(v.line("   " + label, tr))
        rs = np.array([t.r for t in tr])
        lo, hi, p0 = v.bootstrap(rs, n=8000)
        years = "  ".join(f"{y}:{np.mean([t.r for t in tr if t.t_in.year == y]):+.2f}R" for y in sorted({t.t_in.year for t in tr}))
        print(f"        IC95 [{lo:+.3f}, {hi:+.3f}]  P(<=0)={p0:.0f}%  |  {years}")

    print(f"\n=== 4. Aportar {args.monthly_eur:.0f} EUR al mes empezando con 200 EUR ===")
    print("   Capital aportado:", ", ".join(f"{y} años = {200 + args.monthly_eur * 12 * y:,.0f} €" for y in (3, 5, 10)))
    print(f"   {'escenario':38s} {'3 años':>10} {'5 años':>10} {'10 años':>11}   (mediana; entre paréntesis el caso malo p10)")
    for label, annual in (("Objetivo: 15 % anual constante", 15.0), ("Constante 5 % anual", 5.0), ("Sin rentabilidad (solo ahorro)", 0.0)):
        mr = (1 + annual / 100) ** (1 / 12) - 1
        cells = []
        for y in (3, 5, 10):
            eq = 200.0
            for _ in range(y * 12):
                eq = eq * (1 + mr) + args.monthly_eur
            cells.append(f"{eq:>10,.0f}")
        print(f"   {label:38s} " + " ".join(cells))
    for label, tr, risk in (("Sistema actual, riesgo 1 %", base, 0.01), ("Sistema actual, riesgo 2 %", base, 0.02),
                            ("Solo largos + filtro BTC, riesgo 2 %", variants["Solo largos + filtro BTC"], 0.02)):
        m = monthly_returns(lab.portfolio(tr, risk=risk, max_pos=3, start=224)["curve"]).to_numpy()
        cells = []
        for y in (3, 5, 10):
            p10, p50, _ = contributions(m, 200, args.monthly_eur, y)
            cells.append(f"{p50:>7,.0f} ({p10:,.0f})")
        print(f"   {label:38s} " + " ".join(cells))


if __name__ == "__main__":
    main()
