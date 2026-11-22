"""Picos de volumen con gran rango en acciones de EE. UU. (huecos tras resultados y noticias).

Evento (en el cierre de la vela i): volumen >= V x media de 20 días previos, rango >= R x ATR,
vela alcista (opcional: tendencia de fondo, hueco de apertura >= 3 %). Entrada larga en la apertura
de i+1. Se mide frente a ENTRAR AL AZAR con la misma salida en el mismo universo.

Robustez: por periodos (<2018 / >=2018), por mitades de tickers (pares / impares), errores estándar
agrupados por semana (los eventos se concentran en días de mercado) y costes de ejecución duplicados.
Sesgo conocido: solo empresas que hoy cotizan (Yahoo).

    python research/stocks_download.py <tickers>   # muestra ampliada
    python research/volspike_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab  # noqa: E402
import stocks_study as ss  # noqa: E402
from quant.strategies import atr, ema  # noqa: E402

TOP = 100
SPLIT = pd.Timestamp("2018-01-01", tz="UTC")


def spec(df: pd.DataFrame, V: float, R: float, trend: bool, gap: bool, exit_: str) -> dict:
    a = atr(df)
    vr = df["volume"] / df["volume"].rolling(20).mean().shift(1)
    rng = (df["high"] - df["low"]) / a.shift()
    cond = (vr >= V) & (rng >= R) & (df["close"].pct_change() > 0)
    if trend:
        cond &= df["close"] > ema(df["close"], 200)
    if gap:
        cond &= (df["open"] / df["close"].shift() - 1) >= 0.03
    return exit_spec(cond.fillna(False).to_numpy().astype(int), exit_, len(df))


def exit_spec(sig, exit_: str, n: int) -> dict:
    if exit_ == "chandelier":
        return dict(sig=sig, stop_k=1.5, exit="chandelier", trail_k=2.0)
    return dict(sig=sig, stop_k=2.0, exit="signal", max_bars=10, exit_long=np.zeros(n, bool), exit_short=np.zeros(n, bool))


def random_spec(df: pd.DataFrame, exit_: str, seed: int, p: float = 0.01) -> dict:
    rng = np.random.default_rng(seed + len(df))
    return exit_spec((rng.random(len(df)) < p).astype(int), exit_, len(df))


def run(frames, make, elig):
    out = []
    for c, df in frames.items():
        if len(df) < 280:
            continue
        mask = elig[c].reindex(df.index).fillna(False).to_numpy() if c in elig else None
        out += lab.simulate(c, df, make(df), "1d", mask)
    return out


def cluster_t(trades: list) -> float:
    """t con errores agrupados por semana (eventos correlacionados en el tiempo)."""
    if len(trades) < 30:
        return float("nan")
    r = np.array([t.r for t in trades]); m = r.mean()
    wk = pd.Series([t.t_in.strftime("%G-%V") for t in trades])
    dev = pd.Series(r - m).groupby(wk.values).sum()
    se = np.sqrt((dev ** 2).sum()) / len(r)
    return float(m / se) if se > 0 else float("nan")


def main() -> None:
    stocks = ss.load_stocks()
    frames = {c: d for c, d in stocks.items() if c not in ss.BENCH}
    dv = pd.DataFrame({c: d["raw_close"] * d["volume"] for c, d in frames.items()})
    liq = dv.rolling(30, min_periods=20).median().shift(1)
    el = liq.rank(axis=1, ascending=False) <= TOP
    elig = {c: el[c] for c in el.columns}
    ss.cost_mode("stocks")
    print(f"Muestra: {len(frames)} acciones, universo diario = top {TOP} por liquidez (punto en el tiempo)\n")
    tick_even = {c for i, c in enumerate(sorted(frames)) if i % 2 == 0}

    print("=== 1. Rejilla de parámetros (solo largos; salida chandelier: stop 1,5 ATR, trailing 2 ATR) ===")
    print(f"   {'V':>3} {'R':>4} {'tend':>5} {'gap3%':>6} | {'n':>5} {'exp R':>7} {'azar':>7} {'ventaja':>8} {'t(sem)':>7} | {'<2018':>7} {'>=2018':>7} {'pares':>7} {'impares':>8}")
    rand = {}
    for ex in ("chandelier", "hold10"):
        rr = []
        for seed in range(6):
            rr += run(frames, lambda d, s=seed, e=ex: random_spec(d, e, s * 131), elig)
        rand[ex] = float(np.mean([t.r for t in rr]))
    print(f"   (azar chandelier {rand['chandelier']:+.3f}R, azar mantener 10 días {rand['hold10']:+.3f}R)")
    rows = []
    for V in (2, 3, 5):
        for R in (1.5, 2, 3):
            for trend in (False, True):
                for gap in (False, True):
                    tr = run(frames, lambda d: spec(d, V, R, trend, gap, "chandelier"), elig)
                    if len(tr) < 60:
                        continue
                    rs = np.array([t.r for t in tr])
                    e1 = np.mean([t.r for t in tr if t.t_in < SPLIT]) if any(t.t_in < SPLIT for t in tr) else float("nan")
                    e2 = np.mean([t.r for t in tr if t.t_in >= SPLIT])
                    ev = np.mean([t.r for t in tr if t.coin in tick_even]); od = np.mean([t.r for t in tr if t.coin not in tick_even])
                    rows.append((V, R, trend, gap, len(tr), rs.mean(), rs.mean() - rand["chandelier"], cluster_t(tr), e1, e2, ev, od))
    for r in sorted(rows, key=lambda x: -x[6])[:14]:
        print(f"   {r[0]:>3} {r[1]:>4} {'sí' if r[2] else 'no':>5} {'sí' if r[3] else 'no':>6} | {r[4]:>5d} {r[5]:>+7.3f} {rand['chandelier']:>+7.3f} {r[6]:>+8.3f} {r[7]:>+7.2f} | {r[8]:>+7.3f} {r[9]:>+7.3f} {r[10]:>+7.3f} {r[11]:>+8.3f}")
    positives = sum(1 for r in rows if r[6] > 0)
    print(f"   Combinaciones con ventaja positiva sobre el azar: {positives} de {len(rows)}")

    print("\n=== 2. Parámetros del bot (V=3, R=2, con tendencia): robustez ===")
    base = run(frames, lambda d: spec(d, 3, 2, True, False, "chandelier"), elig)
    hold = run(frames, lambda d: spec(d, 3, 2, True, False, "hold10"), elig)
    for label, tr, rnd in (("salida chandelier", base, rand["chandelier"]), ("mantener 10 días", hold, rand["hold10"])):
        rs = np.array([t.r for t in tr])
        lo, hi, p0 = __import__("validate").bootstrap(rs, n=8000)
        print(f"   {label:20s} n={len(rs):4d} exp {rs.mean():+.3f}R (azar {rnd:+.3f}R) IC95 [{lo:+.3f}, {hi:+.3f}] t semanal {cluster_t(tr):+.2f}")
    print("   Por año (chandelier): " + "  ".join(f"{y}:{np.mean([t.r for t in base if t.t_in.year == y]):+.2f}R({sum(1 for t in base if t.t_in.year == y)})"
                                                    for y in sorted({t.t_in.year for t in base})))
    lab.SLIP_SIDE = 0.0015
    costly = run(frames, lambda d: spec(d, 3, 2, True, False, "chandelier"), elig)
    lab.SLIP_SIDE = 0.0005
    print(f"   Con deslizamiento triple (0,15 %/lado, huecos con peor ejecución): exp {np.mean([t.r for t in costly]):+.3f}R")

    print("\n=== 3. Cartera SIN apalancamiento (cuenta al contado) frente a SPY ===")
    spy = stocks["SPY"]["close"]; c, d = ss.cagr_dd(spy[spy.index >= min(t.t_in for t in base)])
    rnd_tr = run(frames, lambda dfx: random_spec(dfx, "chandelier", 77, 0.004), elig)
    print(f"   SPY mismo periodo: {c:+.1f} % anual, caída {d:.0f} %")
    for label, tr in (("picos de volumen (bot)", base), ("entradas al azar, misma salida", rnd_tr)):
        for risk, mp in ((0.01, 5), (0.02, 5), (0.02, 10)):
            p = lab.portfolio(tr, risk=risk, max_pos=mp, lev_cap=1.0, start=224)
            print(f"   {label:32s} riesgo {risk:.0%}, {mp:>2} pos: {p['cagr']:+5.1f} % anual, caída {p['maxdd']:4.1f} %, {p['taken']} de {len(tr)} operaciones")
    ens_names = ["donchian55_tendencia", "squeeze_breakout", "donchian20_tendencia"]
    from quant.strategies import STRATEGIES
    el40 = {c: (liq.rank(axis=1, ascending=False) <= 40)[c] for c in liq.columns}
    sig_sum = lambda d: (lambda sp: (sp.update(sig=np.clip(sp["sig"], 0, None)), sp)[1])(STRATEGIES["ensemble_diario"](d))
    ens = run(frames, sig_sum, el40)
    print(f"   Conjunto de tendencia largo, top 40 ({len(ens)} ops):")
    for risk, mp in ((0.01, 5), (0.02, 5)):
        p = lab.portfolio(ens, risk=risk, max_pos=mp, lev_cap=1.0, start=224)
        print(f"      riesgo {risk:.0%}, {mp} pos, sin apalancamiento: {p['cagr']:+5.1f} % anual, caída {p['maxdd']:4.1f} %")
    both = sorted(base + ens, key=lambda t: t.t_in)
    for risk, mp in ((0.01, 5), (0.02, 5)):
        p = lab.portfolio(both, risk=risk, max_pos=mp, lev_cap=1.0, start=224)
        print(f"   Tendencia + picos de volumen juntos, riesgo {risk:.0%}, {mp} pos: {p['cagr']:+5.1f} % anual, caída {p['maxdd']:4.1f} %")


if __name__ == "__main__":
    main()
