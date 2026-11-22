"""Acciones de EE. UU. frente a cripto: mismo sistema, mismas pruebas.

1. Características de cada mercado (volatilidad, correlación, persistencia...).
2. Las 11 estrategias del laboratorio en acciones (diario, costes de acciones),
   por periodos, largos/cortos y frente a comprar y mantener SPY.
3. Comparación directa con cripto en el mismo periodo (mayo 2023 -> hoy).

Sesgo conocido: Yahoo solo ofrece empresas que HOY cotizan -> sesgo de
supervivencia que favorece a las compras. En cripto sí incluimos deslistados.

    python research/stocks_download.py && python research/stocks_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab  # noqa: E402
import validate as v  # noqa: E402
from quant.strategies import STRATEGIES  # noqa: E402

SD = lab.DATA / "stocks"
TOP = 40
SINCE_CRYPTO = v.HL_START
BENCH = {"SPY", "QQQ", "IWM", "GLD", "TLT"}


def load_stocks() -> dict[str, pd.DataFrame]:
    out = {}
    for f in SD.glob("*.json"):
        rows = json.loads(f.read_text())
        if len(rows) < 300:
            continue
        a = np.array(rows, dtype=float)
        idx = pd.to_datetime(a[:, 0], unit="s", utc=True).normalize()
        df = pd.DataFrame({"open": a[:, 1], "high": a[:, 2], "low": a[:, 3], "close": a[:, 4],
                           "volume": a[:, 5], "raw_close": a[:, 6]}, index=idx)
        out[f.stem] = df[~df.index.duplicated()]
    return out


def stock_elig(stocks: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    dv = pd.DataFrame({c: d["raw_close"] * d["volume"] for c, d in stocks.items() if c not in BENCH})
    liq = dv.rolling(30, min_periods=20).median().shift(1)
    el = liq.rank(axis=1, ascending=False) <= TOP
    return {c: el[c] for c in el.columns}


def run_stocks(stocks, name, elig, since=None, until=None, side=0):
    fn = STRATEGIES[name]
    trades = []
    for c, df in stocks.items():
        if c in BENCH or len(df) < 280:
            continue
        mask = elig[c].reindex(df.index).fillna(False).to_numpy() if elig is not None and c in elig else None
        trades += lab.simulate(c, df, fn(df), "1d", mask)
    trades = [t for t in trades if (since is None or t.t_in >= since) and (until is None or t.t_in < until)]
    return [t for t in trades if side == 0 or t.side == side]


def cost_mode(kind: str) -> None:
    if kind == "stocks":
        lab.FEE_SIDE, lab.SLIP_SIDE, lab.FUNDING_HOURLY, lab.USE_REAL_FUNDING = 0.0, 0.0005, 0.0, False
    else:
        lab.FEE_SIDE, lab.SLIP_SIDE, lab.FUNDING_HOURLY = 0.00045, 0.0003, 0.0000125
        lab.USE_REAL_FUNDING = True


def characteristics(frames: dict[str, pd.DataFrame], days_year: int, start: pd.Timestamp) -> dict:
    rets = pd.DataFrame({c: d["close"].pct_change() for c, d in frames.items()})
    rets = rets[rets.index >= start].dropna(how="all")
    vol = rets.std() * np.sqrt(days_year)
    atrp = []
    for c, d in frames.items():
        d = d[d.index >= start]
        tr = pd.concat([d.high - d.low, (d.high - d.close.shift()).abs(), (d.low - d.close.shift()).abs()], axis=1).max(axis=1)
        atrp.append(float((tr.rolling(14).mean() / d.close).median()))
    ac1 = rets.apply(lambda s: s.autocorr(1))
    vr5 = rets.apply(lambda s: s.rolling(5).sum().var() / (5 * s.var()))
    corr = rets.corr().to_numpy()
    avg_corr = float((corr.sum() - len(corr)) / (len(corr) * (len(corr) - 1)))
    z = (rets - rets.mean()) / rets.std()
    tail = float((z.abs() > 3).mean().mean() * 100)
    dd = []
    for c, d in frames.items():
        d = d[d.index >= start]["close"]
        dd.append(float(((d.cummax() - d) / d.cummax()).max()))
    return dict(vol=float(vol.median() * 100), atrp=float(np.median(atrp) * 100), ac1=float(ac1.median()), vr5=float(vr5.median()),
                corr=avg_corr, tail=tail, dd=float(np.median(dd) * 100))


def cagr_dd(close: pd.Series) -> tuple[float, float]:
    yrs = (close.index[-1] - close.index[0]).days / 365.25
    return float(((close.iloc[-1] / close.iloc[0]) ** (1 / yrs) - 1) * 100), float(((close.cummax() - close) / close.cummax()).max() * 100)


def main() -> None:
    stocks = load_stocks()
    n_st = len([c for c in stocks if c not in BENCH])
    first = min(d.index[0] for c, d in stocks.items() if c not in BENCH)
    print(f"Acciones: {n_st} empresas (+{len(BENCH & set(stocks))} ETF de referencia), histórico desde {first.date()}\n")
    elig = stock_elig(stocks)

    # ---- 1. características ------------------------------------------- #
    data_c, _ = v.load_all()
    crypto_top = [c for c in json.loads((lab.DATA / "universe.json").read_text()) if ":" not in c and c != "PAXG"]
    crypto_fr = {c: data_c[c] for c in crypto_top if c in data_c}
    liquid = sorted((c for c in stocks if c not in BENCH), key=lambda c: -(stocks[c]["raw_close"] * stocks[c]["volume"]).tail(250).median())[:TOP]
    st_fr = {c: stocks[c] for c in liquid}
    a = characteristics(st_fr, 252, SINCE_CRYPTO)
    b = characteristics(crypto_fr, 365, SINCE_CRYPTO)
    print(f"=== 1. Cómo se comportan (desde {SINCE_CRYPTO.date()}; acciones: 40 más líquidas; cripto: 20 grandes) ===")
    rows = [("Volatilidad anual (mediana)", f"{a['vol']:.0f} %", f"{b['vol']:.0f} %"),
            ("Rango diario medio (ATR / precio)", f"{a['atrp']:.2f} %", f"{b['atrp']:.2f} %"),
            ("Correlación media entre activos", f"{a['corr']:.2f}", f"{b['corr']:.2f}"),
            ("Autocorrelación diaria (lag 1)", f"{a['ac1']:+.3f}", f"{b['ac1']:+.3f}"),
            ("Ratio de varianza a 5 días (1 = azar, >1 tendencia)", f"{a['vr5']:.2f}", f"{b['vr5']:.2f}"),
            ("Días extremos (> 3 desviaciones)", f"{a['tail']:.2f} %", f"{b['tail']:.2f} %"),
            ("Caída máxima mediana por activo", f"{a['dd']:.0f} %", f"{b['dd']:.0f} %")]
    print(f"   {'':54s} {'Acciones':>10} {'Cripto':>10}")
    for r in rows:
        print(f"   {r[0]:54s} {r[1]:>10} {r[2]:>10}")
    stop_cost = {k: 2 * (f + s) / (2.0 * x["atrp"] / 100) for k, (f, s, x) in
                 {"Acciones": (0.0, 0.0005, a), "Cripto": (0.00045, 0.0003, b)}.items()}
    print(f"   Coste de entrar y salir en R (stop a 2 ATR): acciones {stop_cost['Acciones']:.3f}R, cripto {stop_cost['Cripto']:.3f}R\n")

    # ---- 2. estrategias en acciones ----------------------------------- #
    names = list(STRATEGIES)
    cost_mode("stocks")
    split = pd.Timestamp("2018-01-01", tz="UTC")
    print("=== 2. Las 11 estrategias en ACCIONES (top 40 por liquidez, punto en el tiempo; costes: 0,05 %/lado) ===")
    print(f"   {'estrategia':24s} {'n':>5} {'exp R':>7} {'t':>6} | {'<2018':>7} {'>=2018':>7} {'>=may23':>8} | {'largos':>7} {'cortos':>7}")
    res = {}
    for n in names:
        tr = run_stocks(stocks, n, elig)
        if len(tr) < 30:
            continue
        rs = np.array([t.r for t in tr])
        e1 = np.mean([t.r for t in tr if t.t_in < split]) if any(t.t_in < split for t in tr) else float("nan")
        e2 = np.mean([t.r for t in tr if t.t_in >= split])
        e3 = [t.r for t in tr if t.t_in >= SINCE_CRYPTO]
        lg = np.mean([t.r for t in tr if t.side > 0]) if any(t.side > 0 for t in tr) else float("nan")
        sh = np.mean([t.r for t in tr if t.side < 0]) if any(t.side < 0 for t in tr) else float("nan")
        st = lab.stats(rs)
        res[n] = tr
        print(f"   {n:24s} {len(tr):>5d} {rs.mean():>+7.3f} {st['t']:>+6.2f} | {e1:>+7.3f} {e2:>+7.3f} {np.mean(e3) if e3 else float('nan'):>+8.3f} | {lg:>+7.3f} {sh:>+7.3f}")

    # ---- 3. el conjunto, cartera y referencia ------------------------- #
    ens = res.get("ensemble_diario")
    print("\n=== 3. Conjunto diario (el del bot) en acciones frente a comprar y mantener ===")
    if ens:
        p = lab.portfolio(ens, risk=0.01, max_pos=3, start=224)
        p2 = lab.portfolio(ens, risk=0.02, max_pos=5, start=224)
        spy = stocks["SPY"]["close"]; spy = spy[spy.index >= min(t.t_in for t in ens)]
        qqq = stocks["QQQ"]["close"]; qqq = qqq[qqq.index >= spy.index[0]]
        print(f"   Bot, 1 % riesgo, 3 pos: {p['cagr']:+.1f} % anual, caída máx. {p['maxdd']:.1f} %  ({p['years']:.1f} años, {p['taken']} operaciones)")
        print(f"   Bot, 2 % riesgo, 5 pos: {p2['cagr']:+.1f} % anual, caída máx. {p2['maxdd']:.1f} %")
        for lab_, s in (("SPY (comprar y mantener)", spy), ("QQQ (comprar y mantener)", qqq)):
            c, d = cagr_dd(s)
            print(f"   {lab_:24s}: {c:+.1f} % anual, caída máx. {d:.1f} %")
        print("   Por año (conjunto): " + "  ".join(f"{y}:{np.mean([t.r for t in ens if t.t_in.year == y]):+.2f}R" for y in sorted({t.t_in.year for t in ens})[-12:]))
        rs = np.array([t.r for t in ens])
        lo, hi, p0 = v.bootstrap(rs, n=10000)
        print(f"   Expectativa {rs.mean():+.3f}R, IC95 [{lo:+.3f}, {hi:+.3f}], probabilidad de ser <= 0: {p0:.1f} %")

    # ---- 4. mismo periodo que cripto ---------------------------------- #
    print(f"\n=== 4. Apples-to-apples: desde {SINCE_CRYPTO.date()} — expectativa por operación (R) ===")
    print(f"   {'estrategia':24s} {'Acciones':>9} {'n':>5}   {'Cripto':>8} {'n':>5}")
    cost_mode("crypto")
    elig_c = v.point_in_time(data_c)
    for n in names:
        tc = v.run(data_c, n, elig_c)
        ts_ = [t for t in res.get(n, []) if t.t_in >= SINCE_CRYPTO]
        if len(tc) < 20 or len(ts_) < 20:
            continue
        print(f"   {n:24s} {np.mean([t.r for t in ts_]):>+9.3f} {len(ts_):>5d}   {np.mean([t.r for t in tc]):>+8.3f} {len(tc):>5d}")


if __name__ == "__main__":
    main()
