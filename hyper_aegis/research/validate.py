"""Validación dura del sistema diario de HyperAegis.

1. Sesgo de supervivencia: universo "punto en el tiempo" -> cada día solo son
   elegibles los 20 mercados con más volumen de los 30 días ANTERIORES, entre
   TODOS los perps de Hyperliquid, incluidos los ya deslistados.
2. Solo la era operable: desde el lanzamiento de Hyperliquid (2023-05-12).
   Las velas anteriores son datos rellenados de otros mercados.
3. Funding real hora a hora (research/funding.py) en lugar del supuesto fijo.
4. Ventanas sucesivas (walk-forward): cada 6 meses se eligen las estrategias
   con los 12 meses anteriores y se operan los 6 siguientes.
5. Robustez de parámetros y bootstrap del resultado.

    python research/download.py --all-daily
    python research/validate.py --fetch-funding   # la primera vez (tarda)
    python research/validate.py
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
from monthly import monthly_returns  # noqa: E402
from quant.strategies import ENSEMBLE, STRATEGIES, s_donchian  # noqa: E402

HL_START = pd.Timestamp("2023-05-12", tz="UTC")
TOP_N = 20
EUR = 0.89342


def load_all() -> tuple[dict[str, pd.DataFrame], dict[str, bool]]:
    info = json.loads((lab.DATA / "universe_all.json").read_text())
    data = {}
    for coin in info:
        df = lab.load(coin, "1d")
        if df is not None and len(df) >= 60:
            data[coin] = df
    return data, info


def point_in_time(data: dict[str, pd.DataFrame], top_n: int = TOP_N) -> dict[str, pd.Series]:
    """Elegibilidad diaria: top_n por mediana de volumen en USD de los 30 días previos."""
    dv = pd.DataFrame({c: (df["close"] * df["volume"]) for c, df in data.items()})
    dv = dv[dv.index >= HL_START - pd.Timedelta(days=40)]
    liq = dv.rolling(30, min_periods=20).median().shift(1)  # solo información pasada
    rank = liq.rank(axis=1, ascending=False)
    elig = rank <= top_n
    return {c: elig[c] for c in elig.columns}


def run(data, name_or_fn, elig=None, coins=None, since=HL_START) -> list[lab.Trade]:
    fn = STRATEGIES[name_or_fn] if isinstance(name_or_fn, str) else name_or_fn
    trades: list[lab.Trade] = []
    for coin, df in data.items():
        if coins is not None and coin not in coins:
            continue
        if len(df) < 280:
            continue
        mask = None
        if elig is not None:
            mask = elig[coin].reindex(df.index).fillna(False).to_numpy() if coin in elig else np.zeros(len(df), bool)
        trades += lab.simulate(coin, df, fn(df), "1d", mask)
    return [t for t in trades if since is None or t.t_in >= since]


def line(label: str, trades: list[lab.Trade], risk: float = 0.01) -> str:
    rs = np.array([t.r for t in trades])
    st = lab.stats(rs)
    p = lab.portfolio(trades, risk=risk, max_pos=3, start=224.0)
    return (f"{label:44s} n={st['n']:4d} exp={st['exp']:+.3f}R t={st['t']:+.2f} gan={st['win']:4.1f}% pf={st['pf']:.2f} "
            f"| anual {p['cagr']:+6.1f}% DD {p['maxdd']:4.1f}% ({p.get('years', 0):.1f} a)")


def bootstrap(rs: np.ndarray, n: int = 20000, seed: int = 11) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(rs, size=(n, len(rs)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float((means <= 0).mean() * 100)


def walk_forward(data, elig, names: list[str]) -> tuple[list[lab.Trade], list[str]]:
    per = {n: run(data, n, elig) for n in names}
    log, chosen_trades = [], []
    start = HL_START + pd.DateOffset(months=12)
    end = max(t.t_out for ts in per.values() for t in ts)
    while start < end:
        nxt = start + pd.DateOffset(months=6)
        scores = {}
        for n, ts in per.items():
            past = np.array([t.r for t in ts if start - pd.DateOffset(months=12) <= t.t_in < start and t.t_out < start])
            if len(past) >= 15:
                scores[n] = past.mean()
        picks = [n for n, v in sorted(scores.items(), key=lambda kv: -kv[1]) if v > 0][:3]
        window = [t for n in picks for t in per[n] if start <= t.t_in < nxt]
        rs = np.array([t.r for t in window])
        log.append(f"   {start.date()} -> {nxt.date()}: elige {picks or 'NINGUNA (no opera)'} | "
                   f"{len(rs)} ops, exp {rs.mean() if len(rs) else float('nan'):+.3f}R")
        chosen_trades += window
        start = nxt
    return chosen_trades, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-funding", action="store_true")
    args = ap.parse_args()
    data, info = load_all()
    delisted = [c for c in data if info.get(c)]
    print(f"Mercados con histórico: {len(data)} ({len(delisted)} deslistados)")
    elig = point_in_time(data)
    ever = sorted(c for c, sr in elig.items() if sr.any())
    print(f"Mercados que alguna vez estuvieron en el top {TOP_N}: {len(ever)} "
          f"(de ellos deslistados: {sum(1 for c in ever if info.get(c))})")
    current = set(json.loads((lab.DATA / "universe.json").read_text()))

    if args.fetch_funding:
        import funding
        need = run(data, "ensemble_diario", elig) + [t for n in ENSEMBLE for t in run(data, n, elig)]
        need += run(data, "ensemble_diario", coins=current)
        funding.fetch_for_trades(need)
        return

    n_f = lab.load_funding()
    print(f"Funding real cargado para {n_f} mercados\n")

    print("=== 1. Sesgo de supervivencia y era operable (conjunto diario, funding fijo) ===")
    lab.USE_REAL_FUNDING = False
    print(line("A. Top 20 de HOY, todo el histórico (antes)", run(data, "ensemble_diario", coins=current, since=None)))
    print(line("B. Top 20 de HOY, solo era Hyperliquid", run(data, "ensemble_diario", coins=current)))
    pit_flat = run(data, "ensemble_diario", elig)
    print(line("C. Top 20 punto-en-el-tiempo + deslistados", pit_flat))
    print(line("D. Todos los mercados sin filtro de liquidez", run(data, "ensemble_diario")))

    print("\n=== 2. Funding real frente al supuesto fijo (universo C) ===")
    lab.USE_REAL_FUNDING = True
    pit = run(data, "ensemble_diario", elig)
    covered = [t for t in pit if lab.real_funding(t.coin, t.t_in, t.t_out) is not None]
    print(f"Operaciones con funding real: {len(covered)} de {len(pit)}")
    print(line("C con funding fijo (0,00125 %/h en contra)", pit_flat))
    print(line("C con funding REAL", pit))
    if covered:
        fr = np.array([t.fund_r for t in covered])
        longs = np.array([t.fund_r for t in covered if t.side > 0])
        shorts = np.array([t.fund_r for t in covered if t.side < 0])
        print(f"   Coste medio de funding por operación: {fr.mean():+.3f}R (largos {longs.mean():+.3f}R, cortos {shorts.mean():+.3f}R; "
              f"negativo = cobrado). Supuesto fijo anterior: {np.mean([24 * lab.FUNDING_HOURLY * t.bars / t.stop_pct for t in covered]):+.3f}R")

    print("\n=== 3. Por componentes, largos/cortos y por año (universo C, funding real) ===")
    for n in ENSEMBLE:
        print(line(f"   {n}", run(data, n, elig)))
    print(line("   solo LARGOS", [t for t in pit if t.side > 0]))
    print(line("   solo CORTOS", [t for t in pit if t.side < 0]))
    for y in sorted({t.t_in.year for t in pit}):
        rs = np.array([t.r for t in pit if t.t_in.year == y])
        print(f"   {y}: {len(rs):3d} ops, exp {rs.mean():+.3f}R, suma {rs.sum():+.1f}R")

    print("\n=== 4. Ventanas sucesivas (walk-forward): elegir con 12 meses, operar los 6 siguientes ===")
    wf, log = walk_forward(data, elig, [n for n in STRATEGIES if n != "ensemble_diario"])
    print("\n".join(log))
    wf_start = HL_START + pd.DateOffset(months=12)
    print(line("Walk-forward (estrategias elegidas)", wf))
    print(line("Conjunto fijo en el mismo periodo", [t for t in pit if t.t_in >= wf_start]))

    print("\n=== 5. Robustez de parámetros: expectativa (R) de Donchian con tendencia, universo C ===")
    trails = (2.0, 3.0, 4.0, 5.0)
    print("   canal\\trailing " + "".join(f"{t:>8.0f}ATR" for t in trails))
    for n in (10, 20, 30, 40, 55, 80, 100):
        cells = []
        for tr in trails:
            rs = np.array([t.r for t in run(data, lambda df, n=n, tr=tr: s_donchian(df, n, trail=tr), elig)])
            cells.append(f"{rs.mean():+.3f}({len(rs):3d})")
        print(f"   {n:>5d} días      " + " ".join(f"{c:>11s}" for c in cells))

    print("\n=== 6. Significación estadística (universo C, funding real) ===")
    rs = np.array([t.r for t in pit])
    lo, hi, p0 = bootstrap(rs)
    print(f"   Expectativa {rs.mean():+.3f}R; intervalo 95 % [{lo:+.3f}, {hi:+.3f}]; probabilidad de que sea <= 0: {p0:.1f} %")
    top = np.sort(rs)[::-1]
    print(f"   Dependencia de pocas operaciones: sin las 5 mejores la expectativa es {np.sort(rs)[:-5].mean():+.3f}R; "
          f"las 10 mejores aportan {top[:10].sum():.0f}R de {rs.sum():.0f}R totales")

    print("\n=== 7. Cuenta de 200 EUR con el sistema validado (universo C, funding real) ===")
    print(f"   {'riesgo':>6} {'final €':>8} {'anual%':>7} {'maxDD%':>7} {'mes medio%':>10} {'mes mediano%':>12} {'peor mes%':>9} {'meses +%':>8}")
    for risk in (0.01, 0.02, 0.03):
        p = lab.portfolio(pit, risk=risk, max_pos=3, start=200 / EUR)
        m = monthly_returns(p["curve"])
        print(f"   {risk:>6.0%} {p['final'] * EUR:>8.0f} {p['cagr']:>7.1f} {p['maxdd']:>7.1f} {m.mean():>10.2f} {m.median():>12.2f} "
              f"{m.min():>9.1f} {(m > 0).mean() * 100:>8.0f}")


if __name__ == "__main__":
    main()
