"""Laboratorio de estrategias de HyperAegis.

Prueba varias familias de estrategia sobre histórico real de Hyperliquid
(1h / 4h / 1d; cripto, oro, índices, acciones) con un motor común:

- Señal al cierre de la vela i -> entrada en la apertura de i+1 (sin look-ahead).
- Stop inicial en k x ATR; el resultado de cada operación se mide en R
  (múltiplos del riesgo inicial), neto de costes.
- Costes: comisión taker + deslizamiento por lado, y funding por hora en contra.
- Ejecución intrabarra pesimista (si una vela toca stop y objetivo, gana el stop).
- Validación fuera de muestra: el primer 60 % del tiempo es "entrenamiento"
  (IS) y el último 40 % "validación" (OOS). Una estrategia solo cuenta si
  es positiva en AMBOS tramos.

Uso:
    python research/download.py      # una vez (cachea 12 h)
    python research/lab.py           # tabla comparativa + Monte Carlo
"""

from __future__ import annotations

import heapq
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "state" / "research_data"

FEE_SIDE = 0.00045        # taker Hyperliquid (tier base)
SLIP_SIDE = 0.0003        # deslizamiento medio supuesto por ejecución
FUNDING_HOURLY = 0.0000125  # 0.00125 %/h (tasa base), contado siempre en contra
HOURS = {"1h": 1, "4h": 4, "1d": 24}

# Funding real (research/funding.py): por activo, serie acumulada de la tasa
# horaria y tramos [inicio, fin] descargados. Si una operación cae fuera de un
# tramo descargado se usa la tasa fija FUNDING_HOURLY.
USE_REAL_FUNDING = False
_FUND: dict[str, tuple[pd.Series, list[tuple[pd.Timestamp, pd.Timestamp]]]] = {}


def load_funding() -> int:
    """Carga en memoria el funding descargado. Devuelve nº de activos con datos."""
    _FUND.clear()
    d = DATA / "funding"
    if not d.exists():
        return 0
    for f in d.glob("*.json"):
        blob = json.loads(f.read_text())
        if not blob["rows"]:
            continue
        rows = sorted(set(map(tuple, blob["rows"])))
        ser = pd.Series([r[1] for r in rows], index=pd.to_datetime([r[0] for r in rows], unit="ms", utc=True)).cumsum()
        spans = [(pd.Timestamp(a, unit="ms", tz="UTC"), pd.Timestamp(b, unit="ms", tz="UTC")) for a, b in blob["spans"]]
        _FUND[f.stem] = (ser, spans)
    return len(_FUND)


def real_funding(coin: str, t_in: pd.Timestamp, t_out: pd.Timestamp) -> float | None:
    """Suma de tasas de funding entre t_in y t_out (positivo = pagan los largos)."""
    item = _FUND.get(coin.replace(":", "_"))
    if item is None:
        return None
    ser, spans = item
    if not any(a <= t_in and t_out <= b for a, b in spans):
        return None
    lo = ser.asof(t_in)
    hi = ser.asof(t_out)
    return float((0.0 if pd.isna(hi) else hi) - (0.0 if pd.isna(lo) else lo))


# ---------------------------------------------------------------------- #
# Datos e indicadores
# ---------------------------------------------------------------------- #
def load(coin: str, iv: str) -> pd.DataFrame | None:
    f = DATA / f"{coin.replace(':', '_')}_{iv}.json"
    if not f.exists():
        return None
    raw = json.loads(f.read_text())
    if len(raw) < 150:
        return None
    df = pd.DataFrame(raw)
    df["time"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return df.set_index("time")[["open", "high", "low", "close", "volume"]].iloc[:-1]


from quant.strategies import STRATEGIES, atr  # noqa: E402


# ---------------------------------------------------------------------- #
# Motor de operaciones (una posición por activo)
# ---------------------------------------------------------------------- #
@dataclass
class Trade:
    coin: str
    t_in: pd.Timestamp
    t_out: pd.Timestamp
    side: int
    r: float          # resultado neto en R
    stop_pct: float   # distancia del stop inicial / precio (para apalancamiento)
    bars: int
    fund_r: float = 0.0  # coste de funding en R (negativo = cobrado)


def simulate(coin: str, df: pd.DataFrame, spec: dict, iv: str, elig: np.ndarray | None = None) -> list[Trade]:
    """`elig[i]` (opcional): solo se aceptan señales en velas donde el activo es elegible."""
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    a = atr(df).to_numpy()
    sig = np.asarray(spec["sig"])
    times = df.index
    mode, stop_k = spec["exit"], spec["stop_k"]
    trail_k = spec.get("trail_k", 3.0)
    max_bars = spec.get("max_bars")
    ex_l, ex_s = spec.get("exit_long"), spec.get("exit_short")
    fund_bar = FUNDING_HOURLY * HOURS[iv]
    n = len(c)
    trades: list[Trade] = []
    i = 210
    while i < n - 1:
        s = int(sig[i])
        if elig is not None and not elig[i]:
            s = 0
        if s == 0 or not np.isfinite(a[i]) or a[i] <= 0:
            i += 1
            continue
        e_i = i + 1
        entry = o[e_i]
        r = stop_k * a[i]
        stop = entry - s * r
        ext = entry
        size, realized, tp_done = 1.0, 0.0, False
        tp = entry + s * 1.5 * r
        j = e_i
        while True:
            adverse = l[j] if s > 0 else h[j]
            fav = h[j] if s > 0 else l[j]
            if (adverse - stop) * s <= 0:
                px = o[j] if (o[j] - stop) * s < 0 else stop
                realized += size * (px - entry) * s
                break
            if mode == "tp_partial" and not tp_done and (fav - tp) * s >= 0:
                realized += 0.5 * (tp - entry) * s
                size, tp_done, stop = 0.5, True, entry
                if (c[j] - stop) * s <= 0:  # pesimista: vuelve a breakeven en la misma vela
                    break
            ext = max(ext, h[j]) if s > 0 else min(ext, l[j])
            if mode == "chandelier" or (mode == "tp_partial" and tp_done):
                cand = ext - s * trail_k * a[j]
                if (cand - stop) * s > 0:
                    stop = cand
            flag = ex_l if s > 0 else ex_s
            timed_out = max_bars is not None and j - e_i + 1 >= max_bars
            if (flag is not None and flag[j]) or timed_out or j >= n - 1:
                px = o[j + 1] if j + 1 < n else c[j]
                realized += size * (px - entry) * s
                j = min(j + 1, n - 1)
                break
            j += 1
        bars = j - e_i + 1
        fund = fund_bar * bars  # supuesto fijo, siempre en contra
        if USE_REAL_FUNDING:
            rf = real_funding(coin, times[e_i], times[j])
            if rf is not None:
                fund = s * rf  # el largo paga si la tasa es positiva; el corto la cobra
        cost = entry * (2 * (FEE_SIDE + SLIP_SIDE) + fund)
        trades.append(Trade(coin, times[e_i], times[j], s, (realized - cost) / r, r / entry, bars, entry * fund / r))
        i = j + 1
    return trades


# ---------------------------------------------------------------------- #
# Métricas y cartera
# ---------------------------------------------------------------------- #
def stats(rs: np.ndarray) -> dict:
    if len(rs) == 0:
        return dict(n=0, exp=float("nan"), win=float("nan"), pf=float("nan"), t=float("nan"))
    wins, losses = rs[rs > 0].sum(), -rs[rs <= 0].sum()
    t = rs.mean() / (rs.std(ddof=1) / math.sqrt(len(rs))) if len(rs) > 2 and rs.std() > 0 else float("nan")
    return dict(n=len(rs), exp=rs.mean(), win=(rs > 0).mean() * 100, pf=wins / losses if losses else float("inf"), t=t)


def portfolio(trades: list[Trade], risk: float = 0.01, max_pos: int = 5, lev_cap: float = 5.0,
              start: float = 500.0) -> dict:
    """Cartera con riesgo fijo fraccional sobre el equity ACTUAL (interés compuesto)."""
    if not trades:
        return dict(final=start, cagr=0.0, maxdd=0.0, taken=0, curve=[])
    trades = sorted(trades, key=lambda t: t.t_in)
    eq, peak, maxdd, taken = start, start, 0.0, 0
    open_heap: list[tuple[pd.Timestamp, int, float]] = []
    busy: set[str] = set()
    coin_of: dict[int, str] = {}
    curve: list[tuple[pd.Timestamp, float]] = [(trades[0].t_in, start)]

    def settle(until: pd.Timestamp | None) -> None:
        nonlocal eq, peak, maxdd
        while open_heap and (until is None or open_heap[0][0] <= until):
            t_out, k, pnl = heapq.heappop(open_heap)
            eq += pnl
            curve.append((t_out, eq))
            busy.discard(coin_of.pop(k))
            peak = max(peak, eq)
            maxdd = max(maxdd, (peak - eq) / peak)

    for k, t in enumerate(trades):
        settle(t.t_in)
        if eq <= 0:
            break
        if len(open_heap) >= max_pos or t.coin in busy:
            continue
        eff_risk = min(risk, lev_cap / max_pos * t.stop_pct)  # tope de apalancamiento por posición
        if eq * eff_risk / t.stop_pct < 10.0:  # nocional mínimo de orden en Hyperliquid (10 USD)
            continue
        heapq.heappush(open_heap, (t.t_out, k, eq * eff_risk * t.r))
        busy.add(t.coin)
        coin_of[k] = t.coin
        taken += 1
    settle(None)
    years = max((trades[-1].t_out - trades[0].t_in).total_seconds() / 31_557_600, 1e-9)
    cagr = (max(eq, 1e-9) / start) ** (1 / years) - 1
    return dict(final=eq, cagr=cagr * 100, maxdd=maxdd * 100, taken=taken, years=years, curve=curve)


def evaluate(universe: list[str], iv: str, name: str) -> dict | None:
    fn = STRATEGIES[name]
    trades: list[Trade] = []
    t0, t1 = None, None
    for coin in universe:
        df = load(coin, iv)
        if df is None or len(df) < 400:
            continue
        t0 = df.index[0] if t0 is None else min(t0, df.index[0])
        t1 = df.index[-1] if t1 is None else max(t1, df.index[-1])
        trades += simulate(coin, df, fn(df), iv)
    if not trades or t0 is None:
        return None
    first = min(t.t_in for t in trades)
    split = first + (t1 - first) * 0.6
    is_r = np.array([t.r for t in trades if t.t_in < split])
    oos_tr = [t for t in trades if t.t_in >= split]
    oos_r = np.array([t.r for t in oos_tr])
    return dict(name=name, iv=iv, trades=trades, oos_trades=oos_tr, split=split,
                all=stats(np.array([t.r for t in trades])), IS=stats(is_r), OOS=stats(oos_r),
                port=portfolio(trades), port_oos=portfolio(oos_tr))


# ---------------------------------------------------------------------- #
# Monte Carlo: ¿cuánto se tarda de 500 a 1.000.000?
# ---------------------------------------------------------------------- #
def monte_carlo(rs: np.ndarray, trades_per_year: float, risk: float, years: float = 10,
                start: float = 500, target: float = 1_000_000, sims: int = 4000, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    n = int(trades_per_year * years)
    draws = rng.choice(rs, size=(sims, n), replace=True)
    growth = np.clip(1 + risk * draws, 1e-6, None)
    eq = start * np.cumprod(growth, axis=1)
    run_max = np.maximum.accumulate(eq, axis=1)
    maxdd = ((run_max - eq) / run_max).max(axis=1)
    ruined = (eq.min(axis=1) < start * 0.1)
    hit = eq >= target
    reached = hit.any(axis=1) & ~ruined
    first = np.where(hit.any(axis=1), hit.argmax(axis=1), n) / trades_per_year
    return dict(risk=risk, p_target=reached.mean() * 100, p_ruin=ruined.mean() * 100,
                median_final=float(np.median(eq[:, -1])), median_dd=float(np.median(maxdd) * 100),
                median_years=float(np.median(first[reached])) if reached.any() else float("nan"))


def main() -> None:
    universe = json.loads((DATA / "universe.json").read_text())
    crypto = [c for c in universe if ":" not in c and c != "PAXG"]
    trad = [c for c in universe if ":" in c or c == "PAXG"]
    print(f"Universo: {len(crypto)} cripto, {len(trad)} tradicionales ({', '.join(trad)})")
    rows, results = [], {}
    for group, coins in (("cripto", crypto), ("trad", trad)):
        for iv in ("1h", "4h", "1d"):
            for name in STRATEGIES:
                res = evaluate(coins, iv, name)
                if res is None or res["all"]["n"] < 20:
                    continue
                results[(group, iv, name)] = res
                rows.append(dict(
                    grupo=group, tf=iv, estrategia=name, n=res["all"]["n"],
                    exp_IS=res["IS"]["exp"], exp_OOS=res["OOS"]["exp"], n_OOS=res["OOS"]["n"],
                    t_all=res["all"]["t"], win=res["all"]["win"], pf=res["all"]["pf"],
                    cagr_1pct=res["port"]["cagr"], maxdd=res["port"]["maxdd"], años=res["port"]["years"],
                ))
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 220, "display.max_rows", 200, "display.float_format", lambda v: f"{v:,.2f}")
    print("\n=== Todas las combinaciones (expectativa en R por operación, neta de costes) ===")
    print(table.sort_values(["grupo", "tf", "exp_OOS"], ascending=[True, True, False]).to_string(index=False))
    robust = table[(table.exp_IS > 0) & (table.exp_OOS > 0) & (table.n_OOS >= 30)].sort_values("t_all", ascending=False)
    print("\n=== Robustas: positivas dentro y fuera de muestra (n_OOS >= 30) ===")
    print(robust.to_string(index=False) if len(robust) else "  NINGUNA")
    out = ROOT / "state" / "research_results.json"
    out.write_text(table.to_json(orient="records", indent=1))

    if len(robust):
        best = robust.iloc[0]
        res = results[(best.grupo, best.tf, best.estrategia)]
        rs = np.array([t.r for t in res["trades"]])
        tpy = res["port"]["taken"] / res["port"]["years"]
        print(f"\n=== Monte Carlo con '{best.estrategia}' {best.tf} {best.grupo} "
              f"(exp {rs.mean():+.3f}R, ~{tpy:.0f} operaciones/año en cartera) ===")
        for risk in (0.01, 0.02, 0.03, 0.05, 0.10):
            m = monte_carlo(rs, tpy, risk)
            print(f"  riesgo {risk:>4.0%}: P(1M en 10 años)={m['p_target']:5.1f}%  P(ruina -90%)={m['p_ruin']:5.1f}%  "
                  f"mediana final={m['median_final']:>14,.0f}  DD mediano={m['median_dd']:4.0f}%  "
                  f"años medianos hasta 1M={m['median_years']:.1f}")


if __name__ == "__main__":
    main()
