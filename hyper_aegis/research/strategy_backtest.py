"""Backtest del sistema `strategy/` (embudo + patrones + confluencia) con el MISMO código del escáner.

Para cada vela cerrada del timeframe de gatillo se reconstruye lo que el escáner habría visto (sin mirar al
futuro: los indicadores son causales y los timeframes superiores usan solo velas ya cerradas en ese instante),
se aplica el embudo, se detectan patrones, se puntúa y, si la puntuación es >= `min_score`, se entra en la
APERTURA de la vela siguiente con el stop y el objetivo de la señal (R:R 1:2 / 1:2,5). Sin trailing: salida por
stop, objetivo o tiempo máximo. Costes: comisión taker 0,045 %/lado + 3 pb de deslizamiento/lado + funding real.

Dos versiones:
  A  exacta:   gatillo 15m, estructura 1h, macro 4h      (~52 días: límite de 5.000 velas de la API)
  B  escalada: gatillo 1h,  estructura 4h, macro 1d      (~200 días)

Limitaciones: sin histórico de Open Interest ni de libro L2 (el OI no puntúa y no hay filtro de spread); activos
= los 20 de mayor volumen de HOY; exposición de cartera = 3 posiciones, riesgo 1 % compuesto, 5x máx.

    python research/strategy_data.py 20          # una vez
    python research/strategy_backtest.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab  # noqa: E402
from strategy.confluence_and_risk import ScoreConfig, evaluate_patterns  # noqa: E402
from strategy.filters_and_candles import MIN_BARS, FilterConfig, detect_patterns, evaluate_funnel, trend_direction  # noqa: E402
from strategy.indicators import add_indicators  # noqa: E402
from strategy.market_data import MarketDataClient, PerpContext  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "state" / "research_data" / "bt"
FEE, SLIP = 0.00045, 0.0003
TF_MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}

VARIANTS = {
    "A": dict(label="A. Exacta: gatillo 15m / estructura 1h / macro 4h", cfg=FilterConfig(), hold=96),
    "B": dict(label="B. Escalada: gatillo 1h / estructura 4h / macro 1d",
              cfg=replace(FilterConfig(), trigger_tf="1h", structure_tf="4h", macro_tf="1d",
                          atr_pct_bounds={"1h": (0.0025, 0.05), "4h": (0.005, 0.10)}), hold=96),
}


@dataclass
class CoinData:
    coin: str
    frames: Dict[str, pd.DataFrame]
    close_ms: Dict[str, np.ndarray]
    fund_t: np.ndarray
    fund_r: np.ndarray
    fund_cum: np.ndarray
    sz: int


@dataclass
class BTrade:
    coin: str
    side: int
    score: float
    pattern: str
    t_sig: pd.Timestamp
    t_in: pd.Timestamp
    t_out: pd.Timestamp
    r: float
    stop_pct: float
    bars: int
    exit_kind: str
    rr: float
    stop_atr: float


def to_ms(idx: pd.Series) -> np.ndarray:
    return ((idx - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1ms")).to_numpy()


def load_coin(coin: str, tfs: Tuple[str, ...], meta: dict, now_ms: int) -> Optional[CoinData]:
    frames, close_ms = {}, {}
    for tf in tfs:
        f = DATA / f"{coin}_{tf}.json"
        if not f.exists():
            return None
        df = MarketDataClient.parse_candles(json.loads(f.read_text()), now_ms)
        df = df[df["closed"]]
        frames[tf] = add_indicators(df)  # causal: la fila i solo usa datos hasta i
        close_ms[tf] = to_ms(frames[tf]["close_time"])
    raw = json.loads((DATA / f"{coin}_funding.json").read_text())
    raw.sort()
    ft = np.array([r[0] for r in raw], dtype="int64")
    fr = np.array([r[1] for r in raw], dtype=float)
    return CoinData(coin, frames, close_ms, ft, fr, np.cumsum(fr), int(meta[coin]["szDecimals"]))


def funding_rate_at(d: CoinData, t_ms: int) -> float:
    k = np.searchsorted(d.fund_t, t_ms, side="right") - 1
    return float(d.fund_r[k]) if k >= 0 else 0.0


def funding_sum(d: CoinData, t0_ms: int, t1_ms: int) -> float:
    a = np.searchsorted(d.fund_t, t0_ms, side="right")
    b = np.searchsorted(d.fund_t, t1_ms, side="right")
    return float(d.fund_cum[b - 1] - (d.fund_cum[a - 1] if a > 0 else 0.0)) if b > a else 0.0


def exit_sim(o, h, l, c, e: int, side: int, stop: float, tp: float, hold: int) -> Tuple[float, int, str]:
    """Salida desde la barra de entrada `e`: stop primero (pesimista), con gap; luego objetivo; luego tiempo."""
    n = len(o)
    for j in range(e, min(e + hold, n)):
        if side > 0:
            if l[j] <= stop:
                return (min(o[j], stop) if j > e else stop), j, "stop"
            if h[j] >= tp:
                return tp, j, "objetivo"
        else:
            if h[j] >= stop:
                return (max(o[j], stop) if j > e else stop), j, "stop"
            if l[j] <= tp:
                return tp, j, "objetivo"
    j = min(e + hold, n) - 1
    return float(c[j]), j, "tiempo"


def trade_r(d: CoinData, tf: str, fill: float, exit_px: float, stop: float, side: int, e: int, j: int) -> float:
    ct = d.close_ms[tf]
    open_ms = int(ct[e]) - TF_MS[tf]
    eff_exit = exit_px * (1 - SLIP * side)
    gross = (eff_exit - fill) * side
    fees = FEE * (fill + eff_exit)
    fund = side * funding_sum(d, open_ms, int(ct[j])) * fill
    risk = abs(fill - stop)
    return (gross - fees - fund) / risk if risk > 0 else 0.0


def backtest_coin(d: CoinData, cfg: FilterConfig, scfg: ScoreConfig, hold: int, since_ms: int) -> List[BTrade]:
    trig, struct, macro = d.frames[cfg.trigger_tf], d.frames[cfg.structure_tf], d.frames[cfg.macro_tf]
    o, h, l, c = (trig[k].to_numpy() for k in ("open", "high", "low", "close"))
    atr = trig["atr14"].to_numpy()
    ct = d.close_ms[cfg.trigger_tf]
    n = len(trig)
    out: List[BTrade] = []
    i = max(MIN_BARS, int(np.searchsorted(ct, since_ms)))
    while i < n - 1:
        t = int(ct[i])
        km = int(np.searchsorted(d.close_ms[cfg.macro_tf], t, side="right"))
        ks = int(np.searchsorted(d.close_ms[cfg.structure_tf], t, side="right"))
        if km < MIN_BARS or ks < MIN_BARS:
            i += 1
            continue
        fr = {cfg.macro_tf: macro.iloc[:km], cfg.structure_tf: struct.iloc[:ks], cfg.trigger_tf: trig.iloc[: i + 1]}
        f = funding_rate_at(d, t)
        ctx = PerpContext(d.coin, c[i], c[i], c[i], c[i], f, None, 0.0, 1.0, 1e9, 20, d.sz, 0.0)
        funnel = evaluate_funnel(d.coin, fr, ctx, None, None, cfg)
        if not funnel.passed:
            i += 1
            continue
        pats = detect_patterns(fr[cfg.trigger_tf], trend_direction(funnel.regimes, cfg), cfg)
        sigs = [s for s in evaluate_patterns(d.coin, pats, funnel, fr[cfg.trigger_tf], d.sz, None, None, cfg, scfg) if s.approved] if pats else []
        if not sigs:
            i += 1
            continue
        s = sigs[0]
        side = 1 if s.direction == "LONG" else -1
        e = i + 1
        fill = o[e] * (1 + SLIP * side)
        if (side > 0 and not s.stop_loss < fill < s.take_profit) or (side < 0 and not s.take_profit < fill < s.stop_loss):
            i += 1  # el hueco de apertura se salta el stop o el objetivo: no se opera
            continue
        exit_px, j, kind = exit_sim(o, h, l, c, e, side, s.stop_loss, s.take_profit, hold)
        r = trade_r(d, cfg.trigger_tf, fill, exit_px, s.stop_loss, side, e, j)
        out.append(BTrade(d.coin, side, s.score, s.pattern, trig.index[i], trig.index[e], trig.index[j], r, abs(fill - s.stop_loss) / fill,
                          j - e + 1, kind, s.rr, abs(fill - s.stop_loss) / atr[i]))
        i = j + 1  # una posición por activo
    return out


def null_expectancy(trades: List[BTrade], data: Dict[str, CoinData], cfg: FilterConfig, hold: int, since_ms: int, draws: int = 30, seed: int = 1) -> Tuple[float, float]:
    """Expectativa de entrar AL AZAR con la misma geometría (distancia de stop en ATR, R:R, dirección) y costes."""
    rng = np.random.default_rng(seed)
    rs: List[float] = []
    for t in trades:
        d = data[t.coin]
        trig = d.frames[cfg.trigger_tf]
        o, h, l, c = (trig[k].to_numpy() for k in ("open", "high", "low", "close"))
        atr = trig["atr14"].to_numpy()
        lo = max(MIN_BARS, int(np.searchsorted(d.close_ms[cfg.trigger_tf], since_ms)))
        for _ in range(draws):
            i = int(rng.integers(lo, len(trig) - 2))
            e = i + 1
            fill = o[e] * (1 + SLIP * t.side)
            dist = t.stop_atr * atr[i]
            stop = fill - t.side * dist
            tp = fill + t.side * t.rr * dist
            ex, j, _ = exit_sim(o, h, l, c, e, t.side, stop, tp, hold)
            rs.append(trade_r(d, cfg.trigger_tf, fill, ex, stop, t.side, e, j))
    a = np.array(rs)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))


def stats_line(label: str, rs: np.ndarray) -> str:
    if len(rs) == 0:
        return f"   {label:28s} n=0"
    st = lab.stats(rs)
    return f"   {label:28s} n={len(rs):4d} exp={rs.mean():+.3f}R t={st['t']:+.2f} gan={st['win']:4.1f}% pf={st['pf']:.2f}"


def portfolio_line(trades: List[BTrade], label: str) -> str:
    lt = [lab.Trade(t.coin, t.t_in, t.t_out, t.side, t.r, t.stop_pct, t.bars) for t in trades]
    if not lt:
        return f"   {label}: sin operaciones"
    p = lab.portfolio(sorted(lt, key=lambda x: x.t_in), risk=0.01, max_pos=3, lev_cap=5.0, start=224.0)
    return f"   {label}: capital 224 -> {p['final']:.0f} USDC ({(p['final'] / 224 - 1) * 100:+.1f} %), caída máx. {p['maxdd']:.1f} %, {p['taken']} de {len(lt)} operaciones tomadas"


def run_variant(key: str, data: Dict[str, CoinData], min_score: float = 6.5, since_ms: int = 0) -> Tuple[List[BTrade], FilterConfig, int]:
    v = VARIANTS[key]
    scfg = replace(ScoreConfig(), min_score=min_score)
    trades: List[BTrade] = []
    for d in data.values():
        trades += backtest_coin(d, v["cfg"], scfg, v["hold"], since_ms)
    return sorted(trades, key=lambda t: t.t_in), v["cfg"], v["hold"]


def report(key: str, data: Dict[str, CoinData]) -> None:
    v = VARIANTS[key]
    cfg = v["cfg"]
    d0 = next(iter(data.values()))
    t0 = d0.frames[cfg.trigger_tf].index[MIN_BARS]
    print(f"\n{'=' * 100}\n{v['label']}\nPeriodo evaluado: {t0.date()} -> {d0.frames[cfg.trigger_tf].index[-1].date()} | {len(data)} activos | salida: stop / objetivo / {v['hold']} velas\n{'=' * 100}")
    t_start = time.time()
    trades, cfg, hold = run_variant(key, data)
    print(f"(simulado en {time.time() - t_start:.0f} s)")
    if not trades:
        print("Sin operaciones.")
        return
    rs = np.array([t.r for t in trades])
    print("\n--- Resultado global (puntuación >= 6,5) ---")
    print(stats_line("Todas", rs))
    print(portfolio_line(trades, "Cartera (1 % riesgo, 3 pos.)"))
    se = rs.std(ddof=1) / np.sqrt(len(rs))
    print(f"   Expectativa {rs.mean():+.3f}R, intervalo 95 % [{rs.mean() - 1.96 * se:+.3f}, {rs.mean() + 1.96 * se:+.3f}]")
    nm, nse = null_expectancy(trades, data, cfg, hold, int(trades[0].t_in.timestamp() * 1000) - 10 ** 9)
    print(f"   AZAR (misma geometría de stop/objetivo y costes): {nm:+.3f}R ± {1.96 * nse:.3f} -> ventaja de la señal {rs.mean() - nm:+.3f}R")
    kinds = {k: sum(1 for t in trades if t.exit_kind == k) for k in ("objetivo", "stop", "tiempo")}
    print(f"   Salidas: {kinds} | duración media {np.mean([t.bars for t in trades]):.0f} velas | stop medio {np.mean([t.stop_atr for t in trades]):.2f} ATR ({np.mean([t.stop_pct for t in trades]) * 100:.2f} %)")

    print("\n--- ¿Predice la puntuación? ---")
    for lo, hi in ((6.5, 7.5), (7.5, 8.5), (8.5, 10.1)):
        print(stats_line(f"puntuación {lo:g}-{min(hi, 10):g}", np.array([t.r for t in trades if lo <= t.score < hi])))
    print("--- Por patrón ---")
    for p in sorted({t.pattern for t in trades}):
        print(stats_line(p, np.array([t.r for t in trades if t.pattern == p])))
    print("--- Por dirección ---")
    for side, name in ((1, "LONG"), (-1, "SHORT")):
        print(stats_line(name, np.array([t.r for t in trades if t.side == side])))
    print("--- Por mitad del periodo (estabilidad) ---")
    mid = trades[len(trades) // 2].t_in
    print(stats_line("primera mitad", np.array([t.r for t in trades if t.t_in < mid])))
    print(stats_line("segunda mitad", np.array([t.r for t in trades if t.t_in >= mid])))
    print("--- Sensibilidad al umbral de puntuación ---")
    for ms in (5.0, 5.5, 6.5, 7.5):
        tr, _, _ = run_variant(key, data, ms)
        print(stats_line(f"umbral {ms:g}", np.array([t.r for t in tr])))
    print("--- Sensibilidad a los costes ---")
    global FEE, SLIP
    keep = (FEE, SLIP)
    for label, (fee, slip) in (("comisión maker 0,015 %", (0.00015, SLIP)), ("sin costes", (0.0, 0.0)), ("costes x2", (FEE * 2, SLIP * 2))):
        FEE, SLIP = fee, slip
        tr, _, _ = run_variant(key, data)
        print(stats_line(label, np.array([t.r for t in tr])))
    FEE, SLIP = keep


def main() -> None:
    which = sys.argv[1].upper() if len(sys.argv) > 1 else "AB"
    meta = json.loads((DATA / "meta.json").read_text())
    coins = json.loads((DATA / "coins.json").read_text())
    now_ms = int(time.time() * 1000)
    for key in which:
        cfg = VARIANTS[key]["cfg"]
        tfs = (cfg.macro_tf, cfg.structure_tf, cfg.trigger_tf)
        data = {c: d for c in coins if (d := load_coin(c, tfs, meta, now_ms)) is not None}
        report(key, data)


if __name__ == "__main__":
    main()
