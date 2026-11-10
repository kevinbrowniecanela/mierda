"""Backtest de HyperAegis sobre velas históricas reales de Hyperliquid.

Qué se prueba:
    Fase 1 (AnomalyTrigger) + RiskManager (tamaño 1 %, House Money, máx. 3
    posiciones, circuit breaker 3 %) + salidas (TP1 1.5R al 50 % -> breakeven
    -> trailing 2x ATR), con comisiones taker y deslizamiento.

Limitaciones (importantes al interpretar resultados):
    - Claude NO participa: reproducir sus decisiones históricas no es posible
      sin llamarlo barra a barra. La dirección se toma del sesgo cuantitativo
      de la Fase 1 (`--direction bias`) y el stop por defecto es 1.5 ATR.
      Mide, por tanto, la calidad del filtro + gestión de riesgo, que es la
      base sobre la que Claude filtra.
    - Solo hay histórico de velas: las señales de Open Interest, funding y
      libro L2 no se pueden reconstruir y quedan fuera.
    - No se modela el funding.
    - La watchlist es la ACTUAL (sesgo de supervivencia).
    - Ejecución intrabarra conservadora: si en la misma vela se tocan stop y
      TP1, se asume que el stop saltó primero.

Uso:
    python backtest.py                       # top 20 volumen + 5 movers, 15m, máx. histórico
    python backtest.py --days 30 --threshold 3.5
    python backtest.py --coins BTC ETH SOL
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from config.settings import Settings, get_settings  # noqa: E402
from data.derivatives_data import DerivativesData  # noqa: E402
from data.market_feed import MarketFeed  # noqa: E402
from execution.risk_manager import RiskManager, round_price, round_size  # noqa: E402
from quant.anomaly_trigger import AnomalyTrigger  # noqa: E402
from quant.technicals import add_indicators, snapshot_at  # noqa: E402

SLIPPAGE = 0.0005  # 5 bps por ejecución a mercado


@dataclass
class BTPosition:
    coin: str
    is_long: bool
    size_initial: float
    size: float
    entry: float
    stop: float
    tp1: float
    atr: float
    sz_dec: int
    opened: pd.Timestamp
    r_usd: float
    score: float
    tp1_hit: bool = False
    extreme: float = 0.0
    pnl: float = 0.0  # neto acumulado (incluye comisiones)
    fees: float = 0.0


@dataclass
class BTTrade:
    coin: str
    side: str
    opened: str
    closed: str
    entry: float
    exit: float
    r_usd: float
    pnl: float
    r_multiple: float
    tp1_hit: bool
    reason: str
    bars: int
    score: float


@dataclass
class BTResult:
    trades: list[BTTrade] = field(default_factory=list)
    equity: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    escalations: int = 0
    anomalies: int = 0
    blocked: Counter = field(default_factory=Counter)
    circuit_breakers: int = 0
    house_money_at: str | None = None


def side_mult(is_long: bool) -> int:
    return 1 if is_long else -1


async def load_data(s: Settings, coins: list[str] | None, interval: str, cache: Path) -> tuple[dict[str, pd.DataFrame], dict[str, tuple[int, int]]]:
    feed = MarketFeed(s)
    deriv = DerivativesData(s, feed)
    universe, _ = await feed.meta_and_ctxs()
    meta = {a["name"]: (int(a["szDecimals"]), int(a.get("maxLeverage", 1))) for a in universe}
    if not coins:
        coins = await deriv.build_watchlist()
    out: dict[str, pd.DataFrame] = {}
    for coin in coins:
        f = cache / f"{coin}_{interval}.parquet.json"
        if f.exists() and time.time() - f.stat().st_mtime < 6 * 3600:
            df = pd.read_json(f, orient="split")
            df.index = pd.to_datetime(df.index, utc=True)
        else:
            df = await feed.candles(coin, interval, 5000)
            df.to_json(f, orient="split", date_format="iso")
            await asyncio.sleep(6)  # ~100 de peso por petición; límite 1200/min
        if len(df) > 300:
            out[coin] = df.iloc[:-1]  # descarta la vela en formación
        print(f"  {coin:10s} {len(df):5d} velas  {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
    return out, meta


def run_backtest(s: Settings, data: dict[str, pd.DataFrame], meta: dict[str, tuple[int, int]],
                 days: float | None, direction: str) -> BTResult:
    res = BTResult()
    ind = {c: add_indicators(df) for c, df in data.items()}
    pos_index = {c: {ts: i for i, ts in enumerate(df.index)} for c, df in ind.items()}
    timeline = sorted(set().union(*[set(df.index) for df in ind.values()]))
    start_ts = timeline[0] + pd.Timedelta(minutes=15 * 210)
    if days:
        start_ts = max(start_ts, timeline[-1] - pd.Timedelta(days=days))
    timeline = [t for t in timeline if t >= start_ts]

    risk = RiskManager(s)
    trig = AnomalyTrigger(s)
    cash = s.initial_capital
    positions: dict[str, BTPosition] = {}
    pending: list[tuple[str, bool, float]] = []  # entradas para la apertura siguiente
    last_close: dict[str, float] = {}

    def open_risk() -> float:
        return sum(max(0.0, (p.entry - p.stop) * side_mult(p.is_long) * p.size) for p in positions.values())

    def equity_now() -> float:
        return cash + sum((last_close.get(p.coin, p.entry) - p.entry) * side_mult(p.is_long) * p.size for p in positions.values())

    def close_part(p: BTPosition, qty: float, px: float, when: pd.Timestamp, reason: str, slip: bool = True) -> None:
        nonlocal cash
        fill = px * (1 - SLIPPAGE * side_mult(p.is_long)) if slip else px
        fee = fill * qty * s.taker_fee_rate
        pnl = (fill - p.entry) * side_mult(p.is_long) * qty - fee
        cash += pnl
        p.pnl += pnl
        p.fees += fee
        p.size = round_size(p.size - qty, p.sz_dec)
        if p.size <= 0:
            positions.pop(p.coin, None)
            bars = int((when - p.opened) / pd.Timedelta(minutes=15))
            res.trades.append(BTTrade(
                coin=p.coin, side="LONG" if p.is_long else "SHORT", opened=str(p.opened), closed=str(when),
                entry=p.entry, exit=fill, r_usd=p.r_usd, pnl=p.pnl, r_multiple=p.pnl / p.r_usd if p.r_usd else 0,
                tp1_hit=p.tp1_hit, reason=reason, bars=bars, score=p.score,
            ))

    for t in timeline:
        now_dt = t.to_pydatetime()
        # 1) Entradas pendientes en la apertura de esta vela
        for coin, is_long, score in pending:
            i = pos_index[coin].get(t)
            if i is None or coin in positions:
                continue
            row = ind[coin].iloc[i]
            atr = float(ind[coin]["atr"].iat[i - 1])
            ok, why = risk.can_open(len(positions), equity_now(), open_risk(), now_dt)
            if not ok:
                res.blocked[why.split(":")[0]] += 1
                continue
            sz_dec, max_lev = meta.get(coin, (2, 10))
            ref = float(row["open"])
            plan, why = risk.build_plan(coin=coin, is_long=is_long, price=ref, atr=atr, proposed_stop=None,
                                        sz_decimals=sz_dec, equity=equity_now(), open_risk=open_risk(),
                                        open_count=len(positions), max_asset_lev=max_lev)
            if plan is None:
                res.blocked[why.split("(")[0].strip()] += 1
                continue
            fill = ref * (1 + SLIPPAGE * side_mult(is_long))
            fee = fill * plan.size * s.taker_fee_rate
            cash -= fee
            r = abs(ref - plan.stop)
            stop = round_price(fill - r * side_mult(is_long), sz_dec)
            tp1 = round_price(fill + r * s.tp1_rr * side_mult(is_long), sz_dec)
            positions[coin] = BTPosition(coin, is_long, plan.size, plan.size, fill, stop, tp1, atr, sz_dec, t,
                                         r_usd=r * plan.size, score=score, extreme=fill, pnl=-fee, fees=fee)
        pending = []

        # 2) Gestión intrabarra de posiciones abiertas
        for coin in list(positions):
            i = pos_index[coin].get(t)
            if i is None:
                continue
            p = positions[coin]
            o, h, l, c = (float(ind[coin][k].iat[i]) for k in ("open", "high", "low", "close"))
            lm = side_mult(p.is_long)
            adverse = l if p.is_long else h
            favorable = h if p.is_long else l
            stop_hit = (adverse - p.stop) * lm <= 0
            if stop_hit:
                gap_px = o if (o - p.stop) * lm < 0 else p.stop  # gap a través del stop
                reason = "stop_loss" if not p.tp1_hit else ("breakeven" if abs(p.stop - p.entry) < 1e-12 else "trailing_stop")
                close_part(p, p.size, gap_px, t, reason)
                continue
            if not p.tp1_hit and (favorable - p.tp1) * lm >= 0:
                qty = round_size(p.size * s.tp1_close_fraction, p.sz_dec)
                if qty > 0:
                    close_part(p, qty, p.tp1, t, "tp1")
                p.tp1_hit = True
                p.stop = round_price(p.entry, p.sz_dec)
                # pesimista: si la misma vela vuelve a breakeven, se cierra
                if coin in positions and (adverse - p.stop) * lm <= 0 and (c - p.stop) * lm <= 0:
                    close_part(p, p.size, p.stop, t, "breakeven")
                    continue
            # trailing 2x ATR (se aplica desde la vela siguiente)
            atr_now = float(ind[coin]["atr"].iat[i])
            p.atr = atr_now
            p.extreme = max(p.extreme, h) if p.is_long else min(p.extreme, l)
            if p.tp1_hit:
                cand = round_price(p.extreme - lm * s.trailing_atr_mult * atr_now, p.sz_dec)
                if (cand - p.stop) * lm > 0:
                    p.stop = cand

        # 3) Marcar a mercado, Capital Shield y circuit breaker
        for coin, df in ind.items():
            i = pos_index[coin].get(t)
            if i is not None:
                last_close[coin] = float(df["close"].iat[i])
        eq = equity_now()
        events = risk.update_equity(eq, now_dt)
        if "house_money_on" in events and not res.house_money_at:
            res.house_money_at = str(t)
        if "circuit_breaker" in events or "shield_breached" in events:
            res.circuit_breakers += "circuit_breaker" in events
            if s.flatten_on_circuit_breaker or "shield_breached" in events:
                for coin in list(positions):
                    close_part(positions[coin], positions[coin].size, last_close[coin], t, "circuit_breaker")
        res.equity.append((t, equity_now()))

        # 4) Fase 1 sobre la vela recién cerrada
        candidates = []
        for coin, df in ind.items():
            i = pos_index[coin].get(t)
            if i is None or coin in positions:
                continue
            tech = snapshot_at(df, i)
            if tech is None:
                continue
            rep = trig.evaluate(coin, tech, None, None, None, now=t.timestamp())
            if rep.triggered:
                candidates.append(rep)
        candidates.sort(key=lambda r: r.score, reverse=True)
        res.anomalies += len(candidates)
        for rep in candidates:
            if not trig.claude_budget_available(t.timestamp()):
                res.blocked["presupuesto Claude/h"] += 1
                continue
            ok, why = risk.can_open(len(positions) + len(pending), eq, open_risk(), now_dt)
            if not ok:
                res.blocked[why.split(":")[0]] += 1
                continue
            trig.mark_escalated(rep.coin, t.timestamp())
            res.escalations += 1
            if direction == "bias":
                if rep.bias == "neutral":
                    res.blocked["sesgo neutral (Claude: NO_TRADE)"] += 1
                    continue
                is_long = rep.bias == "long"
            else:  # contrarian, para comparar
                if rep.bias == "neutral":
                    continue
                is_long = rep.bias == "short"
            pending.append((rep.coin, is_long, rep.score))

    # cierre final a mercado de lo que quede abierto
    for coin in list(positions):
        close_part(positions[coin], positions[coin].size, last_close[coin], timeline[-1], "fin_backtest")
    return res


def report(res: BTResult, s: Settings, label: str) -> dict:
    eq = pd.Series([e for _, e in res.equity], index=[t for t, _ in res.equity])
    trades = pd.DataFrame([t.__dict__ for t in res.trades])
    days = (eq.index[-1] - eq.index[0]).total_seconds() / 86400 if len(eq) > 1 else 0
    final = eq.iloc[-1] if len(eq) else s.initial_capital
    dd = ((eq.cummax() - eq) / eq.cummax()).max() if len(eq) else 0
    out = {
        "label": label, "dias": round(days, 1), "equity_final": round(final, 2),
        "retorno_pct": round((final / s.initial_capital - 1) * 100, 2), "max_drawdown_pct": round(dd * 100, 2),
        "trades": len(trades), "anomalias": res.anomalies, "escalados_claude": res.escalations,
    }
    if len(trades):
        wins = trades[trades.pnl > 0]
        losses = trades[trades.pnl <= 0]
        out.update({
            "win_rate_pct": round(len(wins) / len(trades) * 100, 1),
            "tp1_alcanzado_pct": round(trades.tp1_hit.mean() * 100, 1),
            "profit_factor": round(wins.pnl.sum() / abs(losses.pnl.sum()), 2) if len(losses) and losses.pnl.sum() else math.inf,
            "R_medio": round(trades.r_multiple.mean(), 3),
            "pnl_medio_usd": round(trades.pnl.mean(), 3),
            "mejor_R": round(trades.r_multiple.max(), 2), "peor_R": round(trades.r_multiple.min(), 2),
            "duracion_media_h": round(trades.bars.mean() * 0.25, 1),
            "trades_dia": round(len(trades) / days, 2) if days else 0,
            "largos_pct": round((trades.side == "LONG").mean() * 100, 1),
        })
    out["circuit_breakers"] = res.circuit_breakers
    out["house_money_desde"] = res.house_money_at
    return out


def print_report(r: dict, res: BTResult) -> None:
    print(f"\n=== {r['label']} ===")
    for k, v in r.items():
        if k != "label":
            print(f"  {k:22s} {v}")
    trades = pd.DataFrame([t.__dict__ for t in res.trades])
    if len(trades):
        print("  Salidas:", dict(Counter(trades.reason)))
        by_coin = trades.groupby("coin").agg(n=("pnl", "size"), pnl=("pnl", "sum")).sort_values("pnl")
        print("  PnL por activo (peores / mejores):")
        print("   ", ", ".join(f"{c} {row.pnl:+.1f}({row.n})" for c, row in by_coin.head(4).iterrows()))
        print("   ", ", ".join(f"{c} {row.pnl:+.1f}({row.n})" for c, row in by_coin.tail(4).iterrows()))
    if res.blocked:
        print("  Señales bloqueadas:", dict(res.blocked.most_common(6)))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="*")
    ap.add_argument("--days", type=float)
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--direction", choices=["bias", "contrarian"], default="bias")
    ap.add_argument("--sweep", action="store_true", help="compara varios umbrales y la variante contrarian")
    ap.add_argument("--out", type=Path, help="carpeta donde guardar trades.csv / equity.csv / resumen.json")
    args = ap.parse_args()

    base = get_settings()
    cache = base.state_dir / "candles_cache"
    cache.mkdir(parents=True, exist_ok=True)
    print("Descargando velas de Hyperliquid...")
    data, meta = await load_data(base, args.coins, args.interval, cache)

    def settings_for(**over) -> Settings:
        tmp = Path(tempfile.mkdtemp())
        return base.model_copy(update={"state_dir": tmp, "trading_mode": "paper", **over})

    runs = []
    if args.sweep:
        for th in (2.5, 3.0, 3.5, 4.0):
            runs.append((f"umbral {th} / sesgo", settings_for(anomaly_score_threshold=th), "bias"))
        runs.append(("umbral 3.0 / contrarian", settings_for(anomaly_score_threshold=3.0), "contrarian"))
    else:
        over = {"anomaly_score_threshold": args.threshold} if args.threshold else {}
        th = args.threshold or base.anomaly_score_threshold
        runs.append((f"umbral {th} / {args.direction}", settings_for(**over), args.direction))

    summaries = []
    for label, s, direction in runs:
        res = run_backtest(s, data, meta, args.days, direction)
        r = report(res, s, label)
        print_report(r, res)
        summaries.append(r)
        if args.out and len(runs) == 1:
            args.out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([t.__dict__ for t in res.trades]).to_csv(args.out / "trades.csv", index=False)
            pd.DataFrame(res.equity, columns=["time", "equity"]).to_csv(args.out / "equity.csv", index=False)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "resumen.json").write_text(json.dumps(summaries, indent=2, default=str, ensure_ascii=False))
    print(f"\nGenerado {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")


if __name__ == "__main__":
    asyncio.run(main())
