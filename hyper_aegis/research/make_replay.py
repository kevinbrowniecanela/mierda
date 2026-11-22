"""Genera dashboard/replay.html: repetición animada del bot sobre datos REALES recientes.

Dos escenarios, ambos con el sistema de tendencia diario del bot y sus reglas de riesgo:
  - Acciones de EE. UU.: solo largos, sin apalancamiento, 1 % de riesgo, 5 posiciones.
  - Cripto (Hyperliquid): largos y cortos, funding real, 1 % de riesgo, 3 posiciones.
Es un backtest, no operaciones reales. La página es autocontenida (sin servidor ni librerías).

    python research/make_replay.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab  # noqa: E402
import stocks_study as ss  # noqa: E402
import validate as v  # noqa: E402
import volspike_study as vs  # noqa: E402
from quant.strategies import STRATEGIES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WINDOW_DAYS = 150


def build(key, label, currency, frames, trades, anchor, risk, max_pos, lev_cap, start):
    end = max(t.t_out for t in trades)
    days = [d for d in frames[anchor].index if d >= end - pd.Timedelta(days=WINDOW_DAYS)]
    d0 = days[0]
    idx = {d: i for i, d in enumerate(days)}
    pos = lambda t: idx.get(t, None)
    win = sorted([t for t in trades if t.t_in >= d0 and t.t_in in idx], key=lambda t: t.t_in)
    eq, open_, taken, closed_events = start, [], [], []
    for t in win:
        for o in [o for o in open_ if o["t"].t_out <= t.t_in]:
            eq += o["pnl"]; open_.remove(o)
        if len(open_) >= max_pos or any(o["t"].coin == t.coin for o in open_):
            continue
        eff = min(risk, lev_cap / max_pos * t.stop_pct)
        notional = eq * eff / t.stop_pct
        if notional < 10:
            continue
        pnl = eq * eff * t.r
        open_.append({"t": t, "pnl": pnl, "notional": notional})
        taken.append({"t": t, "pnl": pnl, "notional": notional})
    # curva diaria con valoración a mercado
    realized, equity = start, []
    for d in days:
        r = start + sum(x["pnl"] for x in taken if x["t"].t_out <= d)
        u = 0.0
        for x in taken:
            t = x["t"]
            if t.t_in <= d < t.t_out and d in frames[t.coin].index:
                u += x["notional"] * (frames[t.coin]["close"].loc[d] / t.entry_px - 1) * t.side
        equity.append(round(r + u, 2))
    out_trades = []
    for x in taken:
        t = x["t"]
        di = idx[t.t_in]
        xi = idx.get(t.t_out, len(days) - 1) if t.t_out <= days[-1] else len(days)  # len(days) = sigue abierta
        stops, cur, path = [], t.entry_px - t.side * (t.entry_px * t.stop_pct), dict(t.stops or [])
        for k in range(di, min(xi, len(days) - 1) + 1):
            cur = path.get(days[k], cur)
            stops.append(round(float(cur), 6))
        moved = len(set(round(s, 4) for s in stops)) > 1
        out_trades.append({"coin": t.coin, "side": "LONG" if t.side > 0 else "SHORT", "di": di, "xi": xi, "entry": round(t.entry_px, 6),
                           "exit": round(t.exit_px, 6), "notional": round(x["notional"], 2), "pnl": round(x["pnl"], 2), "r": round(t.r, 3),
                           "reason": ("trailing_stop" if moved else "stop_loss") if xi < len(days) else "abierta", "stops": stops})
    coins = sorted({x["coin"] for x in out_trades})
    candles = {}
    for c in coins:
        df = frames[c]; df = df[(df.index >= d0 - pd.Timedelta(days=90)) & (df.index <= days[-1])]
        candles[c] = [[int(t.timestamp()), round(float(r.open), 6), round(float(r.high), 6), round(float(r.low), 6), round(float(r.close), 6)] for t, r in df.iterrows()]
    return {"key": key, "label": label, "currency": currency, "start": start, "days": [int(d.timestamp()) for d in days],
            "equity": equity, "trades": out_trades, "candles": candles, "risk": risk, "max_pos": max_pos}


def main() -> None:
    sets = []
    # --- acciones ---------------------------------------------------------- #
    stocks = ss.load_stocks(); ss.cost_mode("stocks")
    frames = {c: d for c, d in stocks.items() if c not in ss.BENCH}
    liq = pd.DataFrame({c: d["raw_close"] * d["volume"] for c, d in frames.items()}).rolling(30, min_periods=20).median().shift(1)
    el40 = {c: (liq.rank(axis=1, ascending=False) <= 40)[c] for c in liq.columns}
    mk = lambda d: (lambda sp: (sp.update(sig=np.clip(sp["sig"], 0, None)), sp)[1])(STRATEGIES["ensemble_diario"](d))
    lab.RECORD_PATHS = True
    tr = vs.run(frames, mk, el40)
    sets.append(build("stocks", "Acciones de EE. UU. (solo largos, sin apalancamiento)", "USD", {**frames, "SPY": stocks["SPY"]}, tr, "SPY", 0.01, 5, 1.0, 224.0))
    # --- cripto ------------------------------------------------------------ #
    data_c, _ = v.load_all()
    elig_c = v.point_in_time(data_c)
    lab.load_funding(); ss.cost_mode("crypto")
    trc = v.run(data_c, "ensemble_diario", elig_c)
    sets.append(build("crypto", "Cripto en Hyperliquid (largos y cortos, funding real)", "USDC", data_c, trc, "BTC", 0.01, 3, 5.0, 224.0))
    lab.RECORD_PATHS = False
    tpl = (ROOT / "dashboard" / "replay.template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*__CHARTS__*/", (ROOT / "dashboard" / "charts.js").read_text(encoding="utf-8")).replace("/*__DATA__*/null", json.dumps(sets, separators=(",", ":")))
    out = ROOT / "dashboard" / "replay.html"
    out.write_text(html, encoding="utf-8")
    for s in sets:
        n = len(s["trades"]); w = sum(1 for t in s["trades"] if t["pnl"] > 0)
        print(f"{s['key']}: {len(s['days'])} días, {n} operaciones ({w} ganadoras), capital {s['start']:.0f} -> {s['equity'][-1]:.0f} {s['currency']}")
    print(f"Escrito {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
