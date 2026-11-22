"""¿Qué fórmula de cartera, con aportaciones mensuales, conserva mejor las ganancias?

Piezas (acciones/ETF de EE. UU., 2010-2026, sin apalancamiento):
  - Núcleo: SPY o QQQ (comprar y mantener), opcionalmente solo cuando SPY > media de 200 días.
  - Satélite: el bot de tendencia largo (top 40 por liquidez, cuenta al contado, 5 posiciones).
  - Bloqueo de ganancias: cada trimestre se traspasa al núcleo una parte de la ganancia del satélite.

Todas las fórmulas reciben la MISMA aportación (224 USD iniciales + 110 USD al mes).
Métricas: TIR anual (ponderada por dinero), caída máxima y peor año (ponderadas por tiempo, sin
efecto de las aportaciones), y valor final frente a lo aportado.

Limitaciones: el satélite se aproxima con su curva de ganancias REALIZADAS (se actualiza al cerrar
operaciones); universo con sesgo de supervivencia; sin impuestos; el efectivo rinde 0 %.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab  # noqa: E402
import stocks_study as ss  # noqa: E402
import volspike_study as vs  # noqa: E402
from quant.strategies import STRATEGIES  # noqa: E402

START_CASH, MONTHLY = 224.0, 110.0


def sleeve_nav(frames, liq, risk: float) -> pd.Series:
    el40 = {c: (liq.rank(axis=1, ascending=False) <= 40)[c] for c in liq.columns}
    mk = lambda d: (lambda sp: (sp.update(sig=np.clip(sp["sig"], 0, None)), sp)[1])(STRATEGIES["ensemble_diario"](d))
    trades = vs.run(frames, mk, el40)
    p = lab.portfolio(trades, risk=risk, max_pos=5, lev_cap=1.0, start=1000.0)
    s = pd.Series([e for _, e in p["curve"]], index=pd.DatetimeIndex([t for t, _ in p["curve"]]).normalize())
    return s.groupby(level=0).last() / 1000.0


def irr_monthly(flows: list[float]) -> float:
    lo, hi = -0.5, 0.5
    for _ in range(200):
        mid = (lo + hi) / 2
        npv = sum(f / (1 + mid) ** i for i, f in enumerate(flows))
        lo, hi = (mid, hi) if npv > 0 else (lo, mid)  # flows: aportes negativos, valor final positivo
    return (1 + (lo + hi) / 2) ** 12 - 1


def simulate(dates, r_core, flag_core, r_sleeve, core_w: float, lock: float, name: str, to_cash: bool = False):
    core = sleeve = vault = 0.0
    basis = 0.0
    v_prev, vals, twr, flows_m, last_month, last_q = 0.0, [], [], [], None, None
    for t in dates:
        flow = 0.0
        q = (t.year, (t.month - 1) // 3)
        if last_q is not None and q != last_q and lock > 0 and sleeve > basis > 0:
            moved = lock * (sleeve - basis)  # traspaso de ganancias al núcleo
            sleeve -= moved
            if to_cash:
                vault += moved  # caja: no se invierte
            else:
                core += moved
            basis = sleeve
        last_q = q
        if (t.year, t.month) != last_month:
            last_month = (t.year, t.month)
            flow = START_CASH if not vals else MONTHLY
            core += flow * core_w
            sleeve += flow * (1 - core_w)
            basis += flow * (1 - core_w)
            flows_m.append(-flow)
        start_v = core + sleeve + vault
        core *= 1 + (r_core[t] if flag_core[t] else 0.0)
        sleeve *= 1 + r_sleeve[t]
        v = core + sleeve + vault
        twr.append(v / start_v - 1 if start_v > 0 else 0.0)
        vals.append(v)
    flows_m[-1] += vals[-1]  # valor final en el último mes
    twr_s = pd.Series(twr, index=dates)
    nav = (1 + twr_s).cumprod()
    dd = float(((nav.cummax() - nav) / nav.cummax()).max() * 100)
    yearly = nav.groupby(nav.index.year).last().pct_change().dropna()
    contributed = START_CASH + MONTHLY * (len(flows_m) - 1)
    return dict(name=name, final=vals[-1], contributed=contributed, irr=irr_monthly(flows_m) * 100, dd=dd,
                worst=float(yearly.min() * 100) if len(yearly) else float("nan"), twr=float((nav.iloc[-1] ** (252 / len(nav)) - 1) * 100))


def main() -> None:
    stocks = ss.load_stocks()
    ss.cost_mode("stocks")
    frames = {c: d for c, d in stocks.items() if c not in ss.BENCH}
    liq = pd.DataFrame({c: d["raw_close"] * d["volume"] for c, d in frames.items()}).rolling(30, min_periods=20).median().shift(1)
    spy, qqq = stocks["SPY"]["close"], stocks["QQQ"]["close"]
    flag_spy = (spy.shift(1) > spy.rolling(200).mean().shift(1))
    print("Calculando el satélite (bot de tendencia largo, sin apalancamiento)...")
    sl = {r: sleeve_nav(frames, liq, r) for r in (0.01, 0.02)}
    for r, s in sl.items():
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        print(f"   satélite con {r:.0%} de riesgo: {(s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1:+.1%} anual aislado")
    start = max(sl[0.02].index[0], spy.index[0]) + pd.Timedelta(days=1)
    dates = spy.index[spy.index >= start]
    ret = lambda s: s.reindex(dates).pct_change().fillna(0.0)
    r_spy, r_qqq = ret(spy), ret(qqq)
    always = pd.Series(True, index=dates)
    flag = flag_spy.reindex(dates).fillna(False)
    r_sl = {r: s.reindex(pd.date_range(s.index[0], dates[-1])).ffill().reindex(dates).pct_change().fillna(0.0) for r, s in sl.items()}
    zero = pd.Series(0.0, index=dates)
    print(f"\nPeriodo {dates[0].date()} -> {dates[-1].date()} | aportación inicial {START_CASH:.0f} USD + {MONTHLY:.0f} USD/mes\n")
    forms = [
        ("A. Solo SPY (comprar y mantener)",         lambda: simulate(dates, r_spy, always, zero, 1.0, 0, "")),
        ("B. Solo QQQ",                               lambda: simulate(dates, r_qqq, always, zero, 1.0, 0, "")),
        ("C. SPY solo si SPY > media 200 d",         lambda: simulate(dates, r_spy, flag, zero, 1.0, 0, "")),
        ("D. Solo el bot (satélite 100 %, 2 % riesgo)", lambda: simulate(dates, r_spy, always, r_sl[0.02], 0.0, 0, "")),
        ("E. 70 % SPY + 30 % bot",                   lambda: simulate(dates, r_spy, always, r_sl[0.02], 0.7, 0, "")),
        ("F. E + bloqueo 50 % de ganancias/trim.",    lambda: simulate(dates, r_spy, always, r_sl[0.02], 0.7, 0.5, "")),
        ("G. E + bloqueo 100 % de ganancias/trim.",   lambda: simulate(dates, r_spy, always, r_sl[0.02], 0.7, 1.0, "")),
        ("H. 70 % SPY(filtro 200 d) + 30 % bot + bloqueo 50 %", lambda: simulate(dates, r_spy, flag, r_sl[0.02], 0.7, 0.5, "")),
        ("E2. 70/30 + bloqueo 50 % a CAJA (efectivo)",  lambda: simulate(dates, r_spy, always, r_sl[0.02], 0.7, 0.5, "", True)),
        ("E3. 70/30 + bloqueo 100 % a CAJA",          lambda: simulate(dates, r_spy, always, r_sl[0.02], 0.7, 1.0, "", True)),
        ("D2. Solo bot + bloqueo 50 % a CAJA",        lambda: simulate(dates, r_spy, always, r_sl[0.02], 0.0, 0.5, "", True)),
        ("D3. Solo bot + bloqueo 100 % a CAJA",       lambda: simulate(dates, r_spy, always, r_sl[0.02], 0.0, 1.0, "", True)),
        ("H2. 70 % SPY(filtro) + 30 % bot + 50 % a CAJA", lambda: simulate(dates, r_spy, flag, r_sl[0.02], 0.7, 0.5, "", True)),
        ("I. 50 % SPY + 50 % bot + bloqueo 50 %",     lambda: simulate(dates, r_spy, always, r_sl[0.02], 0.5, 0.5, "")),
        ("J. 70 % QQQ + 30 % bot + bloqueo 50 %",     lambda: simulate(dates, r_qqq, always, r_sl[0.02], 0.7, 0.5, "")),
        ("K. 70 % SPY + 30 % bot(1 % riesgo) + bloqueo 50 %", lambda: simulate(dates, r_spy, always, r_sl[0.01], 0.7, 0.5, "")),
    ]
    print(f"{'fórmula':56s} {'final USD':>10} {'aportado':>9} {'x':>5} {'TIR %':>6} {'caída %':>8} {'peor año %':>10} {'TIR/caída':>9}")
    for label, fn in forms:
        r = fn()
        print(f"{label:56s} {r['final']:>10,.0f} {r['contributed']:>9,.0f} {r['final'] / r['contributed']:>5.2f} {r['irr']:>6.1f} {r['dd']:>8.1f} {r['worst']:>10.1f} {r['irr'] / r['dd']:>9.2f}")


if __name__ == "__main__":
    main()
