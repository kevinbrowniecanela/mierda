"""Validación de la rotación de estrategias (walk-forward con control aleatorio).

Cada `step` meses se eligen las k mejores estrategias según su expectativa en
los `lookback` meses anteriores (solo con operaciones YA cerradas) y se operan
durante el periodo siguiente. Se compara contra:
  - el conjunto fijo,
  - "todas las estrategias a la vez",
  - elegir k estrategias AL AZAR (500 sorteos): si elegir no supera al azar,
    la selección no aporta habilidad y el resultado es suerte.

    python research/rotation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab  # noqa: E402
import validate as v  # noqa: E402
from quant.strategies import STRATEGIES  # noqa: E402

START_EVAL = v.HL_START + pd.DateOffset(months=18)


def windows(end: pd.Timestamp, step: int):
    s = START_EVAL
    while s < end:
        yield s, s + pd.DateOffset(months=step)
        s += pd.DateOffset(months=step)


def pick(per: dict, start, lookback: int, k: int, min_trades: int = 15) -> list[str]:
    scores = {}
    for n, ts in per.items():
        past = [t.r for t in ts if t.t_out < start and t.t_in >= start - pd.DateOffset(months=lookback)]
        if len(past) >= min_trades:
            scores[n] = float(np.mean(past))
    return [n for n, s in sorted(scores.items(), key=lambda kv: -kv[1]) if s > 0][:k]


def collect(per: dict, picker, end, step) -> list:
    out = []
    for s, e in windows(end, step):
        for n in picker(s):
            out += [t for t in per[n] if s <= t.t_in < e]
    return out


def summary(trades: list) -> tuple[float, float, float, float, int]:
    rs = np.array([t.r for t in trades])
    if len(rs) < 20:
        return (float("nan"),) * 4 + (len(rs),)
    p = lab.portfolio(trades, risk=0.01, max_pos=3, start=224.0)
    t = rs.mean() / (rs.std(ddof=1) / np.sqrt(len(rs)))
    return float(rs.mean()), float(t), float(p["cagr"]), float(p["maxdd"]), len(rs)


def main() -> None:
    data, _ = v.load_all()
    elig = v.point_in_time(data)
    lab.load_funding()
    lab.USE_REAL_FUNDING = True
    names = [n for n in STRATEGIES if n != "ensemble_diario"]
    per = {n: v.run(data, n, elig) for n in names}
    end = max(t.t_out for ts in per.values() for t in ts)
    base = [t for t in v.run(data, "ensemble_diario", elig) if t.t_in >= START_EVAL]
    allstr = [t for n in names for t in per[n] if t.t_in >= START_EVAL]
    print(f"Evaluación desde {START_EVAL.date()} hasta {end.date()} | {len(names)} estrategias | universo punto-en-el-tiempo\n")
    print(f"{'configuración':34s} {'exp R':>7} {'t':>6} {'anual%':>7} {'DD%':>6} {'ops':>5} | azar: exp medio  % veces que gana al azar")
    for label, tr in (("Conjunto fijo (actual)", base), ("Todas las estrategias a la vez", allstr)):
        e, t, c, d, n = summary(tr)
        print(f"{label:34s} {e:>+7.3f} {t:>+6.2f} {c:>+7.1f} {d:>6.1f} {n:>5d}")
    rng = np.random.default_rng(3)
    best = None
    for step in (1, 3, 6):
        for lookback in (6, 9, 12, 18):
            for k in (1, 3):
                tr = collect(per, lambda s: pick(per, s, lookback, k), end, step)
                e, t, c, d, n = summary(tr)
                # control: k estrategias al azar en cada ventana
                rand = []
                for _ in range(300):
                    sel = {s: list(rng.choice(names, size=k, replace=False)) for s, _e in windows(end, step)}
                    r = [t_.r for s, e_ in windows(end, step) for nme in sel[s] for t_ in per[nme] if s <= t_.t_in < e_]
                    if len(r) >= 20:
                        rand.append(np.mean(r))
                rm = float(np.mean(rand))
                beat = float(np.mean([e > x for x in rand]) * 100) if rand and not np.isnan(e) else float("nan")
                print(f"rotar cada {step}m, ventana {lookback:>2}m, top {k}      {e:>+7.3f} {t:>+6.2f} {c:>+7.1f} {d:>6.1f} {n:>5d} | {rm:>+7.3f}   {beat:>5.0f}%")
                if best is None or (not np.isnan(e) and e > best[0]):
                    best = (e, step, lookback, k)
    print(f"\nMejor combinación (ojo: elegida mirando el resultado, por tanto optimista): {best}")
    print("Si 'gana al azar' no supera ~95 % de forma consistente en muchas combinaciones, la selección no demuestra habilidad.")


if __name__ == "__main__":
    main()
