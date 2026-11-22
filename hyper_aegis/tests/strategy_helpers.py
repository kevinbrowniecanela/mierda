"""Datos sintéticos para probar el módulo `strategy` sin red."""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.indicators import add_indicators


def frame(closes, spread=0.05, vol=100.0, freq="15min", highs=None, lows=None, vols=None) -> pd.DataFrame:
    """Velas con open = cierre previo; high/low = extremos del cuerpo +/- `spread` (o los dados)."""
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + spread if highs is None else np.asarray(highs, dtype=float)
    l = np.minimum(o, c) - spread if lows is None else np.asarray(lows, dtype=float)
    v = np.full(len(c), vol) if vols is None else np.asarray(vols, dtype=float)
    idx = pd.date_range("2026-01-01", periods=len(c), freq=freq, tz="UTC")
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)
    df["trades"], df["closed"] = 10, True
    df["close_time"] = df.index + pd.Timedelta(freq) - pd.Timedelta(milliseconds=1)
    return df


def random_walk(n=300, vol=0.004, seed=1, start=100.0, freq="15min", volume=100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = start * np.exp(np.cumsum(rng.normal(0, vol, n)))
    spread = closes * vol * 0.5
    return frame(closes, spread=spread, vol=volume, freq=freq)


def with_ind(df: pd.DataFrame) -> pd.DataFrame:
    return add_indicators(df[df["closed"]])


def replace_last(df: pd.DataFrame, **cols) -> pd.DataFrame:
    """Copia de `df` con la última vela modificada."""
    out = df.copy()
    for k, v in cols.items():
        out.iloc[-1, out.columns.get_loc(k)] = v
    return out
