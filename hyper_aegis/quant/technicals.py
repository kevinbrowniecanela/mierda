"""Indicadores técnicos: EMA 20/50/200, RSI, ATR, VWAP.

Usa `pandas_ta` si está instalado; si no (p. ej. incompatibilidad de numpy
en alguna versión de macOS/Python), cae a implementaciones equivalentes en
pandas puro para que el bot nunca quede bloqueado por la dependencia.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:  # pragma: no cover - depende del entorno
    import pandas_ta as ta  # type: ignore

    HAS_PANDAS_TA = True
except Exception:  # noqa: BLE001
    ta = None
    HAS_PANDAS_TA = False
    log.info("pandas_ta no disponible; usando implementaciones nativas de pandas")


# ---------------------------------------------------------------------- #
# Implementaciones de respaldo (mismas fórmulas que pandas_ta)
# ---------------------------------------------------------------------- #
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift()
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _vwap_daily(df: pd.DataFrame) -> pd.Series:
    """VWAP anclado al inicio de cada día UTC."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical * df["volume"]
    day = df.index.floor("D")
    return pv.groupby(day).cumsum() / df["volume"].groupby(day).cumsum().replace(0, np.nan)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Añade columnas de indicadores al DataFrame OHLCV (índice datetime UTC)."""
    out = df.copy()
    if HAS_PANDAS_TA:
        out["ema20"] = ta.ema(out["close"], length=20)
        out["ema50"] = ta.ema(out["close"], length=50)
        out["ema200"] = ta.ema(out["close"], length=200)
        out["rsi"] = ta.rsi(out["close"], length=14)
        out["atr"] = ta.atr(out["high"], out["low"], out["close"], length=14)
        try:
            out["vwap"] = ta.vwap(out["high"], out["low"], out["close"], out["volume"], anchor="D")
        except Exception:  # noqa: BLE001
            out["vwap"] = _vwap_daily(out)
    else:
        out["ema20"] = _ema(out["close"], 20)
        out["ema50"] = _ema(out["close"], 50)
        out["ema200"] = _ema(out["close"], 200)
        out["rsi"] = _rsi(out["close"], 14)
        out["atr"] = _atr(out, 14)
        out["vwap"] = _vwap_daily(out)
    out["vol_sma20"] = out["volume"].rolling(20).mean()
    out["ret"] = out["close"].pct_change()
    return out


@dataclass
class TechSnapshot:
    close: float
    ema20: float
    ema50: float
    ema200: float
    rsi: float
    atr: float
    atr_pct: float
    vwap: float
    volume: float
    vol_sma20: float
    vol_ratio: float
    bar_range: float
    range_atr: float
    ret_1: float
    ret_4: float
    trend: str  # "alcista" | "bajista" | "lateral"

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def snapshot(df: pd.DataFrame) -> TechSnapshot | None:
    """Resumen de la última vela CERRADA (se descarta la vela en formación)."""
    if len(df) < 210:
        return None
    ind = add_indicators(df)
    return snapshot_at(ind, len(ind) - 2)  # la última vela de Hyperliquid está abierta


def snapshot_at(ind: pd.DataFrame, i: int) -> TechSnapshot | None:
    """Snapshot usando la fila `i` de un DataFrame con indicadores como última
    vela cerrada. Solo usa datos <= i (sin look-ahead); útil para backtesting."""
    if i < 205:
        return None
    last = ind.iloc[i]
    prev_atr = float(ind["atr"].iat[i - 1])
    close = float(last["close"])
    ema20, ema50, ema200 = float(last["ema20"]), float(last["ema50"]), float(last["ema200"])
    if close > ema20 > ema50 > ema200:
        trend = "alcista"
    elif close < ema20 < ema50 < ema200:
        trend = "bajista"
    else:
        trend = "lateral"
    vol_sma = float(last["vol_sma20"]) or 1e-12
    bar_range = float(last["high"] - last["low"])
    atr = float(last["atr"])
    closes = ind["close"]
    return TechSnapshot(
        close=close,
        ema20=ema20,
        ema50=ema50,
        ema200=ema200,
        rsi=float(last["rsi"]),
        atr=atr,
        atr_pct=atr / close if close else 0.0,
        vwap=float(last["vwap"]) if pd.notna(last["vwap"]) else close,
        volume=float(last["volume"]),
        vol_sma20=vol_sma,
        vol_ratio=float(last["volume"]) / vol_sma,
        bar_range=bar_range,
        range_atr=bar_range / prev_atr if prev_atr else 0.0,
        ret_1=float(closes.iat[i] / closes.iat[i - 1] - 1),
        ret_4=float(closes.iat[i] / closes.iat[i - 4] - 1),
        trend=trend,
    )
