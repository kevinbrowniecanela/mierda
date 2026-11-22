"""Indicadores técnicos: `pandas-ta` como motor principal, con respaldo en pandas puro.

`add_indicators(df)` devuelve una copia con: ema21, ema50, ema200, atr14, rsi14, adx14, di_plus,
di_minus, bb_upper, bb_mid, bb_lower, vol_sma20 (media de los 20 volúmenes ANTERIORES a cada vela,
sin incluirla) y rel_volume (volumen / vol_sma20). Todo es causal: la fila i solo usa datos hasta i.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:  # pragma: no cover - depende del entorno
    import pandas_ta as ta  # type: ignore

    HAS_PANDAS_TA = True
except Exception:  # noqa: BLE001
    ta = None  # type: ignore[assignment]
    HAS_PANDAS_TA = False


# ---------------------------------------------------------------------- #
# Respaldo en pandas puro (mismas fórmulas que pandas-ta)
# ---------------------------------------------------------------------- #
def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def native_ema(s: pd.Series, n: int) -> pd.Series:
    """EMA con semilla SMA(n), igual que pandas-ta."""
    x = s.astype(float).copy()
    if len(x) < n:
        return pd.Series(np.nan, index=s.index)
    seed = x.iloc[:n].mean()
    x.iloc[: n - 1] = np.nan
    x.iloc[n - 1] = seed
    return x.ewm(span=n, adjust=False).mean()


def native_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up, dn = d.clip(lower=0), (-d.clip(upper=0))
    rs = _rma(up, n) / _rma(dn, n).replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(_rma(up, n).notna())


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift()
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def native_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return _rma(true_range(df), n)


def native_adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    up, dn = df["high"].diff(), -df["low"].diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr = native_atr(df, n)
    di_p, di_m = 100 * _rma(plus, n) / atr, 100 * _rma(minus, n) / atr
    dx = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    return pd.DataFrame({"adx": _rma(dx.fillna(0.0), n).where(atr.notna()), "di_plus": di_p, "di_minus": di_m})


def native_bbands(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid, sd = close.rolling(n).mean(), close.rolling(n).std(ddof=1)  # ddof=1, como pandas-ta
    return pd.DataFrame({"bb_lower": mid - k * sd, "bb_mid": mid, "bb_upper": mid + k * sd})


# ---------------------------------------------------------------------- #
def _col(frame: pd.DataFrame, prefix: str) -> pd.Series:
    """Columna de pandas-ta por prefijo (el sufijo con los parámetros cambia entre versiones)."""
    name = next(c for c in frame.columns if c.startswith(prefix))
    return frame[name]


def _with_pandas_ta(out: pd.DataFrame) -> pd.DataFrame:
    out["ema21"] = ta.ema(out["close"], length=21)
    out["ema50"] = ta.ema(out["close"], length=50)
    out["ema200"] = ta.ema(out["close"], length=200)
    out["atr14"] = ta.atr(out["high"], out["low"], out["close"], length=14)
    out["rsi14"] = ta.rsi(out["close"], length=14)
    adx = ta.adx(out["high"], out["low"], out["close"], length=14)
    out["adx14"], out["di_plus"], out["di_minus"] = _col(adx, "ADX_"), _col(adx, "DMP_"), _col(adx, "DMN_")
    bb = ta.bbands(out["close"], length=20, std=2.0)  # type: ignore[arg-type]
    out["bb_lower"], out["bb_mid"], out["bb_upper"] = _col(bb, "BBL_"), _col(bb, "BBM_"), _col(bb, "BBU_")
    return out


def _with_native(out: pd.DataFrame) -> pd.DataFrame:
    out["ema21"] = native_ema(out["close"], 21)
    out["ema50"] = native_ema(out["close"], 50)
    out["ema200"] = native_ema(out["close"], 200)
    out["atr14"] = native_atr(out, 14)
    out["rsi14"] = native_rsi(out["close"], 14)
    adx = native_adx(out, 14)
    out["adx14"], out["di_plus"], out["di_minus"] = adx["adx"], adx["di_plus"], adx["di_minus"]
    bb = native_bbands(out["close"], 20, 2.0)
    out["bb_lower"], out["bb_mid"], out["bb_upper"] = bb["bb_lower"], bb["bb_mid"], bb["bb_upper"]
    return out


def add_indicators(df: pd.DataFrame, use_pandas_ta: bool = True) -> pd.DataFrame:
    """Añade los indicadores a una copia de `df` (columnas open, high, low, close, volume).

    Con pocas velas (activo recién listado) `pandas-ta` devuelve `None`; en ese caso, o si falla de cualquier
    otra forma, se usa el cálculo nativo, que devuelve NaN donde no hay datos suficientes.
    """
    out = df.copy()
    if use_pandas_ta and HAS_PANDAS_TA and len(out) >= 30:
        try:
            out = _with_pandas_ta(out)
        except (AttributeError, TypeError, StopIteration, KeyError) as exc:
            log.debug("pandas-ta falló (%r); se usa el cálculo nativo", exc)
            out = _with_native(df.copy())
    else:
        out = _with_native(out)
    out["vol_sma20"] = out["volume"].rolling(20).mean().shift(1)  # media de las 20 velas ANTERIORES
    out["rel_volume"] = out["volume"] / out["vol_sma20"].replace(0, np.nan)
    return out
