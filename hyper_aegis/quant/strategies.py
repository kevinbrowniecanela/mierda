"""Estrategias validadas en el laboratorio (research/lab.py).

Cada función devuelve un `spec` con arrays causales (solo usan datos hasta la
vela i): `sig` (+1 largo, -1 corto, 0 nada), `stop_k` (stop inicial en ATR) y
el modo de salida. El bot en vivo usa `latest_signal()` sobre el conjunto
`ENSEMBLE`, que es la familia de seguimiento de tendencia en velas diarias que
resultó positiva dentro y fuera de muestra en 5,6 años de datos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + g / l.replace(0, np.nan))).fillna(100)


def trend_ok(df: pd.DataFrame, n: int = 200) -> tuple[pd.Series, pd.Series]:
    e = ema(df["close"], n)
    slope = e - e.shift(10)
    return (df["close"] > e) & (slope > 0), (df["close"] < e) & (slope < 0)


def s_donchian(df: pd.DataFrame, n: int = 20, trail: float = 3.0, filt: bool = True) -> dict:
    hi = df["high"].rolling(n).max().shift()
    lo = df["low"].rolling(n).min().shift()
    up, dn = trend_ok(df) if filt else (True, True)
    sig = np.where((df["close"] > hi) & up, 1, np.where((df["close"] < lo) & dn, -1, 0))
    return dict(sig=sig, stop_k=2.0, exit="chandelier", trail_k=trail)


def s_pullback(df: pd.DataFrame, rsi_lvl: float = 40, exit: str = "chandelier") -> dict:
    """Retroceso dentro de tendencia: RSI sale de sobreventa a favor de la tendencia."""
    e50, e200 = ema(df["close"], 50), ema(df["close"], 200)
    r = rsi(df["close"])
    up = (e50 > e200) & (df["close"] > e200)
    dn = (e50 < e200) & (df["close"] < e200)
    long = up & (r.shift() < rsi_lvl) & (r >= rsi_lvl)
    short = dn & (r.shift() > 100 - rsi_lvl) & (r <= 100 - rsi_lvl)
    return dict(sig=np.where(long, 1, np.where(short, -1, 0)), stop_k=1.5, exit=exit, trail_k=3.0)


def s_tsmom(df: pd.DataFrame, look: int = 60, trail: float = 4.0) -> dict:
    """Momentum de serie temporal: retorno de `look` velas cambia de signo."""
    mom = df["close"] / df["close"].shift(look) - 1
    flip_up = (mom > 0) & (mom.shift() <= 0)
    flip_dn = (mom < 0) & (mom.shift() >= 0)
    sig = np.where(flip_up, 1, np.where(flip_dn, -1, 0))
    return dict(sig=sig, stop_k=3.0, exit="chandelier", trail_k=trail,
                exit_long=(mom < 0).to_numpy(), exit_short=(mom > 0).to_numpy())


def s_meanrev(df: pd.DataFrame, lvl: float = 10, hold: int = 8) -> dict:
    """Reversión a la media: RSI(2) extremo A FAVOR de la tendencia de fondo."""
    r2 = rsi(df["close"], 2)
    e200, e5 = ema(df["close"], 200), ema(df["close"], 5)
    long = (df["close"] > e200) & (r2 < lvl)
    short = (df["close"] < e200) & (r2 > 100 - lvl)
    return dict(sig=np.where(long, 1, np.where(short, -1, 0)), stop_k=2.5, exit="signal", max_bars=hold,
                exit_long=(df["close"] > e5).to_numpy(), exit_short=(df["close"] < e5).to_numpy())


def s_squeeze(df: pd.DataFrame, n: int = 20, trail: float = 3.0) -> dict:
    """Ruptura tras compresión de volatilidad (ancho de Bollinger en mínimos)."""
    ma = df["close"].rolling(n).mean()
    sd = df["close"].rolling(n).std()
    width = (4 * sd / ma)
    squeezed = width.shift() <= width.shift().rolling(120).quantile(0.2)
    up, dn = trend_ok(df)
    long = squeezed & (df["close"] > ma + 2 * sd) & up
    short = squeezed & (df["close"] < ma - 2 * sd) & dn
    return dict(sig=np.where(long, 1, np.where(short, -1, 0)), stop_k=2.0, exit="chandelier", trail_k=trail)


def s_volspike(df: pd.DataFrame, exit: str = "tp_partial") -> dict:
    """La lógica del bot original: spike de volumen + rango, a favor de tendencia."""
    a = atr(df)
    vol_ratio = df["volume"] / df["volume"].rolling(20).mean()
    rng = (df["high"] - df["low"]) / a.shift()
    ret = df["close"].pct_change()
    up, dn = trend_ok(df)
    long = (vol_ratio >= 3) & (rng >= 2) & (ret > 0) & up
    short = (vol_ratio >= 3) & (rng >= 2) & (ret < 0) & dn
    return dict(sig=np.where(long, 1, np.where(short, -1, 0)), stop_k=1.5, exit=exit, trail_k=2.0)


STRATEGIES = {
    "donchian20_tendencia": lambda df: s_donchian(df, 20),
    "donchian55_tendencia": lambda df: s_donchian(df, 55),
    "donchian20_sin_filtro": lambda df: s_donchian(df, 20, filt=False),
    "pullback_rsi40": lambda df: s_pullback(df, 40),
    "pullback_rsi40_tp": lambda df: s_pullback(df, 40, exit="tp_partial"),
    "tsmom_60": lambda df: s_tsmom(df, 60),
    "tsmom_120": lambda df: s_tsmom(df, 120),
    "meanrev_rsi2": lambda df: s_meanrev(df),
    "squeeze_breakout": lambda df: s_squeeze(df),
    "volspike_bot_tp": lambda df: s_volspike(df, "tp_partial"),
    "volspike_chandelier": lambda df: s_volspike(df, "chandelier"),
}


ENSEMBLE = ("donchian55_tendencia", "squeeze_breakout", "donchian20_tendencia")


def s_ensemble(df: pd.DataFrame, names: tuple[str, ...] = ENSEMBLE) -> dict:
    """Entra cuando cualquiera de las estrategias de tendencia del conjunto da señal."""
    total = np.zeros(len(df))
    for n in names:
        total = total + np.asarray(STRATEGIES[n](df)["sig"])
    return dict(sig=np.sign(total).astype(int), stop_k=2.0, exit="chandelier", trail_k=3.0)


STRATEGIES["ensemble_diario"] = s_ensemble


@dataclass
class Signal:
    coin: str
    direction: int            # +1 largo, -1 corto
    strategies: list[str]     # estrategias del conjunto que coinciden
    bar_time: pd.Timestamp    # vela cerrada que genera la señal
    close: float
    atr: float
    stop_k: float
    trail_k: float

    @property
    def side(self) -> str:
        return "LONG" if self.direction > 0 else "SHORT"


def latest_signal(coin: str, closed: pd.DataFrame, names: tuple[str, ...] = ENSEMBLE) -> Signal | None:
    """Señal del conjunto en la ÚLTIMA vela cerrada de `closed` (o None)."""
    if len(closed) < 260:
        return None
    fired: dict[str, int] = {}
    for n in names:
        v = int(np.asarray(STRATEGIES[n](closed)["sig"])[-1])
        if v:
            fired[n] = v
    total = sum(fired.values())
    if total == 0:
        return None
    direction = 1 if total > 0 else -1
    a = float(atr(closed).iloc[-1])
    if not np.isfinite(a) or a <= 0:
        return None
    return Signal(coin=coin, direction=direction, strategies=[n for n, v in fired.items() if v == direction],
                  bar_time=closed.index[-1], close=float(closed["close"].iloc[-1]), atr=a, stop_k=2.0, trail_k=3.0)


def regime(closed: pd.DataFrame) -> str:
    """Régimen de tendencia de fondo en la última vela: alcista / bajista / lateral."""
    up, dn = trend_ok(closed)
    return "alcista" if bool(up.iloc[-1]) else "bajista" if bool(dn.iloc[-1]) else "lateral"
