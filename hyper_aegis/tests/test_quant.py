import numpy as np
import pandas as pd

from data.market_feed import L2Book
from quant.anomaly_trigger import AnomalyTrigger
from quant.technicals import snapshot


def make_df(n=300, spike=False):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    df = pd.DataFrame({"open": close, "high": close + 0.4, "low": close - 0.4, "close": close,
                       "volume": rng.uniform(90, 110, n)}, index=idx)
    if spike:
        # la penúltima vela (última cerrada) con volumen y rango anómalos
        df.iloc[-2, df.columns.get_loc("volume")] = 1000
        df.iloc[-2, df.columns.get_loc("close")] = df["close"].iloc[-3] + 3
        df.iloc[-2, df.columns.get_loc("high")] = df["close"].iloc[-3] + 3.2
    return df


def test_snapshot_fields():
    t = snapshot(make_df())
    assert t is not None
    assert t.atr > 0 and 0 <= t.rsi <= 100
    assert t.trend in {"alcista", "bajista", "lateral"}


def test_snapshot_needs_history():
    assert snapshot(make_df(100)) is None


def test_anomaly_trigger_and_cooldown(settings):
    trig = AnomalyTrigger(settings)
    calm = trig.evaluate("BTC", snapshot(make_df()), None, None, None)
    assert not calm.triggered
    tech = snapshot(make_df(spike=True))
    book = L2Book("BTC", bids=[(99.9, 50)], asks=[(100.1, 5)])
    rep = trig.evaluate("BTC", tech, None, 0.05, book)
    assert rep.triggered and rep.bias == "long", rep.summary()
    trig.mark_escalated("BTC")
    assert not trig.evaluate("BTC", tech, None, 0.05, book).triggered  # cooldown


def test_book_imbalance():
    book = L2Book("X", bids=[(99.9, 10)], asks=[(100.1, 10)])
    assert abs(book.imbalance()) < 0.01
    assert book.spread_bps > 0
