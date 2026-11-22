"""Lectura REAL (solo lectura, sin órdenes) contra la API de Hyperliquid.

Se omite por defecto para no depender de la red. Para ejecutarla:

    RUN_LIVE_TESTS=1 pytest tests/test_strategy_live.py -v          # testnet (TESTNET_API_URL)
    RUN_LIVE_TESTS=1 LIVE_NETWORK=mainnet pytest tests/test_strategy_live.py -v
"""
import os

import pandas as pd
import pytest

from strategy.market_data import MarketDataClient

pytestmark = pytest.mark.skipif(os.environ.get("RUN_LIVE_TESTS") != "1", reason="prueba de red: define RUN_LIVE_TESTS=1")


@pytest.fixture(scope="module")
def live(request):
    from config.settings import Settings
    testnet = os.environ.get("LIVE_NETWORK", "testnet") != "mainnet"
    return MarketDataClient(Settings(_env_file=None, use_testnet=testnet), pace_sec=0.1)


def test_live_read_only_pipeline(live):
    offset = live.sync_clock()
    assert abs(offset) < 60_000  # el reloj local no se desvía más de un minuto
    ctxs = live.get_perp_contexts()
    assert len(ctxs) > 20 and "BTC" in ctxs
    btc = ctxs["BTC"]
    assert btc.mark_px > 0 and btc.open_interest > 0 and btc.sz_decimals >= 0 and btc.max_leverage >= 1
    assert btc.predicted_funding is not None  # endpoint predictedFundings (venue HlPerp)
    wl = live.build_watchlist(ctxs, top_volume=5, top_movers=2)
    assert wl[0] in ctxs and len(wl) >= 5
    cs = live.get_candle_set("BTC")
    for tf in ("4h", "1h", "15m"):
        df = cs.frames[tf]
        assert len(df) > 100 and df["closed"].iloc[0]
        now = pd.Timestamp(live.now_ms(), unit="ms", tz="UTC")
        assert (df[df["closed"]]["close_time"] < now).all() and (df[~df["closed"]]["close_time"] >= now).all()
        assert cs.closed(tf).index[-1] <= df.index[-1]
    assert cs.forming("15m") is not None  # a mitad de vela siempre hay una en formación
    l2 = live.get_l2_metrics("BTC")
    assert 0 < l2.spread_bps < 50 and l2.bid_depth_usd > 0 and l2.ask_depth_usd > 0
