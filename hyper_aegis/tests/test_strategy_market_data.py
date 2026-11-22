"""Módulo 1: velas con marca de cierre, contexto de perpetuos, OI, L2, watchlist y reintentos."""
import time

import pytest

from strategy import market_data as md
from strategy.market_data import LiquidityPolicy, MarketDataClient, MarketDataError

NOW = 1_800_000_000_000  # ms
M15, H1 = 900_000, 3_600_000


def candle(t, o=100, h=101, l=99, c=100.5, v=10.0, interval=M15):
    return {"t": t, "T": t + interval - 1, "s": "BTC", "i": "15m", "o": str(o), "c": str(c), "h": str(h), "l": str(l), "v": str(v), "n": 5}


class FakeInfo:
    def __init__(self):
        self.calls = []
        self.fail_times = 0
        base = NOW - NOW % M15
        self.raw = [candle(base - 3 * M15), candle(base - 2 * M15), candle(base - M15), candle(base)]  # la última está en formación
        self.universe = [
            {"name": "BTC", "szDecimals": 5, "maxLeverage": 40},
            {"name": "ETH", "szDecimals": 4, "maxLeverage": 25},
            {"name": "PEPE", "szDecimals": 0, "maxLeverage": 10},
            {"name": "OLD", "szDecimals": 1, "maxLeverage": 3, "isDelisted": True},
            {"name": "LOW", "szDecimals": 1, "maxLeverage": 3},
        ]
        self.ctxs = [
            {"markPx": "60000", "oraclePx": "60010", "midPx": "60001", "prevDayPx": "59000", "funding": "0.0000125", "premium": "0.0001", "openInterest": "1000", "dayNtlVlm": "900000000"},
            {"markPx": "3000", "oraclePx": "3000", "midPx": None, "prevDayPx": "3300", "funding": "-0.0002", "premium": "0", "openInterest": "5000", "dayNtlVlm": "400000000"},
            {"markPx": "0.001", "oraclePx": "0.001", "prevDayPx": "0.0007", "funding": "0.0003", "premium": "0", "openInterest": "1000000000", "dayNtlVlm": "30000000"},
            {"markPx": "1", "oraclePx": "1", "prevDayPx": "1", "funding": "0", "premium": "0", "openInterest": "1", "dayNtlVlm": "1"},
            {"markPx": "10", "oraclePx": "10", "prevDayPx": "10", "funding": "0", "premium": "0", "openInterest": "1", "dayNtlVlm": "1000"},
        ]

    def _maybe_fail(self, name):
        self.calls.append(name)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ConnectionError("boom")

    def candles_snapshot(self, coin, interval, start, end):
        self._maybe_fail("candles")
        return self.raw

    def meta_and_asset_ctxs(self):
        self._maybe_fail("meta")
        return {"universe": self.universe}, self.ctxs

    def post(self, path, payload):
        assert payload == {"type": "predictedFundings"}
        return [["BTC", [["BinPerp", {"fundingRate": "0.0004", "fundingIntervalHours": 4}], ["HlPerp", {"fundingRate": "0.00002", "fundingIntervalHours": 1}]]]]

    def l2_snapshot(self, coin):
        self._maybe_fail("l2")
        return {"coin": coin, "time": NOW + 1500, "levels": [
            [{"px": "59990", "sz": "1.0", "n": 3}, {"px": "59710", "sz": "2.0", "n": 2}, {"px": "50000", "sz": "9", "n": 1}],
            [{"px": "60010", "sz": "0.5", "n": 2}, {"px": "60290", "sz": "1.0", "n": 1}, {"px": "70000", "sz": "9", "n": 1}]]}

    def user_state(self, address):
        return {"marginSummary": {"accountValue": "123.45"}}


@pytest.fixture
def client(settings, monkeypatch):
    monkeypatch.setattr(time, "time", lambda: NOW / 1000)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    return MarketDataClient(settings, info=FakeInfo(), pace_sec=0)


def test_uses_network_from_settings(settings):
    from hyperliquid.utils import constants
    assert MarketDataClient(settings.model_copy(update={"use_testnet": True}), info=object()).api_url == constants.TESTNET_API_URL
    assert MarketDataClient(settings.model_copy(update={"use_testnet": False}), info=object()).api_url == constants.MAINNET_API_URL


def test_candles_flag_the_forming_candle(client):
    df = client.get_candles("BTC", "15m")
    assert list(df["closed"]) == [True, True, True, False]  # la última vela está en formación
    assert df.index.is_monotonic_increasing and df["close"].dtype == float


def test_clock_offset_changes_closed_flag(client):
    base = NOW - NOW % M15
    client.clock_offset_ms = 0
    assert client.get_candles("BTC", "15m")["closed"].iloc[-1] == False  # noqa: E712
    client.clock_offset_ms = M15 + 10  # el servidor va por delante: esa vela ya cerró para él
    assert client.get_candles("BTC", "15m")["closed"].iloc[-1]
    assert client.sync_clock() == NOW + 1500 - NOW  # offset = hora del servidor - local
    assert base > 0


def test_candle_set_closed_helpers(client):
    cs = client.get_candle_set("BTC")
    assert set(cs.frames) == {"4h", "1h", "15m"}
    assert len(cs.closed("15m")) == 3 and cs.last_closed("15m").name == cs.closed("15m").index[-1]
    assert cs.forming("15m") is not None and not cs.forming("15m")["closed"]


def test_perp_contexts_and_predicted_funding(client):
    ctxs = client.get_perp_contexts()
    assert set(ctxs) == {"BTC", "ETH", "PEPE", "LOW"}  # el deslistado no aparece
    btc = ctxs["BTC"]
    assert btc.funding == pytest.approx(1.25e-5) and btc.predicted_funding == pytest.approx(2e-5)  # venue HlPerp, no BinPerp
    assert btc.oi_usd == pytest.approx(6e7) and btc.sz_decimals == 5 and btc.max_leverage == 40
    assert ctxs["ETH"].change_24h == pytest.approx(-0.0909, abs=1e-3) and ctxs["ETH"].mid_px is None
    assert client.sz_decimals("PEPE") == 0


def test_oi_and_price_change_from_memory(client):
    assert client.oi_change_pct("BTC") is None  # sin lecturas previas
    t = time.time()
    client.seed_oi_history("BTC", [(t - 1800, 800.0, 58000.0)])
    client.get_perp_contexts()  # añade la lectura actual: OI 1000, precio 60000
    assert client.oi_change_pct("BTC", 3600) == pytest.approx(0.25)
    assert client.price_change_pct("BTC", 3600) == pytest.approx(60000 / 58000 - 1)
    assert client.oi_change_pct("BTC", 60) is None  # la lectura previa queda fuera de la ventana


def test_l2_metrics_spread_and_depth(client):
    m = client.get_l2_metrics("BTC", depth_pct=0.005)
    assert m.spread_bps == pytest.approx(20 / 60000 * 10_000)
    assert m.bid_depth_usd == pytest.approx(59990 * 1.0 + 59710 * 2.0)  # el nivel a 50000 queda fuera de ±0,5 %
    assert m.ask_depth_usd == pytest.approx(60010 * 0.5 + 60290 * 1.0)


def test_liquidity_policy(client):
    ctxs = client.get_perp_contexts()
    pol = LiquidityPolicy(min_day_volume_usd=5e6, max_spread_bps=10, min_depth_usd=20_000)
    assert pol.check(ctxs["BTC"])[0] and not pol.check(ctxs["LOW"])[0]
    l2 = client.get_l2_metrics("BTC")
    assert pol.check(ctxs["BTC"], l2)[0]
    assert not LiquidityPolicy(max_spread_bps=1.0).check(ctxs["BTC"], l2)[0]
    assert not LiquidityPolicy(min_depth_usd=1e9).check(ctxs["BTC"], l2)[0]


def test_watchlist_top_volume_plus_movers(client):
    ctxs = client.get_perp_contexts()
    wl = client.build_watchlist(ctxs, top_volume=2, top_movers=1, movers_min_volume_usd=1e6)
    assert wl[:2] == ["BTC", "ETH"] and wl[2] == "PEPE"  # PEPE: +43 % en 24 h y volumen suficiente
    assert "LOW" not in client.build_watchlist(ctxs, 2, 5)  # volumen insuficiente para los movers


def test_account_value(client):
    assert client.get_account_value("0xabc") == pytest.approx(123.45)


def test_retries_then_succeeds_and_then_raises(client):
    client.info.fail_times = 2
    assert len(client.get_candles("BTC", "15m")) == 4  # 2 fallos y éxito al tercer intento
    client.info.fail_times = 99
    with pytest.raises(MarketDataError):
        client.get_candles("BTC", "15m")


def test_invalid_interval(client):
    with pytest.raises(ValueError):
        client.get_candles("BTC", "7m")


@pytest.mark.asyncio
async def test_async_wrappers(client):
    cs = await client.aget_candle_set("BTC")
    assert len(cs.closed("1h")) == 3
    assert "BTC" in await client.aget_perp_contexts()
    assert (await client.aget_l2_metrics("BTC")).spread_bps > 0
    assert (await client.abuild_watchlist(2, 0)) == ["BTC", "ETH"]


def test_parse_empty():
    assert md.MarketDataClient.parse_candles([], NOW).empty
