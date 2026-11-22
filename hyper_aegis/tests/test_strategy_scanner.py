"""Escáner: plomería con un cliente simulado y persistencia del histórico de OI."""
import json
import time

import pytest

from strategy import scanner
from strategy.market_data import LiquidityPolicy, MarketDataClient
from tests.test_strategy_market_data import NOW, FakeInfo


@pytest.fixture
def client(settings, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    return MarketDataClient(settings, info=FakeInfo(), pace_sec=0, policy=LiquidityPolicy(min_day_volume_usd=5e6))


def test_scan_discards_by_stage_and_never_orders(client):
    rep = scanner.scan(client, coins=["BTC", "LOW", "NOPE"], account_value=224.0)
    assert rep.discarded["liquidez_volumen"] == 1 and rep.discarded["no_existe"] == 1  # LOW sin volumen; NOPE inexistente
    assert rep.discarded["filtro_data"] == 1  # BTC: el simulador solo da 3 velas cerradas
    assert rep.signals == [] and rep.network == client.api_url
    assert "LOW" in rep.notes and "volumen" in rep.notes["LOW"]
    assert not hasattr(client.info, "order")  # el cliente es de solo lectura


def test_scan_builds_watchlist_when_none_given(client):
    rep = scanner.scan(client, top_volume=2, top_movers=0)
    assert rep.watchlist == ["BTC", "ETH"]


def test_oi_history_roundtrip(client, tmp_path):
    path = tmp_path / "oi.json"
    client.get_perp_contexts()
    scanner.save_oi_history(client, path)
    saved = json.loads(path.read_text())
    assert "BTC" in saved and saved["BTC"][0][1] == 1000.0
    fresh = MarketDataClient(client.settings, info=FakeInfo(), pace_sec=0)
    scanner.load_oi_history(fresh, path)
    assert len(fresh._oi_hist["BTC"]) == 1  # noqa: SLF001
    old = {"BTC": [[time.time() - 10 * 3600, 5.0, 1.0]]}  # lecturas de hace más de 6 h se ignoran
    path.write_text(json.dumps(old))
    again = MarketDataClient(client.settings, info=FakeInfo(), pace_sec=0)
    scanner.load_oi_history(again, path)
    assert len(again._oi_hist["BTC"]) == 0  # noqa: SLF001


def test_print_report_runs(client, capsys):
    rep = scanner.scan(client, coins=["BTC", "LOW"])
    scanner.print_report(rep, verbose=True)
    out = capsys.readouterr().out
    assert "SOLO LECTURA" in out and "Sin señales" in out and "liquidez_volumen=1" in out


def test_cli_json_and_network_flags(settings, monkeypatch, capsys):
    monkeypatch.setattr(scanner, "get_settings", lambda: settings.model_copy(update={"state_dir": settings.state_dir}))
    seen = {}

    def fake_init(self, settings=None, **kw):
        seen["url"] = settings.hl_api_url
        MarketDataClient.__init_orig__(self, settings, info=FakeInfo(), pace_sec=0, policy=kw.get("policy"))

    monkeypatch.setattr(MarketDataClient, "__init_orig__", MarketDataClient.__init__, raising=False)
    monkeypatch.setattr(MarketDataClient, "__init__", fake_init)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    assert scanner.main(["--testnet", "--coins", "LOW", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "testnet" in seen["url"] and out["discarded"] == {"liquidez_volumen": 1} and out["signals"] == []
    assert NOW > 0
