"""HyperAegis — orquestador principal asíncrono.

Uso:
    python run.py              # arranca el bot (modo según TRADING_MODE)
    python run.py --check      # diagnóstico: conectividad, watchlist y un escaneo sin operar
    python run.py --reset-shield   # rearma el Capital Shield tras una violación

Tareas concurrentes:
    watchlist_loop  -> recalcula Top20 volumen + Top5 movers
    scan_loop       -> Fase 1 (filtro cuantitativo) y, si procede, Fase 2 (Claude)
    position_loop   -> equity, Capital Shield, circuit breaker, TP1/BE/trailing
    WebSocket L2    -> libro de órdenes en tiempo real
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aiohttp  # noqa: E402

from alerts.telegram_notifier import TelegramNotifier  # noqa: E402
from brain.claude_analyst import ClaudeAnalyst  # noqa: E402
from brain.schemas import Action  # noqa: E402
from config.settings import Settings, get_settings  # noqa: E402
from data.derivatives_data import DerivativesData  # noqa: E402
from data.market_feed import MarketFeed  # noqa: E402
from data.news_engine import NewsEngine  # noqa: E402
from execution.hyperliquid_trader import HyperliquidTrader, TradeEvent  # noqa: E402
from execution.risk_manager import RiskManager  # noqa: E402
from execution.tax_exporter import TaxExporter  # noqa: E402
from quant.anomaly_trigger import AnomalyReport, AnomalyTrigger  # noqa: E402
from quant.technicals import TechSnapshot, snapshot  # noqa: E402

log = logging.getLogger("hyperaegis")


def setup_logging(s: Settings) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(s.log_level.upper())
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    file = RotatingFileHandler(s.log_dir / "hyperaegis.log", maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    file.setFormatter(fmt)
    root.handlers[:] = [console, file]
    for noisy in ("websockets", "urllib3", "httpx", "anthropic._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class HyperAegis:
    def __init__(self, settings: Settings, session: aiohttp.ClientSession):
        self.s = settings
        self.http = session
        self.feed = MarketFeed(settings)
        self.deriv = DerivativesData(settings, self.feed)
        self.news = NewsEngine(settings, session)
        self.trigger = AnomalyTrigger(settings)
        self.analyst = ClaudeAnalyst(settings)
        self.risk = RiskManager(settings)
        self.tax = TaxExporter(settings, session)
        self.trader = HyperliquidTrader(settings, self.feed, self.tax)
        self.tg = TelegramNotifier(settings, session)
        self.tech: dict[str, TechSnapshot] = {}
        self.marks: dict[str, float] = {}
        self._stop = asyncio.Event()
        self._escalation_lock = asyncio.Lock()
        self.stats = {"scans": 0, "anomalies": 0, "claude_calls": 0, "trades": 0}

    # ------------------------------------------------------------------ #
    # Bucle: watchlist
    # ------------------------------------------------------------------ #
    async def watchlist_loop(self) -> None:
        while not self._stop.is_set():
            try:
                wl = await self.deriv.build_watchlist()
                # Las monedas con posición abierta siempre se vigilan
                self.feed.set_ws_coins(sorted(set(wl) | set(self.trader.positions)))
            except Exception:  # noqa: BLE001
                log.exception("Error actualizando watchlist")
            await self._sleep(self.s.watchlist_refresh_sec)

    # ------------------------------------------------------------------ #
    # Bucle: escaneo (Fase 1 -> Fase 2)
    # ------------------------------------------------------------------ #
    async def _analyze_coin(self, coin: str) -> tuple[str, TechSnapshot | None, AnomalyReport | None]:
        try:
            df = await self.feed.candles(coin)
            tech = snapshot(df)
            if tech is None:
                return coin, None, None
            book = self.feed.books.get(coin)
            report = self.trigger.evaluate(coin, tech, self.deriv.snapshots.get(coin), self.deriv.oi_change(coin), book)
            return coin, tech, report
        except Exception as exc:  # noqa: BLE001
            log.warning("Análisis %s falló: %s", coin, exc)
            return coin, None, None

    async def scan_once(self, allow_trading: bool = True) -> list[AnomalyReport]:
        await self.deriv.refresh()
        coins = sorted(set(self.deriv.watchlist) | set(self.trader.positions))
        results = await asyncio.gather(*(self._analyze_coin(c) for c in coins))
        triggered: list[tuple[AnomalyReport, TechSnapshot]] = []
        for coin, tech, report in results:
            if tech:
                self.tech[coin] = tech
            if report and report.triggered:
                triggered.append((report, tech))
        self.stats["scans"] += 1
        triggered.sort(key=lambda x: x[0].score, reverse=True)
        for report, _ in triggered:
            log.info("ANOMALÍA %s", report.summary())
        self.stats["anomalies"] += len(triggered)
        if allow_trading:
            for report, tech in triggered:
                if self._stop.is_set():
                    break
                await self.escalate(report, tech)
        return [r for r, _ in triggered]

    async def scan_loop(self) -> None:
        while not self.deriv.watchlist and not self._stop.is_set():
            await asyncio.sleep(1)
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                await self.scan_once()
            except Exception:  # noqa: BLE001
                log.exception("Error en el escaneo")
            await self._sleep(max(5.0, self.s.scan_interval_sec - (time.monotonic() - t0)))

    async def escalate(self, report: AnomalyReport, tech: TechSnapshot) -> None:
        coin = report.coin
        async with self._escalation_lock:
            if coin in self.trader.positions:
                return
            equity = await self.trader.equity(self.marks)
            ok, why = self.risk.can_open(self.trader.open_count, equity, self.trader.open_risk)
            if not ok:
                log.info("Anomalía %s ignorada: %s", coin, why)
                return
            if not self.trigger.claude_budget_available():
                log.info("Presupuesto horario de Claude agotado; %s ignorada", coin)
                return
            self.trigger.mark_escalated(coin)
            self.stats["claude_calls"] += 1

            deriv = self.deriv.snapshots.get(coin)
            news_items = await self.news.for_coin(coin)
            try:
                funding_24h = await self.deriv.funding_history(coin, 24)
            except Exception:  # noqa: BLE001
                funding_24h = []
            book = self.feed.books.get(coin)
            context = {
                "coin": coin,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "timeframe": self.s.candle_interval,
                "anomaly": {"score": round(report.score, 2), "bias_quant": report.bias, "signals": report.reasons},
                "technicals": tech.to_dict(),
                "derivatives": None if not deriv else {
                    "mark_px": deriv.mark_px, "oracle_px": deriv.oracle_px, "change_24h": round(deriv.change_24h, 5),
                    "volume_24h_usd": round(deriv.day_ntl_vlm), "open_interest_usd": round(deriv.oi_usd),
                    "oi_change_1h": self.deriv.oi_change(coin), "funding_hourly": deriv.funding_hourly,
                    "funding_annualized": round(deriv.funding_annualized, 4), "premium": deriv.premium,
                    "funding_24h_avg": (sum(funding_24h) / len(funding_24h)) if funding_24h else None,
                },
                "orderbook": None if not book else {
                    "mid": book.mid, "spread_bps": book.spread_bps,
                    "imbalance_0.5pct": round(book.imbalance(0.005), 3), "imbalance_1pct": round(book.imbalance(0.01), 3),
                },
                "account": {
                    "equity_usdc": round(equity, 2), "initial_capital": self.risk.initial,
                    "house_money_active": self.risk.state.house_money_active,
                    "daily_drawdown": round(self.risk.daily_drawdown(equity), 4),
                    "open_positions": [
                        {"coin": p.coin, "side": "LONG" if p.is_long else "SHORT", "entry": p.entry_px, "tp1_hit": p.tp1_hit}
                        for p in self.trader.positions.values()
                    ],
                },
            }
            result = await self.analyst.analyze(context, [n.line() for n in news_items])
            d = result.decision
            if d.action == Action.NO_TRADE or d.confidence < self.s.claude_min_confidence:
                log.info("Claude: %s %s (conf %d) — %s", coin, d.action.value, d.confidence, d.thesis[:200])
                return
            is_long = d.action == Action.LONG
            if (report.bias == "long" and not is_long) or (report.bias == "short" and is_long):
                log.info("Claude va contra el sesgo cuantitativo (%s vs %s); se exige confianza extra", d.action.value, report.bias)
                if d.confidence < min(100, self.s.claude_min_confidence + 10):
                    return

            mids = await self.feed.all_mids()
            self.marks.update(mids)
            price = mids.get(coin, tech.close)
            if not d.stop_is_coherent(price):
                log.warning("Stop de Claude incoherente para %s (%s @ %s); se usa ATR", coin, d.stop_loss, price)
            sz_dec = await self.feed.sz_decimals(coin)
            plan, why = self.risk.build_plan(
                coin=coin, is_long=is_long, price=price, atr=tech.atr,
                proposed_stop=d.stop_loss if d.stop_is_coherent(price) else None,
                sz_decimals=sz_dec, equity=equity, open_risk=self.trader.open_risk,
                open_count=self.trader.open_count, max_asset_lev=deriv.max_leverage if deriv else self.s.max_leverage,
            )
            if plan is None:
                log.info("Plan de %s descartado: %s", coin, why)
                return
            event = await self.trader.open_position(
                plan, sz_dec, deriv.max_leverage if deriv else self.s.max_leverage, d.confidence, d.setup.value
            )
            if event.kind == "entry":
                self.stats["trades"] += 1
                pos = event.data["pos"]
                self.feed.set_ws_coins(sorted(set(self.deriv.watchlist) | set(self.trader.positions)))
                self.tg.entry(coin, pos.is_long, pos.size, pos.entry_px, pos.stop, pos.tp1, pos.risk_usd,
                              d.confidence, d.setup.value, d.thesis)
            else:
                self.tg.alert(f"{coin}: {event.data.get('msg')}")

    # ------------------------------------------------------------------ #
    # Bucle: posiciones y Capital Shield
    # ------------------------------------------------------------------ #
    async def position_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.position_tick()
            except Exception:  # noqa: BLE001
                log.exception("Error en el bucle de posiciones")
            await self._sleep(self.s.position_check_interval_sec)

    async def position_tick(self) -> None:
        self.marks.update(await self.feed.all_mids())
        equity = await self.trader.equity(self.marks)
        for ev in self.risk.update_equity(equity):
            await self._handle_risk_event(ev, equity)
        atrs = {c: t.atr for c, t in self.tech.items()}
        fundings = {c: a.funding_hourly for c, a in self.deriv.snapshots.items()}
        for ev in await self.trader.manage(self.marks, atrs, fundings):
            self._notify_trade_event(ev)

    async def _handle_risk_event(self, ev: str, equity: float) -> None:
        if ev == "house_money_on":
            self.tg.alert(f"HOUSE MONEY activado (equity {equity:.2f} USDC). El depósito inicial queda blindado.")
        elif ev == "circuit_breaker":
            self.tg.alert(f"CIRCUIT BREAKER: drawdown diario {self.risk.daily_drawdown(equity):.2%}. Sin nuevas entradas hoy.")
            if self.s.flatten_on_circuit_breaker:
                for e in await self.trader.close_all(self.marks, "circuit_breaker"):
                    self._notify_trade_event(e)
        elif ev == "shield_breached":
            self.tg.alert("CAPITAL SHIELD violado: cerrando todo y deteniendo nuevas entradas (rearme: run.py --reset-shield).")
            for e in await self.trader.close_all(self.marks, "capital_shield"):
                self._notify_trade_event(e)
        elif ev == "new_day":
            self.daily_summary(equity)
            await self.tax.backfill_pending()

    def _notify_trade_event(self, ev: TradeEvent) -> None:
        d = ev.data
        if ev.kind == "tp1":
            self.tg.tp1(ev.coin, d["closed"], d["px"], d["pnl"], d["new_stop"])
        elif ev.kind == "exit":
            self.tg.exit(ev.coin, d["reason"], d["px"], d["pnl"], d["pnl_eur"])
        elif ev.kind == "stop_moved":
            log.info("Trailing %s: %s -> %s", ev.coin, d["old"], d["new"])
        elif ev.kind == "error":
            self.tg.alert(f"{ev.coin}: {d.get('msg')}")

    def daily_summary(self, equity: float) -> None:
        rows = self.trader.closed_today
        net = sum(r.net_pnl_usd for r in rows)
        net_eur = sum(r.net_pnl_eur for r in rows)
        trades = {r.trade_id for r in rows}
        wins = {r.trade_id for r in rows if r.net_pnl_usd > 0}
        st = self.risk.state
        total = equity - self.risk.initial
        self.tg.summary(
            "Resumen diario",
            [
                f"Equity: {equity:,.2f} USDC ({total:+.2f} / {total / self.risk.initial:+.2%} desde inicio)",
                f"Cierres registrados: {len(rows)} (operaciones: {len(trades)}, con beneficio: {len(wins)})",
                f"PnL neto realizado: {net:+.2f} USDC ({net_eur:+.2f} €)",
                f"Posiciones abiertas: {self.trader.open_count} | Riesgo abierto: {self.trader.open_risk:.2f} USDC",
                f"House Money: {'SÍ' if st.house_money_active else 'no'} | Pico equity: {st.peak_equity:,.2f}",
                f"Escaneos: {self.stats['scans']} | Anomalías: {self.stats['anomalies']} | Llamadas Claude: {self.stats['claude_calls']}",
            ],
        )
        self.trader.closed_today.clear()

    # ------------------------------------------------------------------ #
    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    def request_stop(self) -> None:
        log.info("Parada solicitada")
        self._stop.set()

    async def run(self) -> None:
        if self.s.is_live:
            self.s.validate_for_live()
        log.info("HyperAegis arrancando en modo %s (%s)", self.s.trading_mode.value.upper(),
                 "testnet" if self.s.use_testnet else "mainnet")
        self.tg.start()
        await self.deriv.build_watchlist()
        self.feed.set_ws_coins(sorted(set(self.deriv.watchlist) | set(self.trader.positions)))
        for w in await self.trader.reconcile():
            self.tg.alert(w)
        await self.tax.backfill_pending()
        self.marks.update(await self.feed.all_mids())
        equity = await self.trader.equity(self.marks)
        self.risk.update_equity(equity)
        self.tg.startup(self.s.trading_mode.value, equity, self.deriv.watchlist, self.s.claude_model)

        tasks = [
            self.feed.start_ws(),
            asyncio.create_task(self.watchlist_loop(), name="watchlist"),
            asyncio.create_task(self.scan_loop(), name="scan"),
            asyncio.create_task(self.position_loop(), name="positions"),
        ]
        await self._stop.wait()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self.s.is_live and self.trader.positions:
            log.info("Las posiciones abiertas siguen protegidas por SL/TP en el exchange")
        self.tg.send("🛑 HyperAegis detenido.")
        await self.tg.stop()


async def check(app: HyperAegis) -> None:
    """Diagnóstico sin operar."""
    s = app.s
    print(f"Modo: {s.trading_mode.value} | Red: {'testnet' if s.use_testnet else 'mainnet'} | Modelo: {s.claude_model}")
    wl = await app.deriv.build_watchlist()
    print(f"Watchlist ({len(wl)}): {', '.join(wl)}")
    app.marks.update(await app.feed.all_mids())
    print(f"Equity ({s.trading_mode.value}): {await app.trader.equity(app.marks):.2f} USDC")
    reports = await app.scan_once(allow_trading=False)
    print(f"Anomalías detectadas: {len(reports)}")
    for r in reports:
        print("  ", r.summary())
    news = await app.news.latest()
    print(f"Titulares cargados: {len(news)}")
    rate, day, src = await app.tax.usd_to_eur(datetime.now(timezone.utc))
    print(f"USD->EUR {rate} ({day}, {src})")


async def main() -> None:
    parser = argparse.ArgumentParser(description="HyperAegis trading bot")
    parser.add_argument("--check", action="store_true", help="diagnóstico sin operar")
    parser.add_argument("--reset-shield", action="store_true", help="rearmar el Capital Shield")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings)

    if args.reset_shield:
        RiskManager(settings).reset_shield()
        print("Capital Shield rearmado.")
        return

    async with aiohttp.ClientSession(headers={"User-Agent": "HyperAegis/1.0"}, trust_env=True) as session:
        app = HyperAegis(settings, session)
        if args.check:
            await check(app)
            return
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, app.request_stop)
        await app.run()


if __name__ == "__main__":
    asyncio.run(main())
