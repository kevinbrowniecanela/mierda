"""HyperAegis — orquestador principal asíncrono.

Uso:
    python run.py                  # arranca el bot (modo según TRADING_MODE)
    python run.py --check          # diagnóstico: conectividad, señales de hoy, sin operar
    python run.py --reset-shield   # rearma el Capital Shield / kill switch

Flujo:
    1. Sistema cuantitativo (quant/strategies.py): seguimiento de tendencia en
       velas diarias sobre la watchlist dinámica. Sin coste de IA.
    2. Claude supervisa cada señal y puede vetarla (noticias, eventos, riesgo).
    3. RiskManager dimensiona (riesgo % del equity actual), el trader ejecuta
       y gestiona el trailing stop.
    4. El vigilante comprueba que nada falla y corta cualquier fuga de dinero.
    5. Panel web local de solo lectura en http://127.0.0.1:8787
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
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
from dashboard.server import Dashboard  # noqa: E402
from data.derivatives_data import DerivativesData  # noqa: E402
from data.market_feed import MarketFeed  # noqa: E402
from data.news_engine import NewsEngine  # noqa: E402
from execution.hyperliquid_trader import HyperliquidTrader, TradeEvent  # noqa: E402
from execution.risk_manager import RiskManager  # noqa: E402
from execution.tax_exporter import TaxExporter  # noqa: E402
from execution.watchdog import Incident, Watchdog  # noqa: E402
from quant.strategies import Signal, atr as atr_fn, latest_signal, regime  # noqa: E402

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
    for noisy in ("websockets", "urllib3", "httpx", "anthropic._base_client", "aiohttp.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class HyperAegis:
    def __init__(self, settings: Settings, session: aiohttp.ClientSession):
        self.s = settings
        self.http = session
        self.feed = MarketFeed(settings)
        self.deriv = DerivativesData(settings, self.feed)
        self.news = NewsEngine(settings, session)
        self.analyst = ClaudeAnalyst(settings)
        self.claude_on = settings.claude_enabled and bool(settings.anthropic_api_key)
        self.risk = RiskManager(settings)
        self.tax = TaxExporter(settings, session)
        self.trader = HyperliquidTrader(settings, self.feed, self.tax)
        self.watchdog = Watchdog(settings, self.trader, self.risk, self.feed)
        self.tg = TelegramNotifier(settings, session)
        self.dashboard = Dashboard(settings.dashboard_host, settings.dashboard_port, self.dashboard_state)
        self.atrs: dict[str, float] = {}
        self.regimes: dict[str, str] = {}
        self.marks: dict[str, float] = {}
        self.equity: float = settings.initial_capital
        self.eur_per_usd: float = 0.0
        self.last_scan: str | None = None
        self._stop = asyncio.Event()
        self._entry_lock = asyncio.Lock()
        self._seen_path = settings.state_dir / "signals_seen.json"
        self._seen: dict[str, str] = self._load_json(self._seen_path, {})
        self._decisions_path = settings.state_dir / "decisions.jsonl"
        self._equity_path = settings.state_dir / "equity_history.csv"
        self._last_equity_write = 0.0
        self._claude_calls: list[float] = []
        self._bg: set[asyncio.Task] = set()
        self.stats = {"scans": 0, "signals": 0, "claude_calls": 0, "vetoes": 0, "trades": 0}

    @staticmethod
    def _load_json(path: Path, default):
        try:
            return json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            return default

    # ------------------------------------------------------------------ #
    # Bucle: watchlist
    # ------------------------------------------------------------------ #
    async def watchlist_loop(self) -> None:
        while not self._stop.is_set():
            try:
                wl = await self.deriv.build_watchlist()
                self.feed.set_ws_coins(sorted(set(wl) | set(self.trader.positions)))
                self.watchdog.record_ok("watchlist")
            except Exception as exc:  # noqa: BLE001
                log.exception("Error actualizando watchlist")
                await self._incident(self.watchdog.record_error("watchlist", exc))
            await self._sleep(self.s.watchlist_refresh_sec)

    # ------------------------------------------------------------------ #
    # Bucle: escaneo de señales
    # ------------------------------------------------------------------ #
    async def _analyze_coin(self, coin: str) -> Signal | None:
        try:
            df = await self.feed.candles(coin)
            closed = df.iloc[:-1]  # la última vela de Hyperliquid está en formación
            if len(closed) < 30:
                return None
            self.atrs[coin] = float(atr_fn(closed).iloc[-1])
            if len(closed) >= 260:
                self.regimes[coin] = regime(closed)
            return latest_signal(coin, closed, tuple(self.s.strategies))
        except Exception as exc:  # noqa: BLE001
            log.warning("Análisis %s falló: %s", coin, exc)
            return None

    async def scan_once(self, allow_trading: bool = True) -> list[Signal]:
        await self.deriv.refresh()
        coins = sorted(set(self.deriv.watchlist) | set(self.trader.positions))
        results = await asyncio.gather(*(self._analyze_coin(c) for c in coins))
        self.stats["scans"] += 1
        self.last_scan = datetime.now(timezone.utc).isoformat(timespec="seconds")
        fresh = [sig for sig in results if sig and self._seen.get(sig.coin) != str(sig.bar_time)]
        # Primero las señales en las que coinciden más estrategias
        fresh.sort(key=lambda sg: len(sg.strategies), reverse=True)
        for sig in fresh:
            log.info("SEÑAL %s %s [%s] vela %s", sig.coin, sig.side, ", ".join(sig.strategies), sig.bar_time.date())
        if allow_trading:
            for sig in fresh:
                if self._stop.is_set():
                    break
                await self.handle_signal(sig)
        return fresh

    async def scan_loop(self) -> None:
        while not self.deriv.watchlist and not self._stop.is_set():
            await asyncio.sleep(1)
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                await self.scan_once()
                self.watchdog.record_ok("scan")
            except Exception as exc:  # noqa: BLE001
                log.exception("Error en el escaneo")
                await self._incident(self.watchdog.record_error("scan", exc))
            await self._sleep(max(5.0, self.s.scan_interval_sec - (time.monotonic() - t0)))

    def _mark_seen(self, sig: Signal) -> None:
        self._seen[sig.coin] = str(sig.bar_time)
        self._seen_path.write_text(json.dumps(self._seen))

    def _record_decision(self, sig: Signal, outcome: str, claude: str | None = None) -> None:
        self.stats["signals"] += 1
        row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "coin": sig.coin, "side": sig.side,
               "strategies": sig.strategies, "outcome": outcome, "claude": claude}
        with self._decisions_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _claude_budget(self) -> bool:
        now = time.time()
        self._claude_calls = [t for t in self._claude_calls if now - t < 3600]
        return len(self._claude_calls) < self.s.max_claude_calls_per_hour

    async def handle_signal(self, sig: Signal) -> None:
        coin = sig.coin
        async with self._entry_lock:
            if coin in self.trader.positions:
                self._mark_seen(sig)
                return
            if self.watchdog.entries_paused:
                log.info("Señal %s en espera: entradas en pausa (%s)", coin, ", ".join(self.watchdog.pause_reasons))
                return  # no se marca como vista: se reintenta cuando se levante la pausa
            equity = await self.trader.equity(self.marks)
            ok, why = self.risk.can_open(self.trader.open_count, equity, self.trader.open_risk)
            if not ok:
                log.info("Señal %s descartada: %s", coin, why)
                self._mark_seen(sig)
                self._record_decision(sig, f"descartada: {why}")
                return

            mids = await self.feed.all_mids()
            self.marks.update(mids)
            price = mids.get(coin, sig.close)
            drift = abs(price - sig.close) / sig.atr
            if drift > self.s.max_entry_drift_atr:
                log.info("Señal %s caducada: el precio se alejó %.1f ATR del cierre", coin, drift)
                self._mark_seen(sig)
                self._record_decision(sig, f"caducada: precio a {drift:.1f} ATR de la señal")
                return

            deriv = self.deriv.snapshots.get(coin)
            is_long = sig.direction > 0
            stop_ref = price - sig.direction * sig.stop_k * sig.atr

            # --- Claude: supervisor con derecho de veto -------------------- #
            verdict: str | None = None
            if self.claude_on and self._claude_budget():
                self._claude_calls.append(time.time())
                self.stats["claude_calls"] += 1
                result = await self.analyst.analyze(
                    await self._build_context(sig, price, stop_ref, equity), [n.line() for n in await self.news.for_coin(coin)]
                )
                d = result.decision
                if result.error:
                    verdict = f"no disponible ({result.error})"
                    if self.s.claude_required:
                        self._record_decision(sig, "en espera: Claude no disponible", verdict)
                        return
                elif d.action == Action.NO_TRADE or (d.action == Action.LONG) != is_long:
                    self.stats["vetoes"] += 1
                    self._mark_seen(sig)
                    self._record_decision(sig, "vetada", d.thesis[:400])
                    self.tg.send(f"🚫 <b>Señal vetada por Claude</b>: {coin} {sig.side}\n<i>{d.thesis[:500]}</i>")
                    return
                else:
                    verdict = f"aprobada ({d.confidence} %): {d.thesis[:300]}"
            elif self.s.claude_required:
                self._record_decision(sig, "en espera: Claude no disponible")
                return

            sz_dec = await self.feed.sz_decimals(coin)
            max_lev = deriv.max_leverage if deriv else self.s.max_leverage
            plan, why = self.risk.build_plan(
                coin=coin, is_long=is_long, price=price, atr=sig.atr, proposed_stop=None, sz_decimals=sz_dec,
                equity=equity, open_risk=self.trader.open_risk, open_count=self.trader.open_count, max_asset_lev=max_lev,
            )
            self._mark_seen(sig)
            if plan is None:
                log.info("Plan de %s descartado: %s", coin, why)
                self._record_decision(sig, f"descartada: {why}", verdict)
                return
            event = await self.trader.open_position(plan, sz_dec, max_lev, 0, "+".join(sig.strategies))
            if event.kind == "entry":
                self.stats["trades"] += 1
                pos = event.data["pos"]
                self.feed.set_ws_coins(sorted(set(self.deriv.watchlist) | set(self.trader.positions)))
                self._record_decision(sig, "ejecutada", verdict)
                self.tg.entry(coin, pos.is_long, pos.size, pos.entry_px, pos.stop, pos.risk_usd,
                              "+".join(sig.strategies), verdict or "Sin revisión de Claude")
            else:
                self._record_decision(sig, f"error: {event.data.get('msg')}", verdict)
                self.tg.alert(f"{coin}: {event.data.get('msg')}")

    async def _build_context(self, sig: Signal, price: float, stop_ref: float, equity: float) -> dict:
        coin = sig.coin
        deriv = self.deriv.snapshots.get(coin)
        try:
            funding_24h = await self.deriv.funding_history(coin, 24)
        except Exception:  # noqa: BLE001
            funding_24h = []
        book = self.feed.books.get(coin)
        return {
            "signal": {
                "coin": coin, "direction": sig.side, "timeframe": "1d", "strategies": sig.strategies,
                "signal_bar_close_utc": str(sig.bar_time), "signal_close": sig.close, "price_now": price,
                "atr_daily": sig.atr, "system_stop": stop_ref, "trend_regime": self.regimes.get(coin),
            },
            "derivatives": None if not deriv else {
                "change_24h": round(deriv.change_24h, 5), "volume_24h_usd": round(deriv.day_ntl_vlm),
                "open_interest_usd": round(deriv.oi_usd), "oi_change_1h": self.deriv.oi_change(coin),
                "funding_hourly": deriv.funding_hourly, "funding_annualized": round(deriv.funding_annualized, 4),
                "funding_24h_avg": (sum(funding_24h) / len(funding_24h)) if funding_24h else None,
            },
            "orderbook": None if not book else {"spread_bps": book.spread_bps, "imbalance_1pct": round(book.imbalance(0.01), 3)},
            "market_breadth": {k: sum(1 for v in self.regimes.values() if v == k) for k in ("alcista", "bajista", "lateral")},
            "account": {
                "equity_usdc": round(equity, 2), "daily_drawdown": round(self.risk.daily_drawdown(equity), 4),
                "open_positions": [{"coin": p.coin, "side": "LONG" if p.is_long else "SHORT", "entry": p.entry_px}
                                   for p in self.trader.positions.values()],
            },
        }

    # ------------------------------------------------------------------ #
    # Bucle: posiciones, Capital Shield y vigilante
    # ------------------------------------------------------------------ #
    async def position_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.position_tick()
                self.watchdog.record_ok("posiciones")
            except Exception as exc:  # noqa: BLE001
                log.exception("Error en el bucle de posiciones")
                await self._incident(self.watchdog.record_error("posiciones", exc))
            await self._sleep(self.s.position_check_interval_sec)

    async def position_tick(self) -> None:
        self.marks.update(await self.feed.all_mids())
        self.watchdog.last_marks_ts = time.time()
        self.equity = await self.trader.equity(self.marks)
        for ev in self.risk.update_equity(self.equity):
            await self._handle_risk_event(ev, self.equity)
        fundings = {c: a.funding_hourly for c, a in self.deriv.snapshots.items()}
        for ev in await self.trader.manage(self.marks, self.atrs, fundings):
            self._notify_trade_event(ev)
        self._write_equity()

    async def watchdog_loop(self) -> None:
        while not self._stop.is_set():
            await self._sleep(self.s.watchdog_interval_sec)
            try:
                for inc in await self.watchdog.check(self.equity, self.marks):
                    await self._incident(inc)
            except Exception:  # noqa: BLE001
                log.exception("Error en el vigilante")

    async def _incident(self, inc: Incident | None) -> None:
        if inc is None:
            return
        icon = {"aviso": "⚠️", "grave": "🟠", "critico": "🔴"}.get(inc.severity, "⚠️")
        self.tg.send(f"{icon} <b>Incidente: {inc.kind}</b>\n{inc.detail}\n<b>Medida:</b> {inc.action}")
        if self.claude_on and inc.severity != "aviso":
            task = asyncio.create_task(self._diagnose(inc))
            self._bg.add(task)
            task.add_done_callback(self._bg.discard)

    async def _diagnose(self, inc: Incident) -> None:
        try:
            tail = "\n".join((self.s.log_dir / "hyperaegis.log").read_text(encoding="utf-8", errors="replace").splitlines()[-60:])
        except OSError:
            tail = ""
        text = await self.analyst.diagnose({"kind": inc.kind, "detail": inc.detail, "action": inc.action}, tail)
        if text:
            inc.diagnosis = text
            self.watchdog.save_diagnosis(inc)
            self.tg.send(f"🩺 <b>Diagnóstico de Claude</b> ({inc.kind})\n{text[:3500]}")

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
            await self._refresh_fx()

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
        st = self.risk.state
        total = equity - self.risk.initial
        self.tg.summary(
            "Resumen diario",
            [
                f"Equity: {equity:,.2f} USDC ({total:+.2f} / {total / self.risk.initial:+.2%} desde inicio)",
                f"Cierres de hoy: {len(rows)} | PnL neto realizado: {net:+.2f} USDC ({net_eur:+.2f} €)",
                f"Posiciones abiertas: {self.trader.open_count} | Riesgo abierto: {self.trader.open_risk:.2f} USDC",
                f"House Money: {'SÍ' if st.house_money_active else 'no'} | Pico equity: {st.peak_equity:,.2f}",
                f"Señales: {self.stats['signals']} | Vetos de Claude: {self.stats['vetoes']} | Operaciones: {self.stats['trades']}",
            ],
        )
        self.trader.closed_today.clear()

    # ------------------------------------------------------------------ #
    # Panel
    # ------------------------------------------------------------------ #
    def _write_equity(self) -> None:
        now = time.time()
        if now - self._last_equity_write < 900:
            return
        self._last_equity_write = now
        new = not self._equity_path.exists()
        with self._equity_path.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["time", "equity"])
            w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), round(self.equity, 4)])

    async def _refresh_fx(self) -> None:
        rate, _, src = await self.tax.usd_to_eur(datetime.now(timezone.utc))
        if rate and src == "BCE/Frankfurter":
            self.eur_per_usd = rate

    @staticmethod
    def _read_tail(path: Path, n: int) -> list[dict]:
        if not path.exists():
            return []
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as fh:
                return list(csv.DictReader(fh))[-n:]
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()[-n:] if line.strip()]

    def dashboard_state(self) -> dict:
        st = self.risk.state
        if st.halted:
            status = {"label": "DETENIDO: " + st.halt_reason, "level": "bad"}
        elif st.shield_breached:
            status = {"label": "Capital Shield activado", "level": "bad"}
        elif self.risk.circuit_breaker_active():
            status = {"label": "Circuit breaker (hasta mañana)", "level": "warn"}
        elif self.watchdog.entries_paused:
            status = {"label": "En pausa: " + ", ".join(self.watchdog.pause_reasons), "level": "warn"}
        else:
            status = {"label": "Operativo", "level": "ok"}
        positions = []
        for p in self.trader.positions.values():
            mark = self.marks.get(p.coin, p.entry_px)
            upnl = p.unrealized(mark)
            positions.append({"coin": p.coin, "side": "LONG" if p.is_long else "SHORT", "size": p.size, "entry": p.entry_px,
                              "mark": mark, "stop": p.stop, "upnl": upnl, "r": upnl / p.risk_usd if p.risk_usd else 0,
                              "opened": p.opened_at})
        trades = self._read_tail(self.s.tax_ledger_path, 5000)
        mode_trades = [t for t in trades if t.get("mode") == self.s.trading_mode.value]
        by_trade: dict[str, float] = {}
        for t in mode_trades:
            by_trade[t["trade_id"]] = by_trade.get(t["trade_id"], 0.0) + float(t["net_pnl_usd"])
        history = [(r["time"], float(r["equity"])) for r in self._read_tail(self._equity_path, 3000)]
        history.append((datetime.now(timezone.utc).isoformat(timespec="seconds"), self.equity))
        return {
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": self.s.trading_mode.value, "status": status, "house_money": st.house_money_active,
            "equity": self.equity, "initial_capital": self.risk.initial, "eur_per_usd": self.eur_per_usd,
            "day_pnl": self.equity - st.day_start_equity if st.day_start_equity else 0.0,
            "daily_drawdown": self.risk.daily_drawdown(self.equity), "daily_limit": self.s.daily_drawdown_limit_pct,
            "open_risk": self.trader.open_risk, "max_positions": self.s.max_open_positions,
            "positions": positions, "trades": list(reversed(mode_trades[-60:])),
            "stats": {"closed": len(by_trade), "net_pnl": sum(by_trade.values()),
                      "win_rate": (sum(1 for v in by_trade.values() if v > 0) / len(by_trade) * 100) if by_trade else 0.0},
            "equity_history": history,
            "decisions": list(reversed(self._read_tail(self._decisions_path, 25))),
            "incidents": list(reversed(self.watchdog.recent(15))),
            "market": {"total": len(self.regimes), "last_scan": self.last_scan,
                       **{k: sum(1 for v in self.regimes.values() if v == k) for k in ("alcista", "bajista", "lateral")}},
        }

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
        log.info("HyperAegis arrancando en modo %s (%s) | velas %s | Claude supervisor: %s",
                 self.s.trading_mode.value.upper(), "testnet" if self.s.use_testnet else "mainnet",
                 self.s.candle_interval, "sí" if self.claude_on else "NO (sin ANTHROPIC_API_KEY)")
        self.tg.start()
        if self.s.dashboard_enabled:
            await self.dashboard.start()
        await self.deriv.build_watchlist()
        self.feed.set_ws_coins(sorted(set(self.deriv.watchlist) | set(self.trader.positions)))
        for w in await self.trader.reconcile():
            self.tg.alert(w)
        await self.tax.backfill_pending()
        await self._refresh_fx()
        self.marks.update(await self.feed.all_mids())
        self.equity = await self.trader.equity(self.marks)
        self.risk.update_equity(self.equity)
        self.tg.startup(self.s.trading_mode.value, self.equity, self.deriv.watchlist,
                        self.s.claude_model if self.claude_on else "desactivado")

        tasks = [
            self.feed.start_ws(),
            asyncio.create_task(self.watchlist_loop(), name="watchlist"),
            asyncio.create_task(self.scan_loop(), name="scan"),
            asyncio.create_task(self.position_loop(), name="positions"),
            asyncio.create_task(self.watchdog_loop(), name="watchdog"),
        ]
        await self._stop.wait()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self.s.is_live and self.trader.positions:
            log.info("Las posiciones abiertas siguen protegidas por el stop en el exchange")
        self.tg.send("🛑 HyperAegis detenido.")
        await self.tg.stop()
        await self.dashboard.stop()


async def check(app: HyperAegis) -> None:
    """Diagnóstico sin operar."""
    s = app.s
    print(f"Modo: {s.trading_mode.value} | Red: {'testnet' if s.use_testnet else 'mainnet'} | Velas: {s.candle_interval} | "
          f"Claude: {s.claude_model if app.claude_on else 'desactivado (sin clave)'}")
    wl = await app.deriv.build_watchlist()
    print(f"Watchlist ({len(wl)}): {', '.join(wl)}")
    app.marks.update(await app.feed.all_mids())
    print(f"Equity ({s.trading_mode.value}): {await app.trader.equity(app.marks):.2f} USDC")
    signals = await app.scan_once(allow_trading=False)
    counts = {k: sum(1 for v in app.regimes.values() if v == k) for k in ("alcista", "bajista", "lateral")}
    print(f"Régimen de mercado: {counts}")
    print(f"Señales nuevas en la última vela diaria cerrada: {len(signals)}")
    for sg in signals:
        print(f"   {sg.coin:8s} {sg.side:5s} [{', '.join(sg.strategies)}] cierre {sg.close:g} ATR {sg.atr:g}")
    print(f"Titulares cargados: {len(await app.news.latest())}")
    rate, day, src = await app.tax.usd_to_eur(datetime.now(timezone.utc))
    print(f"USD->EUR {rate} ({day}, {src})")


async def main() -> None:
    parser = argparse.ArgumentParser(description="HyperAegis trading bot")
    parser.add_argument("--check", action="store_true", help="diagnóstico sin operar")
    parser.add_argument("--reset-shield", action="store_true", help="rearmar el Capital Shield / kill switch")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings)

    if args.reset_shield:
        RiskManager(settings).reset_shield()
        print("Capital Shield y kill switch rearmados.")
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
