"""Notificaciones en tiempo real vía Bot API de Telegram.

Envío asíncrono no bloqueante mediante una cola: un fallo de Telegram nunca
detiene ni retrasa la ejecución de órdenes.
"""

from __future__ import annotations

import asyncio
import html
import logging

import aiohttp

from config.settings import Settings

log = logging.getLogger(__name__)


def esc(text: object) -> str:
    return html.escape(str(text), quote=False)


class TelegramNotifier:
    def __init__(self, settings: Settings, session: aiohttp.ClientSession):
        self.s = settings
        self.http = session
        self.enabled = bool(settings.telegram_bot_token and settings.telegram_chat_id)
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=500)
        self._task: asyncio.Task | None = None
        if not self.enabled:
            log.info("Telegram desactivado (faltan TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)")

    def start(self) -> asyncio.Task | None:
        if self.enabled and self._task is None:
            self._task = asyncio.create_task(self._worker(), name="telegram")
        return self._task

    async def stop(self) -> None:
        if self._task:
            try:
                await asyncio.wait_for(self._queue.join(), timeout=10)
            except asyncio.TimeoutError:
                pass
            self._task.cancel()

    def send(self, text: str) -> None:
        """Encola un mensaje HTML (no bloqueante)."""
        log.info("[TG] %s", text.replace("\n", " | "))
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(text)
        except asyncio.QueueFull:
            log.warning("Cola de Telegram llena; mensaje descartado")

    async def _worker(self) -> None:
        url = f"https://api.telegram.org/bot{self.s.telegram_bot_token}/sendMessage"
        while True:
            text = await self._queue.get()
            try:
                for attempt in range(4):
                    try:
                        async with self.http.post(
                            url,
                            json={"chat_id": self.s.telegram_chat_id, "text": text[:4000], "parse_mode": "HTML", "disable_web_page_preview": True},
                            timeout=aiohttp.ClientTimeout(total=15),
                        ) as resp:
                            if resp.status == 429:
                                retry = (await resp.json()).get("parameters", {}).get("retry_after", 5)
                                await asyncio.sleep(retry)
                                continue
                            if resp.status >= 400:
                                log.warning("Telegram %s: %s", resp.status, await resp.text())
                            break
                    except Exception as exc:  # noqa: BLE001
                        log.warning("Telegram error (%s), intento %d", exc, attempt + 1)
                        await asyncio.sleep(2**attempt)
                await asyncio.sleep(0.05)  # respeta límites de la API
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------ #
    # Plantillas
    # ------------------------------------------------------------------ #
    def startup(self, mode: str, equity: float, watchlist: list[str], model: str) -> None:
        self.send(
            f"🛡️ <b>HyperAegis iniciado</b>\nModo: <b>{esc(mode.upper())}</b>\n"
            f"Equity: {equity:,.2f} USDC\nModelo IA: {esc(model)}\n"
            f"Watchlist ({len(watchlist)}): {esc(', '.join(watchlist))}"
        )

    def entry(self, coin: str, is_long: bool, size: float, price: float, stop: float, tp1: float,
              risk_usd: float, confidence: int, setup: str, thesis: str) -> None:
        arrow = "🟢 LONG" if is_long else "🔴 SHORT"
        self.send(
            f"{arrow} <b>{esc(coin)}</b>\nEntrada: {price:g} | Tamaño: {size:g}\n"
            f"SL: {stop:g} | TP1 (1.5R): {tp1:g}\nRiesgo: {risk_usd:.2f} USDC | Conf: {confidence}% | {esc(setup)}\n"
            f"<i>{esc(thesis[:600])}</i>"
        )

    def tp1(self, coin: str, closed_size: float, price: float, pnl_usd: float, new_stop: float) -> None:
        self.send(
            f"🎯 <b>TP1 alcanzado {esc(coin)}</b>\nCerrado {closed_size:g} @ {price:g} ({pnl_usd:+.2f} USDC)\n"
            f"SL movido a breakeven: {new_stop:g}. Resto con trailing 2x ATR."
        )

    def exit(self, coin: str, reason: str, price: float, pnl_usd: float, pnl_eur: float) -> None:
        icon = "✅" if pnl_usd >= 0 else "❌"
        self.send(f"{icon} <b>Cierre {esc(coin)}</b> ({esc(reason)})\nPrecio: {price:g}\nPnL: {pnl_usd:+.2f} USDC ({pnl_eur:+.2f} €)")

    def alert(self, text: str) -> None:
        self.send(f"⚠️ {esc(text)}")

    def summary(self, title: str, lines: list[str]) -> None:
        self.send(f"📊 <b>{esc(title)}</b>\n" + "\n".join(esc(line) for line in lines))
