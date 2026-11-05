"""Registro contable fiscal (España) con conversión automática a euros.

Cada cierre (total o parcial, p. ej. el 50 % en TP1) es una transmisión y se
registra como una fila independiente de `tax_ledger.csv`.

Tipo de cambio: tipo de referencia oficial del BCE vía Frankfurter
(`/v1/{fecha}?from=USD&to=EUR`). El BCE publica UN tipo de referencia por día
hábil (~16:00 CET); no existe un tipo oficial "al segundo". Se registra el
timestamp exacto del cierre y el tipo BCE vigente para esa fecha (si es
festivo o fin de semana, Frankfurter devuelve el último publicado), indicando
en la fila la fecha del tipo aplicado. Se asume USDC = USD a efectos de
valoración.
"""

from __future__ import annotations

import asyncio
import csv
import logging
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

from config.settings import Settings

log = logging.getLogger(__name__)


@dataclass
class LedgerRow:
    trade_id: str
    coin: str
    side: str  # LONG | SHORT
    open_time_utc: str
    close_time_utc: str
    size: float
    entry_px_usd: float
    exit_px_usd: float
    gross_pnl_usd: float
    fees_usd: float
    funding_usd: float  # positivo = cobrado, negativo = pagado
    net_pnl_usd: float
    usd_eur_rate: float  # EUR por 1 USD
    rate_date: str
    rate_source: str
    acquisition_value_eur: float
    transmission_value_eur: float
    net_pnl_eur: float
    close_reason: str
    mode: str


LEDGER_FIELDS = [f.name for f in fields(LedgerRow)]


class TaxExporter:
    def __init__(self, settings: Settings, session: aiohttp.ClientSession):
        self.s = settings
        self.http = session
        self.path: Path = settings.tax_ledger_path
        self._rates: dict[str, tuple[float, str]] = {}
        self._lock = asyncio.Lock()

    async def usd_to_eur(self, when: datetime) -> tuple[float, str, str]:
        """Devuelve (tipo EUR/USD, fecha del tipo, fuente)."""
        day = when.astimezone(timezone.utc).strftime("%Y-%m-%d")
        if day in self._rates:
            rate, rate_date = self._rates[day]
            return rate, rate_date, "BCE/Frankfurter"
        url = f"{self.s.fx_api_url.rstrip('/')}/{day}"
        for attempt in range(3):
            try:
                async with self.http.get(url, params={"base": "USD", "symbols": "EUR"}, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                rate = float(data["rates"]["EUR"])
                rate_date = data.get("date", day)
                self._rates[day] = (rate, rate_date)
                return rate, rate_date, "BCE/Frankfurter"
            except Exception as exc:  # noqa: BLE001
                log.warning("Tipo de cambio %s falló (%s), intento %d", day, exc, attempt + 1)
                await asyncio.sleep(2**attempt)
        # Último recurso: tipo más reciente en caché, marcado como provisional
        if self._rates:
            last_day = max(self._rates)
            rate, rate_date = self._rates[last_day]
            return rate, rate_date, "PROVISIONAL(cache)"
        return 0.0, "", "PENDIENTE"

    async def record_close(
        self,
        *,
        trade_id: str,
        coin: str,
        is_long: bool,
        open_time: datetime,
        close_time: datetime,
        size: float,
        entry_px: float,
        exit_px: float,
        fees_usd: float,
        funding_usd: float,
        close_reason: str,
    ) -> LedgerRow:
        direction = 1 if is_long else -1
        gross = (exit_px - entry_px) * size * direction
        net = gross - fees_usd + funding_usd
        rate, rate_date, source = await self.usd_to_eur(close_time)
        # Valor de adquisición / transmisión (criterio: compra = adquisición)
        buy_value = (entry_px if is_long else exit_px) * size
        sell_value = (exit_px if is_long else entry_px) * size
        row = LedgerRow(
            trade_id=trade_id,
            coin=coin,
            side="LONG" if is_long else "SHORT",
            open_time_utc=open_time.astimezone(timezone.utc).isoformat(timespec="seconds"),
            close_time_utc=close_time.astimezone(timezone.utc).isoformat(timespec="seconds"),
            size=size,
            entry_px_usd=round(entry_px, 8),
            exit_px_usd=round(exit_px, 8),
            gross_pnl_usd=round(gross, 6),
            fees_usd=round(fees_usd, 6),
            funding_usd=round(funding_usd, 6),
            net_pnl_usd=round(net, 6),
            usd_eur_rate=rate,
            rate_date=rate_date,
            rate_source=source,
            acquisition_value_eur=round((buy_value + fees_usd) * rate, 2),
            transmission_value_eur=round((sell_value + funding_usd) * rate, 2),
            net_pnl_eur=round(net * rate, 2),
            close_reason=close_reason,
            mode=self.s.trading_mode.value,
        )
        await self._append(row)
        return row

    async def backfill_pending(self) -> int:
        """Recalcula los importes en EUR de las filas registradas sin tipo oficial
        (API caída: fuente PENDIENTE o PROVISIONAL). Devuelve filas corregidas."""
        if not self.path.exists():
            return 0
        async with self._lock:
            rows = await asyncio.to_thread(self._read_rows)
            fixed = 0
            for r in rows:
                if r["rate_source"] not in ("PENDIENTE", "PROVISIONAL(cache)"):
                    continue
                close_dt = datetime.fromisoformat(r["close_time_utc"])
                rate, rate_date, source = await self.usd_to_eur(close_dt)
                if source != "BCE/Frankfurter":
                    continue
                size, entry, exit_ = float(r["size"]), float(r["entry_px_usd"]), float(r["exit_px_usd"])
                fees, funding, net = float(r["fees_usd"]), float(r["funding_usd"]), float(r["net_pnl_usd"])
                is_long = r["side"] == "LONG"
                buy_value = (entry if is_long else exit_) * size
                sell_value = (exit_ if is_long else entry) * size
                r.update(
                    usd_eur_rate=rate, rate_date=rate_date, rate_source=source,
                    acquisition_value_eur=round((buy_value + fees) * rate, 2),
                    transmission_value_eur=round((sell_value + funding) * rate, 2),
                    net_pnl_eur=round(net * rate, 2),
                )
                fixed += 1
            if fixed:
                await asyncio.to_thread(self._rewrite, rows)
                log.info("Ledger fiscal: %d filas actualizadas con tipo BCE oficial", fixed)
            return fixed

    def _read_rows(self) -> list[dict]:
        with self.path.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    def _rewrite(self, rows: list[dict]) -> None:
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(self.path)

    async def _append(self, row: LedgerRow) -> None:
        async with self._lock:
            await asyncio.to_thread(self._write_row, row)

    def _write_row(self, row: LedgerRow) -> None:
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(asdict(row))
