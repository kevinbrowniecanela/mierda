"""Vigilante (watchdog) de HyperAegis: detecta fallos y corta fugas de dinero.

Las medidas correctivas son SIEMPRE deterministas y conservadoras (reponer un
stop, pausar entradas, cerrar todo); Claude no ejecuta acciones ni modifica
código: recibe el incidente y redacta el diagnóstico para el dueño. Un sistema
que se reescribe a sí mismo con dinero real en juego es más peligroso que el
fallo que intenta arreglar.

Comprobaciones:
  1. Bucles que fallan repetidamente      -> pausa nuevas entradas.
  2. Datos de mercado obsoletos            -> pausa nuevas entradas (se reanuda solo).
  3. Posición sin stop en el exchange      -> repone el stop; si no puede, cierra.
  4. Posición en el exchange no gestionada -> aviso.
  5. Fuga: caída brusca de equity mayor que el riesgo abierto -> KILL SWITCH
     (cierra todo y detiene el bot hasta rearme manual `run.py --reset-shield`).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Incident:
    kind: str
    severity: str  # "aviso" | "grave" | "critico"
    detail: str
    action: str
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    diagnosis: str | None = None


class Watchdog:
    def __init__(self, settings, trader, risk, feed):
        self.s = settings
        self.trader = trader
        self.risk = risk
        self.feed = feed
        self.path: Path = settings.state_dir / "incidents.jsonl"
        self._errors: dict[str, int] = {}
        self._last_ok: dict[str, float] = {}
        self._pause_reasons: set[str] = set()
        self._last_equity: float | None = None
        self._alerted: set[str] = set()
        self.last_marks_ts: float = time.time()

    # ------------------------------------------------------------------ #
    @property
    def entries_paused(self) -> bool:
        return bool(self._pause_reasons)

    @property
    def pause_reasons(self) -> list[str]:
        return sorted(self._pause_reasons)

    def record_ok(self, loop: str) -> None:
        self._errors[loop] = 0
        self._last_ok[loop] = time.time()
        self._pause_reasons.discard(f"bucle {loop} fallando")

    def record_error(self, loop: str, exc: BaseException) -> Incident | None:
        n = self._errors.get(loop, 0) + 1
        self._errors[loop] = n
        if n == self.s.watchdog_max_loop_errors:
            self._pause_reasons.add(f"bucle {loop} fallando")
            return self._log(Incident("bucle_fallando", "grave", f"El bucle '{loop}' ha fallado {n} veces seguidas: {exc!r}",
                                      "Nuevas entradas en pausa hasta que el bucle se recupere"))
        return None

    def _log(self, inc: Incident) -> Incident:
        log.error("INCIDENTE [%s/%s] %s -> %s", inc.kind, inc.severity, inc.detail, inc.action)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(inc), ensure_ascii=False) + "\n")
        return inc

    def save_diagnosis(self, inc: Incident) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(inc), ensure_ascii=False) + "\n")

    def recent(self, n: int = 20) -> list[dict]:
        if not self.path.exists():
            return []
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()[-200:] if line.strip()]
        latest: dict[tuple, dict] = {}
        for r in rows:  # la versión con diagnóstico sustituye a la original
            latest[(r["ts"], r["kind"])] = r
        return list(latest.values())[-n:]

    # ------------------------------------------------------------------ #
    async def check(self, equity: float, marks: dict[str, float]) -> list[Incident]:
        incidents: list[Incident] = []

        # 2) Datos obsoletos
        stale = time.time() - self.last_marks_ts > self.s.watchdog_stale_data_sec
        key = "datos de mercado obsoletos"
        if stale and key not in self._pause_reasons:
            self._pause_reasons.add(key)
            incidents.append(self._log(Incident("datos_obsoletos", "grave",
                                                f"Sin precios nuevos desde hace {time.time() - self.last_marks_ts:.0f}s",
                                                "Nuevas entradas en pausa; los stops del exchange siguen activos")))
        elif not stale:
            self._pause_reasons.discard(key)

        # 5) Fuga de dinero: el equity cae más de lo que explican los stops
        if self._last_equity and self._last_equity > 0 and not self.risk.state.halted:
            drop = (self._last_equity - equity) / self._last_equity
            limit = max(self.s.watchdog_leak_pct, 2 * self.trader.open_risk / self._last_equity)
            if drop > limit:
                reason = f"Caída anómala de equity {drop:.1%} entre chequeos (límite {limit:.1%})"
                self.risk.halt(reason)
                closed = await self.trader.close_all(marks, "kill_switch")
                incidents.append(self._log(Incident("fuga_equity", "critico", reason,
                                                    f"KILL SWITCH: {len(closed)} posiciones cerradas y bot detenido (rearme: run.py --reset-shield)")))
        self._last_equity = equity

        # 3 y 4) Solo en real: protección en el exchange y posiciones huérfanas
        if self.s.is_live:
            incidents += await self._check_exchange(marks)
        return incidents

    async def _check_exchange(self, marks: dict[str, float]) -> list[Incident]:
        out: list[Incident] = []
        addr = self.s.hl_account_address
        orders = await self.feed.call("open_orders", addr)
        resting = {int(o["oid"]) for o in orders or []}
        state = await self.feed.call("user_state", addr)
        on_exchange = {p["position"]["coin"]: float(p["position"]["szi"]) for p in state.get("assetPositions", [])}

        for coin, pos in list(self.trader.positions.items()):
            if abs(on_exchange.get(coin, 0.0)) == 0:
                continue  # el gestor de posiciones registrará el cierre
            if pos.sl_oid is None or pos.sl_oid not in resting:
                try:
                    await self.trader.repair_stop(pos)
                    out.append(self._log(Incident("stop_ausente", "grave", f"{coin}: la posición no tenía stop en el exchange",
                                                  f"Stop repuesto en {pos.stop:g}")))
                except Exception as exc:  # noqa: BLE001
                    await self.trader.close_all({coin: marks.get(coin, pos.entry_px)}, "sin_proteccion", only=coin)
                    out.append(self._log(Incident("stop_ausente", "critico",
                                                  f"{coin}: sin stop y no se pudo reponer ({exc!r})", "Posición cerrada a mercado")))
        for coin, szi in on_exchange.items():
            if szi != 0 and coin not in self.trader.positions and coin not in self._alerted:
                self._alerted.add(coin)
                out.append(self._log(Incident("posicion_no_gestionada", "aviso",
                                              f"{coin}: posición {szi:g} en el exchange que el bot no gestiona",
                                              "Ninguna (revísala manualmente)")))
        return out
