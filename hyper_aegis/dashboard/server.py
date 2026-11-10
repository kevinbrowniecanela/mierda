"""Panel web local (solo lectura) de HyperAegis.

Sirve una página en http://127.0.0.1:8787 con el estado del bot: equity,
posiciones abiertas, operaciones cerradas, decisiones de Claude e incidentes.
No expone ninguna acción: desde el panel no se puede operar ni mover dinero.
Por defecto solo escucha en localhost.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from aiohttp import web

log = logging.getLogger(__name__)
HTML = Path(__file__).with_name("index.html")


class Dashboard:
    def __init__(self, host: str, port: int, state_fn: Callable[[], dict]):
        self.host, self.port, self.state_fn = host, port, state_fn
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/api/state", self._state)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        try:
            await web.TCPSite(self._runner, self.host, self.port).start()
            log.info("Panel disponible en http://%s:%d", self.host, self.port)
        except OSError as exc:
            log.warning("No se pudo abrir el panel en el puerto %d: %s", self.port, exc)

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    async def _index(self, _: web.Request) -> web.Response:
        return web.Response(text=HTML.read_text(encoding="utf-8"), content_type="text/html")

    async def _state(self, _: web.Request) -> web.Response:
        return web.json_response(self.state_fn(), headers={"Cache-Control": "no-store"})
