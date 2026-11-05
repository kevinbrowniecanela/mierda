# 🛡️ HyperAegis

Bot de trading algorítmico para **Futuros Perpetuos en Hyperliquid**, con arquitectura de 2 fases:

1. **Fase 1, filtro cuantitativo** (Python, sin coste de IA): escanea cada 60 s la watchlist dinámica
   (Top 20 por volumen + Top 5 movers) y puntúa anomalías: spike de volumen, expansión de rango vs ATR,
   cambio de Open Interest, funding extremo, desequilibrio del libro L2, RSI extremo y cruce de VWAP.
2. **Fase 2, analista Claude**: solo cuando la puntuación supera el umbral. Claude recibe técnicos,
   derivados, libro, noticias y estado de la cuenta, y devuelve una decisión en JSON validada con
   *structured outputs* + Pydantic. Cualquier respuesta inválida se convierte en `NO_TRADE`.

> ⚠️ **Software experimental. El trading con apalancamiento puede hacerte perder todo el capital.**
> Empieza siempre en `TRADING_MODE=paper` y después en `USE_TESTNET=true` antes de pasar a real.

## Estructura

```
hyper_aegis/
├── config/        settings.py (.env + parámetros), prompts.py (system prompt)
├── data/          market_feed.py (REST + WebSocket L2), derivatives_data.py (OI, funding, watchlist), news_engine.py
├── quant/         technicals.py (EMA20/50/200, RSI, ATR, VWAP), anomaly_trigger.py (Fase 1)
├── brain/         claude_analyst.py (Fase 2), schemas.py (TradeDecision)
├── execution/     risk_manager.py (Capital Shield), hyperliquid_trader.py (paper/live), tax_exporter.py (CSV en €)
├── alerts/        telegram_notifier.py
├── tests/         tests unitarios y de ciclo de vida (pytest)
└── run.py         orquestador asyncio
```

## Reglas de riesgo implementadas

| Regla | Implementación |
|---|---|
| Riesgo por operación | 1 % del capital **inicial** (400 USDC → 4 USDC), incluyendo comisiones |
| Tamaño | `riesgo / distancia al stop`; stop de Claude si está entre 0.5 y 3 ATR, si no 1.5 ATR |
| TP1 | R:R 1:1.5 → cierra 50 % y mueve el SL a breakeven |
| Resto | Trailing stop a 2× ATR desde el extremo favorable (solo mejora, nunca retrocede) |
| House Money | Al llegar a +15 % se activa y queda fijado: el riesgo abierto total nunca puede superar el colchón de ganancias (`equity − depósito`). Si el equity toca el depósito, cierra todo y se detiene (`run.py --reset-shield` para rearmar) |
| Límites | Máximo 3 posiciones, apalancamiento máximo configurable (5x por defecto) y nocional mínimo de 10 USD |
| Circuit breaker | Drawdown diario (UTC) > 3 % → sin nuevas entradas hasta el día siguiente (y cierre total si `FLATTEN_ON_CIRCUIT_BREAKER=true`) |
| Protección en exchange | En LIVE, SL y TP1 son órdenes *trigger reduce-only* residentes en Hyperliquid: la posición sigue protegida aunque el Mac se apague. Al mover el stop se coloca el nuevo antes de cancelar el anterior |

## Instalación en macOS

```bash
brew install python@3.12        # recomendado (pandas-ta se instala sin problemas en 3.12)
cd hyper_aegis
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # y rellena tus claves
pytest                          # 18 tests
python run.py --check           # diagnóstico: conectividad, watchlist, escaneo sin operar, tipo €
python run.py                   # arranca (paper por defecto)
```

Para evitar que el Mac entre en reposo mientras corre el bot:

```bash
caffeinate -i python run.py
```

### Agent Wallet (obligatorio en LIVE)

En app.hyperliquid.xyz → *More → API* crea una **API Wallet** y autorízala. Pon su clave privada en
`HL_AGENT_PRIVATE_KEY` y la dirección de tu cuenta principal en `HL_ACCOUNT_ADDRESS`. Esa clave solo
puede firmar órdenes: **no puede retirar fondos**. Nunca pongas la clave de tu wallet principal.

### Ejecutarlo como servicio (launchd)

`~/Library/LaunchAgents/com.hyperaegis.bot.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.hyperaegis.bot</string>
  <key>ProgramArguments</key><array>
    <string>/usr/bin/caffeinate</string><string>-i</string>
    <string>/RUTA/hyper_aegis/.venv/bin/python</string><string>/RUTA/hyper_aegis/run.py</string>
  </array>
  <key>WorkingDirectory</key><string>/RUTA/hyper_aegis</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardErrorPath</key><string>/RUTA/hyper_aegis/logs/launchd.err</string>
</dict></plist>
```

`launchctl load ~/Library/LaunchAgents/com.hyperaegis.bot.plist`

## Modelo de IA

Pediste `claude-3-5-sonnet`, pero Anthropic **retiró ese modelo en octubre de 2025** y la API ya no lo
sirve. El bot usa por defecto su sucesor, `claude-sonnet-5-5` (configurable con `CLAUDE_MODEL`), con:

- *structured outputs* (`output_config.format`) + re-validación Pydantic,
- `effort` configurable (`CLAUDE_EFFORT`, `high` por defecto),
- prompt caching del system prompt,
- *fallback* del lado del servidor ante `stop_reason: "refusal"` (`fallbacks: "default"`, beta
  `server-side-fallback-2026-07-01`). Se desactiva con `CLAUDE_ENABLE_FALLBACKS=false`.

El coste queda acotado por `MAX_CLAUDE_CALLS_PER_HOUR` y por un cooldown de 45 min por activo.

## Contabilidad fiscal (España)

`tax_ledger.csv` registra **cada cierre** (el 50 % de TP1 y el resto por separado) con: fechas UTC al
segundo, tamaño, precios, PnL bruto, comisiones, funding, PnL neto, valor de adquisición/transmisión y
PnL en € con el tipo de referencia del **BCE** (vía Frankfurter).

Notas importantes:
- El BCE publica **un tipo de referencia por día hábil** (~16:00 CET); no existe un tipo oficial "al
  segundo". Se registra el timestamp exacto del cierre y el tipo BCE de esa fecha (o el último
  publicado si es festivo o fin de semana), indicando la fecha del tipo aplicado (`rate_date`).
- Se asume USDC = USD.
- Si la API de tipos falla, la fila queda marcada como `PENDIENTE` y se corrige automáticamente al
  arrancar y en cada cambio de día.
- Esto es un registro de apoyo, no asesoramiento fiscal: revísalo con tu asesor (tributación de
  derivados, compensación de pérdidas, modelo 721, etc.).

## Telegram

Crea un bot con @BotFather, pon el token en `TELEGRAM_BOT_TOKEN` y tu chat id en `TELEGRAM_CHAT_ID`.
Recibirás: arranque, entradas (con tesis de Claude), TP1, cierres (USDC y €), House Money, circuit
breaker, Capital Shield y un resumen diario.

## Estado persistente

`state/` guarda el estado del Capital Shield, las posiciones gestionadas y la cuenta paper, de modo
que un reinicio no pierde el seguimiento. En LIVE, al arrancar se concilia con el exchange: las
posiciones cerradas mientras el bot estaba parado se registran en el ledger, y se avisa de las
posiciones abiertas manualmente que el bot no gestiona.
