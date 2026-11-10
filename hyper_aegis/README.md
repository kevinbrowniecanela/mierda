# 🛡️ HyperAegis

Bot de trading algorítmico para **Futuros Perpetuos en Hyperliquid** (cripto y, en el mismo exchange,
oro, índices y acciones). Opera en **velas diarias**:

1. **Sistema cuantitativo** (`quant/strategies.py`, sin coste de IA): seguimiento de tendencia
   (rupturas de canal Donchian 20/55 y rupturas tras compresión de volatilidad, a favor de la
   tendencia de fondo) sobre la watchlist dinámica (Top 20 por volumen + Top 5 movers).
2. **Claude como supervisor**: revisa cada señal con noticias, funding, Open Interest y estado de la
   cuenta, y puede **vetarla**. No inventa operaciones ni cambia el stop. Si no hay clave de Anthropic
   el bot opera solo con el sistema cuantitativo.
3. **Vigilante** (`execution/watchdog.py`): detecta fallos y corta fugas de dinero (ver abajo).
4. **Panel web local** de solo lectura en http://127.0.0.1:8787.

> ⚠️ **Software experimental. El trading con apalancamiento puede hacerte perder todo el capital.**
> Empieza siempre en `TRADING_MODE=paper` y después en `USE_TESTNET=true` antes de pasar a real.

## Por qué velas diarias

`research/lab.py` probó 11 estrategias en 1h / 4h / 1d sobre 33 mercados con costes reales y validación
fuera de muestra. En 15 min y 1 h **todo pierde** (los costes se comen la señal). En diario, toda la
familia de seguimiento de tendencia es positiva dentro y fuera de muestra en 5,6 años.

Prueba con 200 € (`python research/monthly.py --eur 200`), 3 posiciones máximas:

| Riesgo/operación | Anual | Mes medio | Mes mediano | Peor mes | Caída máx. |
|---|---|---|---|---|---|
| 1 % (por defecto) | 19 % | +1,6 % | 0 % | −4,8 % | 15 % |
| 2 % | 37 % | +3,2 % | 0 % | −9,5 % | 28 % |
| 3 % | 53 % | +4,7 % | 0 % | −14 % | 39 % |

Gana ~38 % de las operaciones; el beneficio viene de pocas tendencias largas, así que habrá muchos
meses planos o negativos. Las cifras son de backtest (universo actual de activos, sin Claude) y el
resultado real será previsiblemente peor.

## Estructura

```
hyper_aegis/
├── config/        settings.py (.env + parámetros), prompts.py (supervisor y diagnóstico)
├── data/          market_feed.py, derivatives_data.py (OI, funding, watchlist), news_engine.py
├── quant/         strategies.py (señales en vivo), technicals.py y anomaly_trigger.py (legado 15m)
├── brain/         claude_analyst.py (veto + diagnóstico), schemas.py
├── execution/     risk_manager.py, hyperliquid_trader.py (paper/live), watchdog.py, tax_exporter.py
├── dashboard/     server.py + index.html (panel local)
├── alerts/        telegram_notifier.py
├── research/      download.py, lab.py (laboratorio), monthly.py (retorno mensual)
├── tests/         23 tests (pytest)
└── run.py         orquestador asyncio
```

## Reglas de riesgo

| Regla | Implementación |
|---|---|
| Riesgo por operación | 1 % del **capital actual** (interés compuesto; `RISK_BASIS=initial` para fijo) |
| Stop inicial | 2 × ATR diario, colocado como orden en el exchange |
| Salida | Trailing stop a 3 × ATR desde el extremo favorable (`EXIT_MODE=tp_partial` recupera TP1 50 % + breakeven) |
| Entrada tardía | No entra si el precio ya se alejó más de 1 ATR del cierre de la señal |
| House Money | A +15 % se activa: el riesgo abierto nunca supera las ganancias acumuladas |
| Límites | Máximo 3 posiciones, apalancamiento máximo 5x, nocional mínimo 10 USD |
| Circuit breaker | Caída diaria > 3 % → cierra todo y no abre más hasta el día siguiente |

## Vigilante: qué fallos corrige solo

| Fallo detectado | Medida automática |
|---|---|
| Un bucle falla 3 veces seguidas | Pausa nuevas entradas hasta que se recupere |
| Precios sin actualizar > 5 min | Pausa nuevas entradas (se reanuda solo) |
| Posición real sin stop en el exchange | Repone el stop; si no puede, cierra la posición |
| Posición en el exchange que el bot no gestiona | Aviso |
| Caída brusca de capital mayor que el riesgo abierto | **Kill switch**: cierra todo y se detiene (`run.py --reset-shield`) |

En cada incidente grave Claude recibe el incidente y el log y envía un diagnóstico por Telegram.
Claude **no** modifica el código ni ejecuta órdenes por su cuenta: las medidas son siempre fijas y
conservadoras. Con `KeepAlive` de launchd, si el proceso muere se reinicia solo y las posiciones
siguen protegidas por el stop del exchange.

## ¿Dónde está el dinero?

En tu cuenta de **Hyperliquid**. El bot no custodia fondos: firma órdenes con una Agent Wallet que
**no puede retirar**. Depositas y retiras tú desde app.hyperliquid.xyz. El panel es solo de lectura.

## Instalación en macOS

```bash
brew install python@3.12        # recomendado (pandas-ta se instala sin problemas en 3.12)
cd hyper_aegis
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # y rellena tus claves
pytest                          # 23 tests
python run.py --check           # diagnóstico: conectividad, régimen de mercado, señales de hoy, tipo €
python run.py                   # arranca (paper por defecto) y abre el panel en http://127.0.0.1:8787
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

`claude-3-5-sonnet` está retirado; el bot usa `claude-sonnet-5-5` (configurable con `CLAUDE_MODEL`) con
salida estructurada validada con Pydantic, prompt caching y fallback ante `refusal`. Con velas diarias
hay pocas señales, así que el coste de Claude es de céntimos al mes. `CLAUDE_REQUIRED=true` hace que
no se opere si Claude no responde.

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
Recibirás: arranque, entradas (con el veredicto de Claude), vetos, cierres (USDC y €), incidentes con
diagnóstico, House Money, circuit breaker y un resumen diario.

## Estado persistente

`state/` guarda el estado del Capital Shield, las posiciones gestionadas y la cuenta paper, de modo
que un reinicio no pierde el seguimiento. En LIVE, al arrancar se concilia con el exchange: las
posiciones cerradas mientras el bot estaba parado se registran en el ledger, y se avisa de las
posiciones abiertas manualmente que el bot no gestiona.
