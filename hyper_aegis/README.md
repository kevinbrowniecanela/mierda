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

## Estado de la validación (léelo antes de operar)

**El sistema diario actual NO tiene una ventaja demostrada.** El primer estudio (`research/lab.py`) daba
~19 % anual, pero la validación dura (`research/validate.py`) lo desmonta:

| Prueba | Expectativa por operación | Anual (1 % riesgo) |
|---|---|---|
| Top 20 de hoy, 5,6 años (primer estudio, sesgado) | +0,17R | +18,8 % |
| Top 20 de hoy, solo desde que existe Hyperliquid (may-2023) | +0,08R | +9,6 % |
| Top 20 de cada momento, con deslistados y **funding real** | **−0,03R** | +4,1 % |

- Intervalo del 95 % de la expectativa: [−0,16R, +0,11R]; 69 % de probabilidad de que sea ≤ 0.
- Subir el riesgo no lo arregla: 5 % por operación da ~7 % anual con caída máxima del 67 %.
- El funding real cuesta más que el supuesto fijo: 0,05R por operación (los largos pagan 0,09R).
- En 15 min y 1 h todo pierde por costes.
- Referencia desde mayo de 2023, comprar y mantener sin apalancamiento: BTC +34 % anual (caída 53 %),
  oro PAXG +20 % anual (caída 30 %).
- Única pista positiva, sin significación estadística: rotar estrategias eligiéndolas con los 12 meses
  anteriores (+0,06R, +13 % anual, caída 5 % en 2,5 años y solo 5 ventanas).

Usa el bot en `TRADING_MODE=paper` hasta que haya un sistema que supere esta validación.
`research/next_steps.py` reproduce el análisis de riesgo, variantes y aportaciones periódicas.

## Acciones de EE. UU. frente a cripto (estudio)

`research/stocks_download.py`, `stocks_study.py` y `stocks_followup.py` aplican el mismo sistema diario a
124 acciones líquidas (2010-2026, ajustadas por splits y dividendos) con costes de acciones.

| | Acciones | Cripto |
|---|---|---|
| Volatilidad anual / rango diario | 33 % / 2,5 % | 99 % / 7,2 % |
| Correlación media entre activos | 0,22 | 0,49 |
| Conjunto diario, largos y cortos | +0,03R | -0,03R |
| Solo largos | +0,16R | -0,02R |
| Solo cortos | -0,29R | -0,04R |

- Los largos de tendencia superan a entrar al azar con la misma salida (+0,16R frente a +0,135R por
  operación; cartera 12,5 % frente a ~2,5 % anual). En cripto no hay ventaja sobre el azar.
- Aun así, comprar y mantener SPY dio 13,3 % anual (caída 34 %) y QQQ 17,5 % (caída 35 %). El bot de
  acciones, solo largos y con 1 % de riesgo, dio 12,5 % con caída 22 %: parecido en rentabilidad, menos
  caída, pero con sesgo de supervivencia (Yahoo solo ofrece empresas que hoy cotizan).
- Control sin ese sesgo, con 5 ETF: 2,3 % anual con la señal frente a 0,5 % al azar. Pocas operaciones.
- Los picos de volumen con gran rango parecían dar +0,2R a +0,3R con ~190 operaciones, pero con 524 acciones y
  569 operaciones (`research/volspike_study.py`) queda en +0,13R frente a +0,09R de entrar al azar (ventaja
  de +0,04R), y solo 16 de 34 combinaciones de parámetros superan al azar. Como cartera sin apalancamiento
  rinde ~2 % anual y, sumado a la tendencia, la empeora (7,6 % frente a 10,0 %). Descartado.
- Sin apalancamiento (cuenta al contado, 5 posiciones) el conjunto de tendencia largo da 10,0 % anual con
  caída del 19 % (1 % de riesgo) y 11,9 % con caída del 22 % (2 %), frente a 13,5 % y caída del 34 % de SPY.

## Modo acciones de EE. UU. (solo simulación)

```bash
cp .env.stocks.example .env.stocks        # y edítalo
ENV_FILE=.env.stocks python run.py --check
ENV_FILE=.env.stocks python run.py        # panel en http://127.0.0.1:8788
```

Opera en simulación las 40 acciones más líquidas de `config/stock_universe.txt`: solo largos, velas diarias
completas, sin apalancamiento, entradas solo con el mercado abierto (9:30-16:00 ET con cotizaciones frescas),
vigilante y supervisor Claude. Estado, ledger y panel están separados del bot de cripto. `TRADING_MODE=live`
no está soportado: no hay broker de acciones integrado. Datos de Yahoo Finance con retardo de ~15 min.


### Modo cartera (acciones, simulación)

Núcleo indexado (SPY) + satélite del bot + caja de ganancias. Se activa con `PORTFOLIO_MODE=true`:

```bash
ENV_FILE=.env.stocks python run.py --deposit 100          # aportación en EUR (--currency usd para dólares); parar el bot antes
```

Cada aportación se reparte `PORTFOLIO_CORE_PCT` (70 %) al núcleo y el resto al satélite. Cada trimestre se asegura
`PORTFOLIO_LOCK_FRAC` (50 %) de la ganancia del satélite, en efectivo (`PORTFOLIO_LOCK_TO=cash`) o en el núcleo
(`core`), y el Capital Shield se rebasa. Estudio (`research/portfolio_formula.py`, aportación mensual, 2010-2026):

| Fórmula | TIR anual | Caída máx. |
|---|---|---|
| Solo SPY | 13,5 % | 34 % |
| Solo QQQ | 17,6 % | 35 % |
| 70 % SPY + 30 % bot | 13,4 % | 26 % |
| 70/30 + 50 % de ganancias a caja | 12,7 % | 26 % |
| SPY solo sobre su media de 200 días | 10,4 % | 20 % |

Ninguna fórmula domina: asegurar ganancias cuesta ~0,7 puntos de rentabilidad y apenas cambia la caída del
conjunto (si se pasan a SPY, ni eso). Lo que más pesa es el núcleo elegido. El satélite se aproxima con ganancias
realizadas y con universo con sesgo de supervivencia.

### Panel visual y repetición

- `python run.py` abre el panel en http://127.0.0.1:8787 (8788 en acciones): velas de cada posición con entrada, stop
  y precio actual, mapa del mercado con señales, actividad, reparto de la cartera y gráfico al pulsar cualquier fila.
- `python research/make_replay.py` genera `dashboard/replay.html`, una repetición animada autocontenida del bot día
  a día sobre datos reales (acciones y cripto). Se abre con doble clic.

## Escáner multi-timeframe (`strategy/`, solo lectura)

Tres módulos desacoplados más un escáner que los une. No firma ni envía órdenes.

```bash
python -m strategy.scanner --testnet --top-volume 15       # lectura contra TESTNET_API_URL
python -m strategy.scanner --mainnet --top-volume 20 --top-movers 5 -v
python -m strategy.scanner --coins BTC ETH SOL --account-value 224 --json
RUN_LIVE_TESTS=1 pytest tests/test_strategy_live.py -v     # prueba de lectura real (LIVE_NETWORK=mainnet para mainnet)
```

| Módulo | Qué hace |
|---|---|
| `market_data.py` | Velas 4h/1h/15m con columna `closed` (reloj del servidor sincronizado con el L2), contexto de perpetuos (funding actual y predicho, OI con histórico en memoria), spread y profundidad del L2, watchlist dinámica; API síncrona y asíncrona con reintentos |
| `filters_and_candles.py` | Embudo (datos → ATR/precio → volumen relativo > 1,5x → régimen ADX/EMA 50-200 en 4h y 1h → derivados) y 4 patrones sobre velas cerradas: barrida de liquidez, ruptura tras compresión, envolvente con tendencia, pin bar en zona clave |
| `confluence_and_risk.py` | Puntuación 0-10 (MTF 2,5 + patrón 3 + volumen 2 + funding/OI 1,5 + ADX 1), aprobación con >= 6,5, stop tras la mecha o a 1,5 ATR (máx. 3 ATR), R:R 1:2 (1:2,5 con puntuación >= 8,5) y tamaño `capital x 1 % / |entrada - stop|` redondeado hacia abajo a `szDecimals` |

Notas de diseño: el funding de Hyperliquid es **horario** (0,0003 = 0,03 %/h, ya extremo; lo habitual es 0,00125 %/h),
por eso hay un umbral de penalización a 0,01 %/h y uno de bloqueo a 0,03 %/h. El OI necesita dos lecturas: el escáner
guarda `state/oi_history.json` y lo reutiliza en la siguiente ejecución. Los indicadores usan `pandas-ta` y, si falla o
hay pocas velas, un cálculo nativo equivalente (diferencia < 2e-5). **Esta capa no está validada con un backtest**: las
pruebas de `research/` mostraron que las estrategias de 15 min y 1 h pierden por costes.

## Estructura

```
hyper_aegis/
├── config/        settings.py (.env + parámetros), prompts.py (supervisor y diagnóstico)
├── data/          market_feed.py, derivatives_data.py (OI, funding, watchlist), news_engine.py, stock_feed.py (acciones)
├── strategy/      market_data.py, filters_and_candles.py, confluence_and_risk.py, scanner.py (escáner de solo lectura)
├── quant/         strategies.py (señales en vivo), technicals.py y anomaly_trigger.py (legado 15m)
├── brain/         claude_analyst.py (veto + diagnóstico), schemas.py
├── execution/     risk_manager.py, hyperliquid_trader.py (paper/live), watchdog.py, tax_exporter.py
├── dashboard/     server.py + index.html (panel local)
├── alerts/        telegram_notifier.py
├── research/      download.py, funding.py, lab.py, monthly.py, validate.py (validación dura), next_steps.py
├── tests/         93 tests (pytest)
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
pytest                          # 93 tests
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
