"""Análisis de mercado, filtrado cuantitativo, patrones de velas y gestión de riesgo.

Tres módulos desacoplados (sin dependencias cruzadas con el resto del bot salvo `config.settings`):

- `market_data`          Extracción de velas multi-timeframe y métricas de perpetuos de Hyperliquid.
- `filters_and_candles`  Embudo de pre-filtros cuantitativos y detección de patrones en velas cerradas.
- `confluence_and_risk`  Puntuación de confluencia, stop/take profit estructurales y sizing al 1 %.

`scanner` los une en un escáner de solo lectura: `python -m strategy.scanner --help`.
"""
