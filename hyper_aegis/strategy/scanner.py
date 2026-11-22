"""Escáner de solo lectura: watchlist -> datos -> embudo -> patrones -> confluencia -> tamaño.

No firma ni envía órdenes: solo usa los endpoints públicos de `Info`.

    python -m strategy.scanner --testnet --top-volume 15        # lectura contra testnet
    python -m strategy.scanner --coins BTC ETH SOL --account-value 224
    python -m strategy.scanner --json > senales.json

Etapas (cada una descarta antes de gastar más peticiones): volumen 24h -> velas (3 timeframes) -> embudo
(datos, volatilidad, volumen relativo, régimen, derivados) -> patrones -> libro L2 (spread/profundidad) -> puntuación.
El Open Interest se compara con la lectura previa guardada en `state/oi_history.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings, get_settings  # noqa: E402
from strategy.confluence_and_risk import ScoreConfig, TradeSignal, evaluate_patterns  # noqa: E402
from strategy.filters_and_candles import FilterConfig, detect_patterns, evaluate_funnel, prepare_frames, trend_direction  # noqa: E402
from strategy.logutil import log_event  # noqa: E402
from strategy.market_data import LiquidityPolicy, MarketDataClient, MarketDataError  # noqa: E402

log = logging.getLogger("strategy.scanner")
OI_KEEP_SEC = 6 * 3600


@dataclass
class ScanReport:
    network: str
    watchlist: List[str]
    signals: List[TradeSignal] = field(default_factory=list)
    discarded: Counter = field(default_factory=Counter)  # etapa -> nº de activos descartados
    notes: Dict[str, str] = field(default_factory=dict)  # activo -> motivo del descarte (modo detallado)
    elapsed_sec: float = 0.0

    @property
    def approved(self) -> List[TradeSignal]:
        return [s for s in self.signals if s.approved]


def load_oi_history(client: MarketDataClient, path: Path) -> None:
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    cutoff = time.time() - OI_KEEP_SEC
    for coin, rows in raw.items():
        client.seed_oi_history(coin, [tuple(r) for r in rows if r[0] >= cutoff])


def save_oi_history(client: MarketDataClient, path: Path) -> None:
    cutoff = time.time() - OI_KEEP_SEC
    data = {c: [list(r) for r in h if r[0] >= cutoff] for c, h in client._oi_hist.items() if h}  # noqa: SLF001
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    except OSError as exc:
        log_event(log, logging.WARNING, "oi_history_not_saved", error=repr(exc))


def scan(
    client: MarketDataClient, coins: Optional[Sequence[str]] = None, account_value: Optional[float] = None,
    top_volume: int = 20, top_movers: int = 5, filt: Optional[FilterConfig] = None, score_cfg: Optional[ScoreConfig] = None,
    fee_rate: float = 0.0, verbose: bool = False,
) -> ScanReport:
    """Ejecuta el escáner completo y devuelve el informe (señales aprobadas y rechazadas, descartes por etapa)."""
    filt, score_cfg = filt or FilterConfig(), score_cfg or ScoreConfig()
    t0 = time.time()
    client.sync_clock()
    ctxs = client.get_perp_contexts()
    watchlist = list(coins) if coins else client.build_watchlist(ctxs, top_volume, top_movers)
    rep = ScanReport(network=client.api_url, watchlist=watchlist)

    for coin in watchlist:
        ctx = ctxs.get(coin)
        if ctx is None:
            rep.discarded["no_existe"] += 1
            continue
        ok, why = client.policy.check(ctx)
        if not ok:
            rep.discarded["liquidez_volumen"] += 1
            rep.notes[coin] = why
            continue
        try:
            frames = prepare_frames(client.get_candle_set(coin))
        except MarketDataError as exc:
            rep.discarded["error_datos"] += 1
            rep.notes[coin] = str(exc)
            continue
        funnel = evaluate_funnel(coin, frames, ctx, client.oi_change_pct(coin, filt.oi_window_sec),
                                 client.price_change_pct(coin, filt.oi_window_sec), filt)
        if not funnel.passed:
            rep.discarded[f"filtro_{funnel.failed_stage}"] += 1
            rep.notes[coin] = next((c.detail for c in reversed(funnel.checks) if not c.passed), "")
            continue
        patterns = detect_patterns(frames[filt.trigger_tf], trend_direction(funnel.regimes, filt), filt)
        if not patterns:
            rep.discarded["sin_patron"] += 1
            continue
        try:
            l2 = client.get_l2_metrics(coin)
        except MarketDataError as exc:
            rep.discarded["error_datos"] += 1
            rep.notes[coin] = str(exc)
            continue
        ok, why = client.policy.check(ctx, l2)
        if not ok:
            rep.discarded["liquidez_libro"] += 1
            rep.notes[coin] = why
            continue
        rep.signals += evaluate_patterns(coin, patterns, funnel, frames[filt.trigger_tf], ctx.sz_decimals, account_value,
                                         ctx.max_leverage, filt, score_cfg, fee_rate)
    rep.signals.sort(key=lambda s: s.score, reverse=True)
    rep.elapsed_sec = time.time() - t0
    log_event(log, logging.INFO, "scan_done", watchlist=len(watchlist), approved=len(rep.approved), candidates=len(rep.signals),
              seconds=round(rep.elapsed_sec, 1))
    return rep


def print_report(rep: ScanReport, verbose: bool = False) -> None:
    print(f"\nRed: {rep.network} | activos analizados: {len(rep.watchlist)} | {rep.elapsed_sec:.0f} s | SOLO LECTURA (no se envía ninguna orden)")
    if rep.discarded:
        print("Descartes por etapa: " + ", ".join(f"{k}={v}" for k, v in rep.discarded.most_common()))
    if verbose and rep.notes:
        for coin, why in rep.notes.items():
            print(f"   - {coin}: {why}")
    if not rep.signals:
        print("\nSin señales: ningún activo superó el embudo con un patrón válido en la última vela cerrada.")
        return
    print(f"\n{'ACTIVO':8} {'DIR':5} {'PUNT':>5} {'PATRÓN':17} {'ENTRADA':>10} {'STOP':>10} {'OBJETIVO':>10} {'TAMAÑO':>9} {'RIESGO$':>8}  ESTADO")
    for s in rep.signals:
        pos = s.position
        size = f"{pos.size:g}" if pos and pos.ok else "-"
        risk = f"{pos.risk_usd:.2f}" if pos and pos.ok else "-"
        state = "APROBADA" if s.approved else f"rechazada: {s.reject_reason}"
        print(f"{s.symbol:8} {s.direction:5} {s.score:5.1f} {s.pattern:17} {s.entry_price:10g} {s.stop_loss:10g} {s.take_profit:10g} {size:>9} {risk:>8}  {state}")
        for r in s.reasons:
            print(f"           · {r}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Escáner de solo lectura de HyperAegis (no envía órdenes)")
    net = ap.add_mutually_exclusive_group()
    net.add_argument("--testnet", action="store_true", help="usar TESTNET_API_URL")
    net.add_argument("--mainnet", action="store_true", help="usar MAINNET_API_URL")
    ap.add_argument("--coins", nargs="*", help="activos concretos en lugar de la watchlist dinámica")
    ap.add_argument("--top-volume", type=int, default=20)
    ap.add_argument("--top-movers", type=int, default=5)
    ap.add_argument("--account-value", type=float, help="capital en USDC para el tamaño (por defecto INITIAL_CAPITAL)")
    ap.add_argument("--address", help="dirección de la cuenta: lee accountValue real (solo lectura)")
    ap.add_argument("--min-score", type=float, default=ScoreConfig().min_score)
    ap.add_argument("--rel-volume", type=float, default=FilterConfig().rel_volume_min, help="volumen relativo mínimo del embudo")
    ap.add_argument("--min-volume-usd", type=float, default=LiquidityPolicy().min_day_volume_usd, help="volumen 24h mínimo (USD)")
    ap.add_argument("--max-spread-bps", type=float, default=LiquidityPolicy().max_spread_bps, help="spread máximo del libro (puntos básicos)")
    ap.add_argument("--fee-rate", type=float, default=0.0, help="comisión por lado a incluir en el riesgo (0,00045 = taker)")
    ap.add_argument("--json", action="store_true", help="salida JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings: Settings = get_settings()
    if a.testnet or a.mainnet:
        settings = settings.model_copy(update={"use_testnet": a.testnet})
    policy = LiquidityPolicy(min_day_volume_usd=a.min_volume_usd, max_spread_bps=a.max_spread_bps)
    client = MarketDataClient(settings, policy=policy)
    oi_path = settings.state_dir / "oi_history.json"
    load_oi_history(client, oi_path)
    account = a.account_value
    if a.address:
        account = client.get_account_value(a.address)
    account = account if account is not None else settings.initial_capital
    from dataclasses import replace

    try:
        rep = scan(client, a.coins, account, a.top_volume, a.top_movers, replace(FilterConfig(), rel_volume_min=a.rel_volume),
                   replace(ScoreConfig(), min_score=a.min_score), a.fee_rate, a.verbose)
    except MarketDataError as exc:
        print(f"Error de datos: {exc}", file=sys.stderr)
        return 1
    finally:
        save_oi_history(client, oi_path)
    if a.json:
        print(json.dumps({"network": rep.network, "discarded": dict(rep.discarded), "signals": [s.to_dict() for s in rep.signals]}, indent=2, default=str))
    else:
        print_report(rep, a.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
