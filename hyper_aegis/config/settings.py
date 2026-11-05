"""Configuración global de HyperAegis.

Todos los parámetros se leen de variables de entorno (o del archivo `.env`)
mediante pydantic-settings. Los valores por defecto son conservadores y el
modo por defecto es PAPER TRADING.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ #
    # Modo de ejecución
    # ------------------------------------------------------------------ #
    trading_mode: TradingMode = TradingMode.PAPER
    use_testnet: bool = False

    # ------------------------------------------------------------------ #
    # Hyperliquid (Agent Wallet: solo trading, sin permisos de retiro)
    # ------------------------------------------------------------------ #
    hl_agent_private_key: str = Field(default="", repr=False)
    hl_account_address: str = ""  # Dirección de la cuenta principal (master)

    # ------------------------------------------------------------------ #
    # Anthropic Claude
    # ------------------------------------------------------------------ #
    anthropic_api_key: str = Field(default="", repr=False)
    # `claude-3-5-sonnet` fue retirado por Anthropic (oct-2025); su sucesor
    # en la línea Sonnet es `claude-sonnet-5-5`.
    claude_model: str = "claude-sonnet-5-5"
    claude_effort: str = "high"  # low | medium | high | xhigh | max
    claude_max_tokens: int = 16000
    claude_enable_fallbacks: bool = True  # fallback server-side ante "refusal"
    claude_min_confidence: int = 70  # confianza mínima para ejecutar

    # ------------------------------------------------------------------ #
    # Noticias
    # ------------------------------------------------------------------ #
    cryptopanic_api_key: str = Field(default="", repr=False)
    cryptopanic_url: str = "https://cryptopanic.com/api/developer/v2/posts/"
    rss_feeds: list[str] = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://www.theblock.co/rss.xml",
    ]

    # ------------------------------------------------------------------ #
    # Telegram
    # ------------------------------------------------------------------ #
    telegram_bot_token: str = Field(default="", repr=False)
    telegram_chat_id: str = ""

    # ------------------------------------------------------------------ #
    # Fiscalidad (España)
    # ------------------------------------------------------------------ #
    fx_api_url: str = "https://api.frankfurter.dev/v1"
    tax_ledger_path: Path = PROJECT_ROOT / "tax_ledger.csv"

    # ------------------------------------------------------------------ #
    # Gestión de riesgo / Capital Shield
    # ------------------------------------------------------------------ #
    initial_capital: float = 400.0  # USDC depositados
    risk_per_trade_pct: float = 0.01  # 1 % del capital inicial
    house_money_trigger_pct: float = 0.15  # +15 % activa "House Money"
    tp1_rr: float = 1.5  # TP1 a R:R 1:1.5
    tp1_close_fraction: float = 0.5  # cerrar 50 % en TP1
    trailing_atr_mult: float = 2.0  # trailing stop = 2x ATR
    default_stop_atr_mult: float = 1.5  # SL por defecto si Claude no da uno válido
    min_stop_atr_mult: float = 0.5
    max_stop_atr_mult: float = 3.0
    max_open_positions: int = 3
    daily_drawdown_limit_pct: float = 0.03  # Circuit breaker al -3 % diario
    flatten_on_circuit_breaker: bool = True
    max_leverage: int = 5
    taker_fee_rate: float = 0.00045  # comisión taker Hyperliquid (tier base)
    slippage: float = 0.01

    # ------------------------------------------------------------------ #
    # Mercado / Watchlist
    # ------------------------------------------------------------------ #
    watchlist_top_volume: int = 20
    watchlist_top_movers: int = 5
    movers_min_volume_usd: float = 5_000_000
    watchlist_refresh_sec: int = 900
    candle_interval: str = "15m"
    candle_lookback: int = 300
    scan_interval_sec: int = 60
    position_check_interval_sec: int = 10
    max_concurrent_requests: int = 6

    # ------------------------------------------------------------------ #
    # Fase 1: disparadores de anomalía
    # ------------------------------------------------------------------ #
    anomaly_score_threshold: float = 3.0
    anomaly_cooldown_min: int = 45
    vol_spike_mult: float = 3.0
    range_atr_mult: float = 2.0
    oi_change_pct: float = 0.03
    funding_extreme_hourly: float = 0.0001  # 0.01 %/h
    book_imbalance_threshold: float = 0.35
    max_claude_calls_per_hour: int = 12

    # ------------------------------------------------------------------ #
    # Rutas
    # ------------------------------------------------------------------ #
    state_dir: Path = PROJECT_ROOT / "state"
    log_dir: Path = PROJECT_ROOT / "logs"
    log_level: str = "INFO"

    @field_validator("hl_account_address")
    @classmethod
    def _lower_addr(cls, v: str) -> str:
        return v.strip()

    # ------------------------------------------------------------------ #
    @property
    def is_live(self) -> bool:
        return self.trading_mode == TradingMode.LIVE

    @property
    def hl_api_url(self) -> str:
        from hyperliquid.utils import constants

        return constants.TESTNET_API_URL if self.use_testnet else constants.MAINNET_API_URL

    @property
    def hl_ws_url(self) -> str:
        return self.hl_api_url.replace("https://", "wss://") + "/ws"

    def validate_for_live(self) -> None:
        """Lanza ValueError si falta algo imprescindible para operar en real."""
        missing = []
        if not self.hl_agent_private_key:
            missing.append("HL_AGENT_PRIVATE_KEY")
        if not self.hl_account_address:
            missing.append("HL_ACCOUNT_ADDRESS")
        if missing:
            raise ValueError(f"Modo LIVE requiere: {', '.join(missing)}")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.state_dir.mkdir(parents=True, exist_ok=True)
    s.log_dir.mkdir(parents=True, exist_ok=True)
    return s
