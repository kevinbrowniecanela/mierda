"""Motor de noticias asíncrono: CryptoPanic + RSS (CoinDesk, The Block).

Las noticias se cachean unos minutos para no saturar las fuentes. Solo se
consultan cuando la Fase 1 despierta a Claude, por lo que el coste es bajo.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import aiohttp
import feedparser

from config.settings import Settings

log = logging.getLogger(__name__)

# Alias habituales para casar titulares con el ticker
_ALIASES: dict[str, list[str]] = {
    "BTC": ["bitcoin"], "ETH": ["ethereum", "ether"], "SOL": ["solana"], "XRP": ["ripple"],
    "DOGE": ["dogecoin"], "AVAX": ["avalanche"], "LINK": ["chainlink"], "ADA": ["cardano"],
    "SUI": ["sui"], "HYPE": ["hyperliquid"], "BNB": ["binance coin", "bnb"], "TON": ["toncoin"],
    "LTC": ["litecoin"], "DOT": ["polkadot"], "ARB": ["arbitrum"], "OP": ["optimism"],
}
_MACRO_KEYWORDS = ("sec", "etf", "fed", "fomc", "cpi", "rate", "hack", "exploit", "liquidat", "tariff", "regulat")


@dataclass
class NewsItem:
    title: str
    source: str
    published: datetime | None
    url: str = ""
    sentiment: str = ""

    def line(self) -> str:
        ts = self.published.strftime("%Y-%m-%d %H:%M UTC") if self.published else "s/f"
        tag = f" [{self.sentiment}]" if self.sentiment else ""
        return f"- ({ts}, {self.source}){tag} {self.title}"


class NewsEngine:
    def __init__(self, settings: Settings, session: aiohttp.ClientSession):
        self.s = settings
        self.http = session
        self._cache: tuple[float, list[NewsItem]] = (0.0, [])
        self._lock = asyncio.Lock()
        self.ttl = 300

    async def _fetch_rss(self, url: str) -> list[NewsItem]:
        try:
            async with self.http.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                text = await resp.text()
        except Exception as exc:  # noqa: BLE001
            log.debug("RSS %s falló: %s", url, exc)
            return []
        feed = await asyncio.to_thread(feedparser.parse, text)
        source = feed.feed.get("title", url) if hasattr(feed, "feed") else url
        items = []
        for e in feed.entries[:40]:
            published = None
            if getattr(e, "published_parsed", None):
                published = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
            items.append(NewsItem(title=e.get("title", "").strip(), source=source, published=published, url=e.get("link", "")))
        return items

    async def _fetch_cryptopanic(self) -> list[NewsItem]:
        if not self.s.cryptopanic_api_key:
            return []
        params = {"auth_token": self.s.cryptopanic_api_key, "public": "true", "kind": "news"}
        try:
            async with self.http.get(self.s.cryptopanic_url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:  # noqa: BLE001
            log.debug("CryptoPanic falló: %s", exc)
            return []
        items = []
        for p in data.get("results", [])[:50]:
            published = None
            if p.get("published_at"):
                try:
                    published = datetime.fromisoformat(p["published_at"].replace("Z", "+00:00"))
                except ValueError:
                    pass
            votes = p.get("votes") or {}
            sentiment = ""
            if votes:
                pos, neg = votes.get("positive", 0), votes.get("negative", 0)
                sentiment = "bullish" if pos > neg else "bearish" if neg > pos else ""
            currencies = " ".join(c.get("code", "") for c in p.get("currencies") or [])
            title = p.get("title", "").strip()
            if currencies:
                title = f"{title} ({currencies})"
            items.append(NewsItem(title=title, source="CryptoPanic", published=published, url=p.get("url", ""), sentiment=sentiment))
        return items

    async def latest(self) -> list[NewsItem]:
        async with self._lock:
            ts, items = self._cache
            if time.time() - ts < self.ttl and items:
                return items
            results = await asyncio.gather(
                self._fetch_cryptopanic(), *(self._fetch_rss(u) for u in self.s.rss_feeds)
            )
            merged = [i for group in results for i in group if i.title]
            merged.sort(key=lambda i: i.published or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
            self._cache = (time.time(), merged)
            return merged

    async def for_coin(self, coin: str, max_items: int = 8, max_age_h: float = 24) -> list[NewsItem]:
        """Titulares que mencionan el activo, más titulares macro relevantes."""
        items = await self.latest()
        cutoff = time.time() - max_age_h * 3600
        terms = [coin.lower(), *_ALIASES.get(coin.upper(), [])]
        pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)
        specific, macro = [], []
        for it in items:
            if it.published and it.published.timestamp() < cutoff:
                continue
            if pattern.search(it.title):
                specific.append(it)
            elif any(k in it.title.lower() for k in _MACRO_KEYWORDS):
                macro.append(it)
        return (specific[:max_items] + macro[: max(0, max_items - len(specific))])[:max_items]
