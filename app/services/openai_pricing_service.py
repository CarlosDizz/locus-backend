from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import requests
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.db.models import PriceSnapshot


logger = logging.getLogger(__name__)


class PricingRefreshError(RuntimeError):
    pass


@dataclass(slots=True)
class ParsedPricing:
    model: str
    source_url: str
    source_label: str
    raw_source_hash: str
    fetched_at: datetime
    text_input_per_million: Decimal = Decimal("0")
    text_cached_input_per_million: Decimal = Decimal("0")
    text_output_per_million: Decimal = Decimal("0")
    audio_input_per_million: Decimal = Decimal("0")
    audio_cached_input_per_million: Decimal = Decimal("0")
    audio_output_per_million: Decimal = Decimal("0")
    image_input_per_million: Decimal = Decimal("0")
    image_cached_input_per_million: Decimal = Decimal("0")


class OpenAIPricingService:
    MODEL_DOC_TEMPLATE = "https://developers.openai.com/api/docs/models/{model}"
    STALE_AFTER = timedelta(days=3)

    def get_or_refresh_snapshot(self, db: Session, *, endpoint: str, model: str) -> PriceSnapshot:
        current = self._latest_snapshot(db, provider="openai", endpoint=endpoint, model=model)
        if current is not None and current.fetched_at and current.fetched_at >= datetime.utcnow() - self.STALE_AFTER:
            return current

        try:
            return self.refresh_snapshot(db, endpoint=endpoint, model=model, current=current)
        except PricingRefreshError:
            if current is not None:
                logger.warning("Using stale price snapshot for %s:%s after refresh failure", endpoint, model, exc_info=True)
                return current
            raise

    def refresh_snapshot(
        self,
        db: Session,
        *,
        endpoint: str,
        model: str,
        current: PriceSnapshot | None = None,
    ) -> PriceSnapshot:
        parsed = self._fetch_model_pricing(model)
        if endpoint == "realtime":
            parsed = self._normalize_realtime_pricing(parsed)
        latest = current or self._latest_snapshot(db, provider="openai", endpoint=endpoint, model=model)
        if latest is None:
            snapshot = PriceSnapshot(
                provider="openai",
                endpoint=endpoint,
                model=model,
                currency="USD",
                input_per_million=parsed.text_input_per_million,
                cached_input_per_million=parsed.text_cached_input_per_million,
                output_per_million=parsed.text_output_per_million,
                text_input_per_million=parsed.text_input_per_million,
                text_cached_input_per_million=parsed.text_cached_input_per_million,
                text_output_per_million=parsed.text_output_per_million,
                audio_input_per_million=parsed.audio_input_per_million,
                audio_cached_input_per_million=parsed.audio_cached_input_per_million,
                audio_output_per_million=parsed.audio_output_per_million,
                image_input_per_million=parsed.image_input_per_million,
                image_cached_input_per_million=parsed.image_cached_input_per_million,
                source_url=parsed.source_url,
                source_label=parsed.source_label,
                raw_source_hash=parsed.raw_source_hash,
                fetched_at=parsed.fetched_at.replace(tzinfo=None),
            )
            db.add(snapshot)
            db.flush()
            return snapshot

        if self._has_same_prices(latest, parsed):
            latest.source_url = parsed.source_url
            latest.source_label = parsed.source_label
            latest.raw_source_hash = parsed.raw_source_hash
            latest.fetched_at = parsed.fetched_at.replace(tzinfo=None)
            db.flush()
            return latest

        snapshot = PriceSnapshot(
            provider="openai",
            endpoint=endpoint,
            model=model,
            currency="USD",
            input_per_million=parsed.text_input_per_million,
            cached_input_per_million=parsed.text_cached_input_per_million,
            output_per_million=parsed.text_output_per_million,
            text_input_per_million=parsed.text_input_per_million,
            text_cached_input_per_million=parsed.text_cached_input_per_million,
            text_output_per_million=parsed.text_output_per_million,
            audio_input_per_million=parsed.audio_input_per_million,
            audio_cached_input_per_million=parsed.audio_cached_input_per_million,
            audio_output_per_million=parsed.audio_output_per_million,
            image_input_per_million=parsed.image_input_per_million,
            image_cached_input_per_million=parsed.image_cached_input_per_million,
            source_url=parsed.source_url,
            source_label=parsed.source_label,
            raw_source_hash=parsed.raw_source_hash,
            fetched_at=parsed.fetched_at.replace(tzinfo=None),
        )
        db.add(snapshot)
        db.flush()
        return snapshot

    def _normalize_realtime_pricing(self, parsed: ParsedPricing) -> ParsedPricing:
        # Some realtime model pages expose only generic input/output pricing even though
        # the model is used for audio. When that happens, keep the documented general
        # prices but mirror them into audio buckets so the billing engine does not
        # accidentally value audio traffic at zero.
        if (
            parsed.audio_input_per_million <= Decimal("0")
            and parsed.audio_cached_input_per_million <= Decimal("0")
            and parsed.audio_output_per_million <= Decimal("0")
        ):
            parsed.audio_input_per_million = parsed.text_input_per_million
            parsed.audio_cached_input_per_million = parsed.text_cached_input_per_million
            parsed.audio_output_per_million = parsed.text_output_per_million
        return parsed

    def _latest_snapshot(self, db: Session, *, provider: str, endpoint: str, model: str) -> PriceSnapshot | None:
        stmt = (
            select(PriceSnapshot)
            .where(PriceSnapshot.provider == provider)
            .where(PriceSnapshot.endpoint == endpoint)
            .where(PriceSnapshot.model == model)
            .order_by(desc(PriceSnapshot.active_from), desc(PriceSnapshot.id))
            .limit(1)
        )
        return db.scalar(stmt)

    def _fetch_model_pricing(self, model: str) -> ParsedPricing:
        url = self.MODEL_DOC_TEMPLATE.format(model=model)
        try:
            response = requests.get(
                url,
                timeout=20,
                headers={"User-Agent": "locus-billing/1.0"},
            )
        except requests.RequestException as exc:
            raise PricingRefreshError(f"No he podido consultar la ficha oficial del modelo {model}") from exc
        if not response.ok:
            raise PricingRefreshError(f"La ficha oficial del modelo {model} ha respondido {response.status_code}")

        html = response.text
        text_prices = self._extract_section_prices(html, "Text tokens")
        if text_prices["input"] <= Decimal("0") and text_prices["output"] <= Decimal("0"):
            raise PricingRefreshError(f"No he podido extraer el pricing oficial del modelo {model}")

        audio_prices = self._extract_section_prices(html, "Audio tokens")
        image_prices = self._extract_section_prices(html, "Image tokens")
        raw_source_hash = hashlib.sha256(html.encode("utf-8")).hexdigest()

        return ParsedPricing(
            model=model,
            source_url=url,
            source_label="openai_model_doc",
            raw_source_hash=raw_source_hash,
            fetched_at=datetime.now(UTC),
            text_input_per_million=text_prices["input"],
            text_cached_input_per_million=text_prices["cached_input"],
            text_output_per_million=text_prices["output"],
            audio_input_per_million=audio_prices["input"],
            audio_cached_input_per_million=audio_prices["cached_input"],
            audio_output_per_million=audio_prices["output"],
            image_input_per_million=image_prices["input"],
            image_cached_input_per_million=image_prices["cached_input"],
        )

    def _extract_section_prices(self, html: str, section_title: str) -> dict[str, Decimal]:
        start = html.find(section_title)
        if start < 0:
            return {"input": Decimal("0"), "cached_input": Decimal("0"), "output": Decimal("0")}

        end_candidates = [
            idx
            for idx in (
                html.find("Text tokens", start + 1),
                html.find("Audio tokens", start + 1),
                html.find("Image tokens", start + 1),
                html.find("Modalities", start + 1),
                html.find("Endpoints", start + 1),
            )
            if idx > start
        ]
        end = min(end_candidates) if end_candidates else min(len(html), start + 4000)
        chunk = html[start:end]

        return {
            "input": self._extract_price_value(chunk, "Input"),
            "cached_input": self._extract_price_value(chunk, "Cached input"),
            "output": self._extract_price_value(chunk, "Output"),
        }

    def _extract_price_value(self, chunk: str, label: str) -> Decimal:
        pattern = re.compile(
            re.escape(f"<div>{label}</div>") + r"\s*<div[^>]*>\$([0-9]+(?:\.[0-9]+)?)</div>",
            re.IGNORECASE,
        )
        match = pattern.search(chunk)
        if not match:
            return Decimal("0")
        return Decimal(match.group(1))

    def _has_same_prices(self, snapshot: PriceSnapshot, parsed: ParsedPricing) -> bool:
        current = (
            self._decimal(snapshot.text_input_per_million),
            self._decimal(snapshot.text_cached_input_per_million),
            self._decimal(snapshot.text_output_per_million),
            self._decimal(snapshot.audio_input_per_million),
            self._decimal(snapshot.audio_cached_input_per_million),
            self._decimal(snapshot.audio_output_per_million),
            self._decimal(snapshot.image_input_per_million),
            self._decimal(snapshot.image_cached_input_per_million),
        )
        latest = (
            parsed.text_input_per_million,
            parsed.text_cached_input_per_million,
            parsed.text_output_per_million,
            parsed.audio_input_per_million,
            parsed.audio_cached_input_per_million,
            parsed.audio_output_per_million,
            parsed.image_input_per_million,
            parsed.image_cached_input_per_million,
        )
        return current == latest

    @staticmethod
    def _decimal(value: Decimal | None) -> Decimal:
        if value is None:
            return Decimal("0")
        return Decimal(str(value))


openai_pricing_service = OpenAIPricingService()
