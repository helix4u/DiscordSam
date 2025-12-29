"""Pricing Tracker for API usage and cost reporting."""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

from api_provider_manager import get_provider_manager, ProviderPricing

logger = logging.getLogger(__name__)


@dataclass
class UsageRecord:
    """Record of API usage."""
    timestamp: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: Optional[float] = None
    currency: str = "USD"
    request_id: Optional[str] = None


class PricingTracker:
    """Tracks API usage and costs."""
    
    def __init__(self, data_file: Optional[str] = None):
        self.data_file = data_file or os.path.join(
            os.path.dirname(__file__), "pricing_data.json"
        )
        self._usage_records: List[UsageRecord] = []
        self._load_data()
    
    def _load_data(self) -> None:
        """Load usage data from disk."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._usage_records = [
                        UsageRecord(**record) for record in data.get("records", [])
                    ]
                logger.info(f"Loaded {len(self._usage_records)} usage records")
            except Exception as e:
                logger.error(f"Failed to load pricing data: {e}", exc_info=True)
                self._usage_records = []
    
    def _save_data(self) -> None:
        """Save usage data to disk."""
        try:
            data = {
                "records": [asdict(record) for record in self._usage_records],
                "last_updated": datetime.now().isoformat(),
            }
            tmp_path = self.data_file + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.data_file)
        except Exception as e:
            logger.error(f"Failed to save pricing data: {e}", exc_info=True)
    
    def record_usage(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        request_id: Optional[str] = None,
    ) -> None:
        """Record API usage."""
        provider_manager = get_provider_manager()
        cost = provider_manager.calculate_cost(model, input_tokens, output_tokens)
        pricing = provider_manager.get_pricing(model)
        currency = pricing.currency if pricing else "USD"
        
        record = UsageRecord(
            timestamp=datetime.now().isoformat(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            currency=currency,
            request_id=request_id,
        )
        
        self._usage_records.append(record)
        self._save_data()
        
        if cost:
            logger.debug(
                f"Recorded usage: {provider}/{model} - "
                f"{input_tokens} in, {output_tokens} out - ${cost:.6f}"
            )
    
    def get_usage_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, any]:
        """Get usage summary for a time period."""
        filtered = self._usage_records
        
        if start_date:
            filtered = [r for r in filtered if datetime.fromisoformat(r.timestamp) >= start_date]
        if end_date:
            filtered = [r for r in filtered if datetime.fromisoformat(r.timestamp) <= end_date]
        if provider:
            filtered = [r for r in filtered if r.provider == provider]
        if model:
            filtered = [r for r in filtered if r.model == model]
        
        total_input_tokens = sum(r.input_tokens for r in filtered)
        total_output_tokens = sum(r.output_tokens for r in filtered)
        total_cost = sum(r.cost or 0.0 for r in filtered)
        request_count = len(filtered)
        
        # Group by provider/model
        by_provider_model: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "requests": 0}
        )
        
        for record in filtered:
            key = (record.provider, record.model)
            by_provider_model[key]["input_tokens"] += record.input_tokens
            by_provider_model[key]["output_tokens"] += record.output_tokens
            by_provider_model[key]["cost"] += record.cost or 0.0
            by_provider_model[key]["requests"] += 1
        
        return {
            "period_start": start_date.isoformat() if start_date else None,
            "period_end": end_date.isoformat() if end_date else None,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "total_cost": total_cost,
            "request_count": request_count,
            "by_provider_model": {
                f"{provider}/{model}": stats
                for (provider, model), stats in by_provider_model.items()
            },
        }
    
    def get_daily_summary(self, date: Optional[datetime] = None) -> Dict[str, any]:
        """Get summary for a specific day."""
        if date is None:
            date = datetime.now()
        
        day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        return self.get_usage_summary(start_date=day_start, end_date=day_end)
    
    def get_hourly_summary(self, hour: Optional[datetime] = None) -> Dict[str, any]:
        """Get summary for a specific hour."""
        if hour is None:
            hour = datetime.now()
        
        hour_start = hour.replace(minute=0, second=0, microsecond=0)
        hour_end = hour_start + timedelta(hours=1)
        
        return self.get_usage_summary(start_date=hour_start, end_date=hour_end)
    
    def get_monthly_summary(self, month: Optional[datetime] = None) -> Dict[str, any]:
        """Get summary for a specific month."""
        if month is None:
            month = datetime.now()
        
        month_start = month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if month.month == 12:
            month_end = month.replace(year=month.year + 1, month=1, day=1)
        else:
            month_end = month.replace(month=month.month + 1, day=1)
        
        return self.get_usage_summary(start_date=month_start, end_date=month_end)
    
    def get_yearly_summary(self, year: Optional[int] = None) -> Dict[str, any]:
        """Get summary for a specific year."""
        if year is None:
            year = datetime.now().year
        
        year_start = datetime(year, 1, 1)
        year_end = datetime(year + 1, 1, 1)
        
        return self.get_usage_summary(start_date=year_start, end_date=year_end)
    
    def cleanup_old_records(self, days_to_keep: int = 365) -> int:
        """Remove records older than specified days."""
        cutoff = datetime.now() - timedelta(days=days_to_keep)
        before_count = len(self._usage_records)
        self._usage_records = [
            r for r in self._usage_records
            if datetime.fromisoformat(r.timestamp) >= cutoff
        ]
        removed = before_count - len(self._usage_records)
        if removed > 0:
            self._save_data()
            logger.info(f"Cleaned up {removed} old usage records")
        return removed


# Global pricing tracker instance
_pricing_tracker: Optional[PricingTracker] = None


def get_pricing_tracker() -> PricingTracker:
    """Get the global pricing tracker instance."""
    global _pricing_tracker
    if _pricing_tracker is None:
        _pricing_tracker = PricingTracker()
    return _pricing_tracker
