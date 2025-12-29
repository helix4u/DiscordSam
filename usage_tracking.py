"""Usage Tracking System.

This module provides comprehensive token usage tracking, cost calculation,
and spending reports for LLM API usage across multiple providers.
"""

import asyncio
import json
import logging
import os
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

from multi_provider_llm import get_model_pricing, estimate_cost, format_cost

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

class TimeFrame(Enum):
    """Time frames for reporting."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    ALL_TIME = "all_time"


@dataclass
class UsageRecord:
    """A single usage record."""
    id: Optional[int]
    timestamp: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    total_tokens: int
    cost: float
    request_type: str  # 'chat', 'completion', 'embedding', etc.
    channel_id: Optional[int]
    user_id: Optional[int]
    success: bool
    latency_ms: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UsageSummary:
    """Summary of usage for a time period."""
    time_frame: str
    period_start: str
    period_end: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_input_tokens: int
    total_output_tokens: int
    total_cached_tokens: int
    total_tokens: int
    total_cost: float
    average_latency_ms: float
    by_provider: Dict[str, Dict[str, Any]]
    by_model: Dict[str, Dict[str, Any]]
    by_user: Dict[str, Dict[str, Any]]
    by_channel: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Usage Tracker
# ============================================================================

class UsageTracker:
    """Tracks and persists LLM API usage."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.join(
            os.path.dirname(__file__), "usage_tracking.db"
        )
        self._lock = asyncio.Lock()
        self._init_database()
        
        # In-memory cache for fast access to recent stats
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 60  # seconds

    def _init_database(self) -> None:
        """Initialize the SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create main usage table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS usage_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        model TEXT NOT NULL,
                        input_tokens INTEGER DEFAULT 0,
                        output_tokens INTEGER DEFAULT 0,
                        cached_tokens INTEGER DEFAULT 0,
                        total_tokens INTEGER DEFAULT 0,
                        cost REAL DEFAULT 0.0,
                        request_type TEXT DEFAULT 'chat',
                        channel_id INTEGER,
                        user_id INTEGER,
                        success INTEGER DEFAULT 1,
                        latency_ms REAL DEFAULT 0.0,
                        metadata TEXT DEFAULT '{}'
                    )
                """)
                
                # Create indexes for efficient queries
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON usage_records(timestamp)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_provider ON usage_records(provider)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_model ON usage_records(model)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user ON usage_records(user_id)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_channel ON usage_records(channel_id)
                """)
                
                # Create daily aggregates table for faster reporting
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS daily_aggregates (
                        date TEXT PRIMARY KEY,
                        total_requests INTEGER DEFAULT 0,
                        total_input_tokens INTEGER DEFAULT 0,
                        total_output_tokens INTEGER DEFAULT 0,
                        total_cached_tokens INTEGER DEFAULT 0,
                        total_cost REAL DEFAULT 0.0,
                        provider_breakdown TEXT DEFAULT '{}',
                        model_breakdown TEXT DEFAULT '{}',
                        updated_at TEXT
                    )
                """)
                
                conn.commit()
                logger.info(f"Usage tracking database initialized at {self.db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize usage database: {e}", exc_info=True)

    async def record_usage(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        request_type: str = "chat",
        channel_id: Optional[int] = None,
        user_id: Optional[int] = None,
        success: bool = True,
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UsageRecord:
        """Record a usage event."""
        timestamp = datetime.now(timezone.utc).isoformat()
        total_tokens = input_tokens + output_tokens
        cost = estimate_cost(model, input_tokens, output_tokens, cached_tokens)
        
        record = UsageRecord(
            id=None,
            timestamp=timestamp,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            total_tokens=total_tokens,
            cost=cost,
            request_type=request_type,
            channel_id=channel_id,
            user_id=user_id,
            success=success,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
        
        async with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO usage_records (
                            timestamp, provider, model, input_tokens, output_tokens,
                            cached_tokens, total_tokens, cost, request_type,
                            channel_id, user_id, success, latency_ms, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        timestamp, provider, model, input_tokens, output_tokens,
                        cached_tokens, total_tokens, cost, request_type,
                        channel_id, user_id, 1 if success else 0, latency_ms,
                        json.dumps(metadata or {}),
                    ))
                    record.id = cursor.lastrowid
                    conn.commit()
                    
                # Update daily aggregate
                await self._update_daily_aggregate(record)
                
                # Invalidate cache
                self._cache.clear()
                
                logger.debug(
                    f"Recorded usage: {model} - {input_tokens}in/{output_tokens}out "
                    f"tokens, cost: {format_cost(cost)}"
                )
                
            except Exception as e:
                logger.error(f"Failed to record usage: {e}", exc_info=True)
        
        return record

    async def _update_daily_aggregate(self, record: UsageRecord) -> None:
        """Update daily aggregate for the record's date."""
        try:
            date_str = record.timestamp[:10]  # YYYY-MM-DD
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get existing aggregate
                cursor.execute(
                    "SELECT * FROM daily_aggregates WHERE date = ?",
                    (date_str,)
                )
                row = cursor.fetchone()
                
                if row:
                    # Update existing
                    provider_breakdown = json.loads(row[6])
                    model_breakdown = json.loads(row[7])
                    
                    # Update provider breakdown
                    if record.provider not in provider_breakdown:
                        provider_breakdown[record.provider] = {
                            "requests": 0, "tokens": 0, "cost": 0.0
                        }
                    provider_breakdown[record.provider]["requests"] += 1
                    provider_breakdown[record.provider]["tokens"] += record.total_tokens
                    provider_breakdown[record.provider]["cost"] += record.cost
                    
                    # Update model breakdown
                    if record.model not in model_breakdown:
                        model_breakdown[record.model] = {
                            "requests": 0, "tokens": 0, "cost": 0.0
                        }
                    model_breakdown[record.model]["requests"] += 1
                    model_breakdown[record.model]["tokens"] += record.total_tokens
                    model_breakdown[record.model]["cost"] += record.cost
                    
                    cursor.execute("""
                        UPDATE daily_aggregates SET
                            total_requests = total_requests + 1,
                            total_input_tokens = total_input_tokens + ?,
                            total_output_tokens = total_output_tokens + ?,
                            total_cached_tokens = total_cached_tokens + ?,
                            total_cost = total_cost + ?,
                            provider_breakdown = ?,
                            model_breakdown = ?,
                            updated_at = ?
                        WHERE date = ?
                    """, (
                        record.input_tokens, record.output_tokens,
                        record.cached_tokens, record.cost,
                        json.dumps(provider_breakdown), json.dumps(model_breakdown),
                        datetime.now(timezone.utc).isoformat(), date_str
                    ))
                else:
                    # Insert new
                    provider_breakdown = {
                        record.provider: {
                            "requests": 1,
                            "tokens": record.total_tokens,
                            "cost": record.cost
                        }
                    }
                    model_breakdown = {
                        record.model: {
                            "requests": 1,
                            "tokens": record.total_tokens,
                            "cost": record.cost
                        }
                    }
                    
                    cursor.execute("""
                        INSERT INTO daily_aggregates (
                            date, total_requests, total_input_tokens,
                            total_output_tokens, total_cached_tokens, total_cost,
                            provider_breakdown, model_breakdown, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        date_str, 1, record.input_tokens, record.output_tokens,
                        record.cached_tokens, record.cost,
                        json.dumps(provider_breakdown), json.dumps(model_breakdown),
                        datetime.now(timezone.utc).isoformat()
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update daily aggregate: {e}", exc_info=True)

    async def get_usage_summary(
        self,
        time_frame: TimeFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> UsageSummary:
        """Get usage summary for a time period."""
        now = datetime.now(timezone.utc)
        
        # Determine date range based on time frame
        if time_frame == TimeFrame.HOURLY:
            if start_date is None:
                start_date = now - timedelta(hours=1)
            if end_date is None:
                end_date = now
        elif time_frame == TimeFrame.DAILY:
            if start_date is None:
                start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            if end_date is None:
                end_date = now
        elif time_frame == TimeFrame.WEEKLY:
            if start_date is None:
                start_date = now - timedelta(days=7)
            if end_date is None:
                end_date = now
        elif time_frame == TimeFrame.MONTHLY:
            if start_date is None:
                start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if end_date is None:
                end_date = now
        elif time_frame == TimeFrame.YEARLY:
            if start_date is None:
                start_date = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            if end_date is None:
                end_date = now
        else:  # ALL_TIME
            start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
            end_date = now
        
        start_iso = start_date.isoformat()
        end_iso = end_date.isoformat()
        
        summary = UsageSummary(
            time_frame=time_frame.value,
            period_start=start_iso,
            period_end=end_iso,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_cached_tokens=0,
            total_tokens=0,
            total_cost=0.0,
            average_latency_ms=0.0,
            by_provider={},
            by_model={},
            by_user={},
            by_channel={},
        )
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get overall stats
                cursor.execute("""
                    SELECT
                        COUNT(*) as total_requests,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                        SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed,
                        SUM(input_tokens) as input_tokens,
                        SUM(output_tokens) as output_tokens,
                        SUM(cached_tokens) as cached_tokens,
                        SUM(total_tokens) as total_tokens,
                        SUM(cost) as total_cost,
                        AVG(latency_ms) as avg_latency
                    FROM usage_records
                    WHERE timestamp >= ? AND timestamp <= ?
                """, (start_iso, end_iso))
                
                row = cursor.fetchone()
                if row and row[0]:
                    summary.total_requests = row[0] or 0
                    summary.successful_requests = row[1] or 0
                    summary.failed_requests = row[2] or 0
                    summary.total_input_tokens = row[3] or 0
                    summary.total_output_tokens = row[4] or 0
                    summary.total_cached_tokens = row[5] or 0
                    summary.total_tokens = row[6] or 0
                    summary.total_cost = row[7] or 0.0
                    summary.average_latency_ms = row[8] or 0.0
                
                # Get by provider
                cursor.execute("""
                    SELECT
                        provider,
                        COUNT(*) as requests,
                        SUM(total_tokens) as tokens,
                        SUM(cost) as cost
                    FROM usage_records
                    WHERE timestamp >= ? AND timestamp <= ?
                    GROUP BY provider
                """, (start_iso, end_iso))
                
                for row in cursor.fetchall():
                    summary.by_provider[row[0]] = {
                        "requests": row[1],
                        "tokens": row[2],
                        "cost": row[3],
                    }
                
                # Get by model
                cursor.execute("""
                    SELECT
                        model,
                        COUNT(*) as requests,
                        SUM(total_tokens) as tokens,
                        SUM(cost) as cost
                    FROM usage_records
                    WHERE timestamp >= ? AND timestamp <= ?
                    GROUP BY model
                """, (start_iso, end_iso))
                
                for row in cursor.fetchall():
                    summary.by_model[row[0]] = {
                        "requests": row[1],
                        "tokens": row[2],
                        "cost": row[3],
                    }
                
                # Get by user (top 10)
                cursor.execute("""
                    SELECT
                        user_id,
                        COUNT(*) as requests,
                        SUM(total_tokens) as tokens,
                        SUM(cost) as cost
                    FROM usage_records
                    WHERE timestamp >= ? AND timestamp <= ? AND user_id IS NOT NULL
                    GROUP BY user_id
                    ORDER BY cost DESC
                    LIMIT 10
                """, (start_iso, end_iso))
                
                for row in cursor.fetchall():
                    summary.by_user[str(row[0])] = {
                        "requests": row[1],
                        "tokens": row[2],
                        "cost": row[3],
                    }
                
                # Get by channel (top 10)
                cursor.execute("""
                    SELECT
                        channel_id,
                        COUNT(*) as requests,
                        SUM(total_tokens) as tokens,
                        SUM(cost) as cost
                    FROM usage_records
                    WHERE timestamp >= ? AND timestamp <= ? AND channel_id IS NOT NULL
                    GROUP BY channel_id
                    ORDER BY cost DESC
                    LIMIT 10
                """, (start_iso, end_iso))
                
                for row in cursor.fetchall():
                    summary.by_channel[str(row[0])] = {
                        "requests": row[1],
                        "tokens": row[2],
                        "cost": row[3],
                    }
                
        except Exception as e:
            logger.error(f"Failed to get usage summary: {e}", exc_info=True)
        
        return summary

    async def get_cost_projection(
        self,
        days_ahead: int = 30,
    ) -> Dict[str, Any]:
        """Project costs based on recent usage."""
        # Get last 7 days of usage for baseline
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT
                        SUM(cost) as total_cost,
                        SUM(total_tokens) as total_tokens,
                        COUNT(*) as total_requests
                    FROM usage_records
                    WHERE timestamp >= ?
                """, (week_ago.isoformat(),))
                
                row = cursor.fetchone()
                if not row or not row[0]:
                    return {
                        "projected_cost": 0.0,
                        "daily_average": 0.0,
                        "weekly_average": 0.0,
                        "monthly_projection": 0.0,
                        "confidence": "low",
                        "message": "Insufficient data for projection",
                    }
                
                weekly_cost = row[0]
                weekly_tokens = row[1]
                weekly_requests = row[2]
                
                daily_avg_cost = weekly_cost / 7
                daily_avg_tokens = weekly_tokens / 7
                daily_avg_requests = weekly_requests / 7
                
                projected_cost = daily_avg_cost * days_ahead
                monthly_projection = daily_avg_cost * 30
                
                # Determine confidence based on data volume
                if weekly_requests < 10:
                    confidence = "low"
                elif weekly_requests < 50:
                    confidence = "medium"
                else:
                    confidence = "high"
                
                return {
                    "projected_cost": round(projected_cost, 4),
                    "daily_average": round(daily_avg_cost, 4),
                    "weekly_average": round(weekly_cost, 4),
                    "monthly_projection": round(monthly_projection, 4),
                    "daily_tokens_average": int(daily_avg_tokens),
                    "daily_requests_average": round(daily_avg_requests, 1),
                    "confidence": confidence,
                    "projection_period_days": days_ahead,
                    "formatted": {
                        "projected": format_cost(projected_cost),
                        "daily": format_cost(daily_avg_cost),
                        "weekly": format_cost(weekly_cost),
                        "monthly": format_cost(monthly_projection),
                    },
                }
                
        except Exception as e:
            logger.error(f"Failed to calculate cost projection: {e}", exc_info=True)
            return {
                "error": str(e),
                "projected_cost": 0.0,
            }

    async def get_recent_records(
        self,
        limit: int = 50,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> List[UsageRecord]:
        """Get recent usage records."""
        records = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM usage_records WHERE 1=1"
                params = []
                
                if provider:
                    query += " AND provider = ?"
                    params.append(provider)
                if model:
                    query += " AND model = ?"
                    params.append(model)
                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                
                for row in cursor.fetchall():
                    records.append(UsageRecord(
                        id=row[0],
                        timestamp=row[1],
                        provider=row[2],
                        model=row[3],
                        input_tokens=row[4],
                        output_tokens=row[5],
                        cached_tokens=row[6],
                        total_tokens=row[7],
                        cost=row[8],
                        request_type=row[9],
                        channel_id=row[10],
                        user_id=row[11],
                        success=bool(row[12]),
                        latency_ms=row[13],
                        metadata=json.loads(row[14]) if row[14] else {},
                    ))
                
        except Exception as e:
            logger.error(f"Failed to get recent records: {e}", exc_info=True)
        
        return records

    async def get_daily_costs(
        self,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get daily cost breakdown for the last N days."""
        result = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
                
                cursor.execute("""
                    SELECT
                        date,
                        total_requests,
                        total_input_tokens,
                        total_output_tokens,
                        total_cost,
                        provider_breakdown,
                        model_breakdown
                    FROM daily_aggregates
                    WHERE date >= ?
                    ORDER BY date DESC
                """, (cutoff,))
                
                for row in cursor.fetchall():
                    result.append({
                        "date": row[0],
                        "requests": row[1],
                        "input_tokens": row[2],
                        "output_tokens": row[3],
                        "cost": row[4],
                        "formatted_cost": format_cost(row[4]),
                        "by_provider": json.loads(row[5]) if row[5] else {},
                        "by_model": json.loads(row[6]) if row[6] else {},
                    })
                
        except Exception as e:
            logger.error(f"Failed to get daily costs: {e}", exc_info=True)
        
        return result

    async def export_usage_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        format: str = "json",
    ) -> Union[str, List[Dict[str, Any]]]:
        """Export usage data for a date range."""
        if start_date is None:
            start_date = datetime.now(timezone.utc) - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        
        records = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM usage_records
                    WHERE timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp
                """, (start_date.isoformat(), end_date.isoformat()))
                
                for row in cursor.fetchall():
                    records.append({
                        "id": row[0],
                        "timestamp": row[1],
                        "provider": row[2],
                        "model": row[3],
                        "input_tokens": row[4],
                        "output_tokens": row[5],
                        "cached_tokens": row[6],
                        "total_tokens": row[7],
                        "cost": row[8],
                        "request_type": row[9],
                        "channel_id": row[10],
                        "user_id": row[11],
                        "success": bool(row[12]),
                        "latency_ms": row[13],
                    })
                
        except Exception as e:
            logger.error(f"Failed to export usage data: {e}", exc_info=True)
        
        if format == "json":
            return json.dumps(records, indent=2)
        return records

    async def cleanup_old_records(
        self,
        days_to_keep: int = 365,
    ) -> int:
        """Remove records older than specified days."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_to_keep)).isoformat()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    "DELETE FROM usage_records WHERE timestamp < ?",
                    (cutoff,)
                )
                
                deleted = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleaned up {deleted} old usage records")
                return deleted
                
        except Exception as e:
            logger.error(f"Failed to cleanup old records: {e}", exc_info=True)
            return 0


# ============================================================================
# Global Instance
# ============================================================================

_usage_tracker: Optional[UsageTracker] = None


def get_usage_tracker() -> UsageTracker:
    """Get or create the global usage tracker."""
    global _usage_tracker
    if _usage_tracker is None:
        _usage_tracker = UsageTracker()
    return _usage_tracker


async def track_usage(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    request_type: str = "chat",
    channel_id: Optional[int] = None,
    user_id: Optional[int] = None,
    success: bool = True,
    latency_ms: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> UsageRecord:
    """Track a usage event."""
    tracker = get_usage_tracker()
    return await tracker.record_usage(
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        request_type=request_type,
        channel_id=channel_id,
        user_id=user_id,
        success=success,
        latency_ms=latency_ms,
        metadata=metadata,
    )


async def get_spending_report(
    time_frame: TimeFrame = TimeFrame.DAILY,
) -> UsageSummary:
    """Get a spending report for the specified time frame."""
    tracker = get_usage_tracker()
    return await tracker.get_usage_summary(time_frame)


async def get_cost_estimate(days_ahead: int = 30) -> Dict[str, Any]:
    """Get cost projection for the next N days."""
    tracker = get_usage_tracker()
    return await tracker.get_cost_projection(days_ahead)


async def get_daily_breakdown(days: int = 30) -> List[Dict[str, Any]]:
    """Get daily cost breakdown."""
    tracker = get_usage_tracker()
    return await tracker.get_daily_costs(days)


# ============================================================================
# Formatting Utilities
# ============================================================================

def format_summary_for_discord(summary: UsageSummary) -> str:
    """Format a usage summary for Discord display."""
    lines = [
        f"**Usage Report: {summary.time_frame.title()}**",
        f"Period: {summary.period_start[:10]} to {summary.period_end[:10]}",
        "",
        "**Overview:**",
        f"- Total Requests: {summary.total_requests:,}",
        f"- Successful: {summary.successful_requests:,} | Failed: {summary.failed_requests:,}",
        f"- Total Tokens: {summary.total_tokens:,}",
        f"- Input: {summary.total_input_tokens:,} | Output: {summary.total_output_tokens:,}",
        f"- **Total Cost: {format_cost(summary.total_cost)}**",
        f"- Avg Latency: {summary.average_latency_ms:.0f}ms",
    ]
    
    if summary.by_provider:
        lines.append("")
        lines.append("**By Provider:**")
        for provider, stats in sorted(
            summary.by_provider.items(),
            key=lambda x: x[1]["cost"],
            reverse=True
        ):
            lines.append(
                f"- {provider}: {stats['requests']:,} reqs, "
                f"{stats['tokens']:,} tokens, {format_cost(stats['cost'])}"
            )
    
    if summary.by_model:
        lines.append("")
        lines.append("**Top Models:**")
        for model, stats in sorted(
            summary.by_model.items(),
            key=lambda x: x[1]["cost"],
            reverse=True
        )[:5]:
            lines.append(
                f"- {model}: {stats['requests']:,} reqs, {format_cost(stats['cost'])}"
            )
    
    return "\n".join(lines)


def format_projection_for_discord(projection: Dict[str, Any]) -> str:
    """Format a cost projection for Discord display."""
    if "error" in projection:
        return f"**Cost Projection Error:** {projection['error']}"
    
    formatted = projection.get("formatted", {})
    
    lines = [
        "**Cost Projection**",
        f"Confidence: {projection.get('confidence', 'unknown').title()}",
        "",
        f"- Daily Average: {formatted.get('daily', 'N/A')}",
        f"- Weekly Average: {formatted.get('weekly', 'N/A')}",
        f"- Monthly Projection: {formatted.get('monthly', 'N/A')}",
        "",
        f"- Avg Daily Tokens: {projection.get('daily_tokens_average', 0):,}",
        f"- Avg Daily Requests: {projection.get('daily_requests_average', 0):.1f}",
    ]
    
    return "\n".join(lines)
