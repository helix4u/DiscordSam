import logging
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class UsageRecord:
    timestamp: datetime
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float

# Default pricing (approximate, per 1M tokens) as of late 2024/early 2025
# Format: (prompt_price, completion_price)
PRICING_PER_1M: Dict[str, Tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o1-preview": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    # Anthropic (via OpenRouter or similar)
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    "claude-3-opus": (15.00, 75.00),
    # Google
    "gemini-1.5-pro": (3.50, 10.50),
    "gemini-1.5-flash": (0.075, 0.30),
    # Mistral
    "mistral-large": (2.00, 6.00),
    "mistral-small": (0.20, 0.60),
    # Local/Free
    "local": (0.0, 0.0),
    "lm-studio": (0.0, 0.0),
    "ollama": (0.0, 0.0),
}

DB_PATH = "usage_stats.db"

class UsageTracker:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS usage_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        model TEXT NOT NULL,
                        prompt_tokens INTEGER,
                        completion_tokens INTEGER,
                        total_tokens INTEGER,
                        cost REAL
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize usage DB: {e}")

    def get_pricing(self, model: str) -> Tuple[float, float]:
        # Simple heuristic to match model names to pricing keys
        model_lower = model.lower()
        for key, prices in PRICING_PER_1M.items():
            if key in model_lower:
                return prices
        return (0.0, 0.0) # Default to free/unknown

    def track_usage(self, provider: str, model: str, usage_data: Any):
        """
        Record usage from an API response.
        usage_data can be an object with prompt_tokens, completion_tokens, etc.
        """
        if not usage_data:
            return

        try:
            # Handle OpenAI usage object or dict
            if isinstance(usage_data, dict):
                prompt_tokens = usage_data.get("prompt_tokens", 0)
                completion_tokens = usage_data.get("completion_tokens", 0)
                total_tokens = usage_data.get("total_tokens", 0)
            else:
                prompt_tokens = getattr(usage_data, "prompt_tokens", 0)
                completion_tokens = getattr(usage_data, "completion_tokens", 0)
                total_tokens = getattr(usage_data, "total_tokens", 0)

            prompt_price, completion_price = self.get_pricing(model)
            
            # Calculate cost
            cost = (prompt_tokens / 1_000_000 * prompt_price) + \
                   (completion_tokens / 1_000_000 * completion_price)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO usage_logs (timestamp, provider, model, prompt_tokens, completion_tokens, total_tokens, cost)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (datetime.now().isoformat(), provider, model, prompt_tokens, completion_tokens, total_tokens, cost))
                conn.commit()
                
            logger.info(f"Usage tracked: {model} | {total_tokens} tokens | ${cost:.6f}")

        except Exception as e:
            logger.error(f"Failed to track usage: {e}", exc_info=True)

    def get_report(self, timeframe: str = "daily") -> str:
        """
        Generate a text report for the given timeframe.
        timeframe: "daily", "monthly", "all"
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if timeframe == "daily":
                    # Group by day
                    cursor.execute("""
                        SELECT date(timestamp) as day, SUM(total_tokens), SUM(cost)
                        FROM usage_logs
                        GROUP BY day
                        ORDER BY day DESC
                        LIMIT 7
                    """)
                    rows = cursor.fetchall()
                    report = "**Daily Usage Report (Last 7 Days):**\n"
                    for row in rows:
                        report += f"- {row[0]}: {row[1]:,} tokens | ${row[2]:.4f}\n"
                        
                elif timeframe == "monthly":
                    # Group by month
                    cursor.execute("""
                        SELECT strftime('%Y-%m', timestamp) as month, SUM(total_tokens), SUM(cost)
                        FROM usage_logs
                        GROUP BY month
                        ORDER BY month DESC
                        LIMIT 12
                    """)
                    rows = cursor.fetchall()
                    report = "**Monthly Usage Report:**\n"
                    for row in rows:
                        report += f"- {row[0]}: {row[1]:,} tokens | ${row[2]:.4f}\n"
                
                elif timeframe == "models":
                     # Group by model
                    cursor.execute("""
                        SELECT model, SUM(total_tokens), SUM(cost)
                        FROM usage_logs
                        GROUP BY model
                        ORDER BY SUM(cost) DESC
                    """)
                    rows = cursor.fetchall()
                    report = "**Usage by Model (All Time):**\n"
                    for row in rows:
                        report += f"- {row[0]}: {row[1]:,} tokens | ${row[2]:.4f}\n"

                else:
                    return "Invalid timeframe."

                return report
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return "Error generating report."

# Global instance
usage_tracker = UsageTracker()
