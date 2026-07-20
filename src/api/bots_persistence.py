"""
Simple JSON file-based persistence for Trading Bots.

Trading bots are kept in memory during the process lifetime (see
`src.api.endpoints.trading_bots`), and mirrored to a JSON file on disk so
that they survive process restarts in environments without a database.
"""

import json
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

BOTS_FILE = Path(__file__).resolve().parents[2] / "data" / "trading_bots.json"


def load_bots() -> Dict[str, dict]:
    """Load persisted trading bots from disk. Returns {} if not found or invalid."""
    try:
        with open(BOTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing trading bots JSON ({BOTS_FILE}): {e}")
        return {}


def save_bots(bots: Dict[str, dict]) -> None:
    """Persist the trading bots dict to disk."""
    try:
        BOTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BOTS_FILE, "w", encoding="utf-8") as f:
            json.dump(bots, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.error(f"Error saving trading bots JSON ({BOTS_FILE}): {e}")
