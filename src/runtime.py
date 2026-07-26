"""Runtime helpers for the self-contained submission layout."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = PROJECT_ROOT / ".vendor"


def enable_vendored_dependencies() -> None:
    if VENDOR_DIR.exists():
        vendor = str(VENDOR_DIR)
        if vendor not in sys.path:
            sys.path.insert(0, vendor)
