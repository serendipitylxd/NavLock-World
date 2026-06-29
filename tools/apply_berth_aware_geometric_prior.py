#!/usr/bin/env python3
"""Apply the berth-aware geometric prior to VLM semantic ship-intention predictions.

This is the preferred entry point. The legacy
``tools/apply_berth_ship_intention_guard.py`` script remains available for
backward-compatible commands.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.apply_berth_ship_intention_guard import main  # noqa: E402


if __name__ == "__main__":
    main(description=__doc__)
