"""Unified test entry (delegates to evaluation script)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/project")

if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from scripts.evaluate import main

    main()
