"""Convenience training entry using default config."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/project")
SCRIPT = ROOT / "scripts" / "train.py"

if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from scripts.train import main

    main()
