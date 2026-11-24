import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
for p in (SRC_DIR, SRC_DIR / "probe"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
