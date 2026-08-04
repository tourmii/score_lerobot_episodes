"""Web application for reviewing LeRobot episode quality.
Start it with ``python -m app``.
"""

import sys
from pathlib import Path

# Run straight from a checkout, without `pip install -e .` first.  Harmless when
# the package is installed: an existing import wins over this path entry.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

__all__ = ["main", "service"]
