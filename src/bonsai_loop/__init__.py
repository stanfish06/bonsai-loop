import sys
from pathlib import Path

_BONSAI = Path(__file__).resolve().parents[2] / "Bonsai-data-representation"
if str(_BONSAI) not in sys.path:
    sys.path.insert(0, str(_BONSAI))

import bonsai
sys.modules[f"{__name__}.bonsai"] = bonsai

__all__ = ["bonsai"]
