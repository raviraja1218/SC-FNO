"""Project-level pytest configuration.

Ensures src/ is on sys.path before any test module is imported, so that
`physics.*`, `data.*`, `physics_informed.*` imports work identically
inside pytest and inside standalone scripts.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
