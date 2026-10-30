"""pytest bootstrap: make src/ importable for all test modules.

This runs before any test collection, so `from data.config_loader import ...`
and `from physics.kinetics import ...` resolve correctly regardless of which
directory pytest is invoked from.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
