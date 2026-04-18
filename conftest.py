# conftest.py — ensures `src/` is on the Python path for pytest.
# Prefer `pip install -e .` instead, but this works as a fallback.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
