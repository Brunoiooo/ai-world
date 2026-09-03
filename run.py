"""Convenience launcher without installing the package: ``python run.py``."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_world.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
