import sys
from pathlib import Path

_repo_root = Path(__file__).parent.parent
# vineland_runner is a top-level package at the repo root
sys.path.insert(0, str(_repo_root))
# vineland_api package lives inside vineland_api/ subdirectory
sys.path.insert(0, str(_repo_root / "vineland_api"))

from vineland_api.main import app  # noqa: E402 — path setup must come first
