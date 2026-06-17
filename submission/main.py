from __future__ import annotations

import sys
from pathlib import Path


try:
    from ptcg_ai.submission import agent
except ModuleNotFoundError:
    repo_src = Path(__file__).resolve().parents[1] / "src"
    if repo_src.exists():
        sys.path.insert(0, str(repo_src))
    from ptcg_ai.submission import agent
