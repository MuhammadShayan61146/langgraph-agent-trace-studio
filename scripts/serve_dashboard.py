"""Launch the live dashboard from the repo checkout without installing the package.

    python scripts/serve_dashboard.py --open
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_research.viz.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
