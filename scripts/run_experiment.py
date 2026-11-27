"""Run an experiment from the repo checkout without installing the package.

    python scripts/run_experiment.py --list
    python scripts/run_experiment.py fault-drill --export-dir out/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_research.experiments.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
