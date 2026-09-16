"""
Entry point for the fly companion project.

Kept deliberately thin: it only decides WHAT to run, never how anything
works. Today it runs the Phase 1 escape-circuit demo; in Phase 4 this is
where the companion window gets launched instead.

Usage:
    python main.py
"""

import sys

from scripts.demo_escape_circuit import main as run_demo


if __name__ == "__main__":
    sys.exit(run_demo())