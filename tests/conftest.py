"""Pytest configuration and environment fixtures."""

import sys
from pathlib import Path

# Ensure src is in sys.path across all test modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Initialize PyTorch OpenMP runtime before scikit-learn to prevent Windows WinError 1114
try:
    import torch  # noqa: F401
except ImportError:
    pass
