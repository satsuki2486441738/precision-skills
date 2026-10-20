#!/usr/bin/env python3
"""Compatibility entry point; reusable replay lives in the installable skill."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'skills/precision-skill/scripts'))
from replay_bundle import check_subset, main, replay, safe_path, write


if __name__ == '__main__':
    raise SystemExit(main())
