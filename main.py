#!/usr/bin/env python3
"""Face verification workflow with optional alignment guidance (CLI entry point)."""
from __future__ import annotations

import sys
from pathlib import Path

from utils.cli import parse_main_args

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.attendance import (
    AttendancePipeline,
    DEFAULT_ATTENDANCE_LOG,
    DEFAULT_FACEBANK,
    DEFAULT_POWER_LOG,
    DEFAULT_SPOOF_WEIGHTS,
    DEFAULT_WEIGHTS,
)


def main() -> None:
    args = parse_main_args(
        default_weights=DEFAULT_WEIGHTS,
        default_facebank=DEFAULT_FACEBANK,
        default_spoof_weights=DEFAULT_SPOOF_WEIGHTS,
        default_attendance_log=DEFAULT_ATTENDANCE_LOG,
        default_power_log=DEFAULT_POWER_LOG,
    )
    pipeline = AttendancePipeline(args)
    pipeline.run_cli()


if __name__ == "__main__":
    main()

