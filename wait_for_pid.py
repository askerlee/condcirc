#!/usr/bin/env python3
"""Run a command after a Linux process exits."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Wait for a Linux PID to exit, then run a command."
    )
    parser.add_argument("pid", type=int, help="PID to wait for")
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="seconds between checks (default: 5)",
    )
    if "--" not in arguments:
        parser.error("provide a command after --")
    separator = arguments.index("--")
    args = parser.parse_args(arguments[:separator])
    args.command = arguments[separator + 1 :]
    if args.pid <= 0:
        parser.error("pid must be positive")
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if not args.command:
        parser.error("provide a command after --")
    return args


def process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    state = stat[stat.rfind(")") + 2]
    return state != "Z"


def wait_for_process(pid: int, interval: float) -> None:
    while process_is_running(pid):
        time.sleep(interval)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"Waiting for PID {args.pid} to exit...", flush=True)
    wait_for_process(args.pid, args.interval)
    print(f"PID {args.pid} has exited; running: {args.command!r}", flush=True)
    return subprocess.run(args.command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())