#!/usr/bin/env python3
"""Run a command while sampling Linux process-tree resources to CSV.

The monitored command is started in its own session.  Samples include its root
process, descendants discovered by PPID, and processes retaining that session's
process group, which covers spawned multiprocessing workers between polls.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SAMPLE_FIELDS = (
    "elapsed_seconds",
    "process_count",
    "tree_cpu_percent",
    "tree_cpu_seconds",
    "tree_rss_bytes",
    "tree_pss_bytes",
    "tree_pss_process_count",
    "tree_swap_bytes",
    "system_swap_used_bytes",
    "system_swap_delta_bytes",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Monitor a Linux process tree while a command runs."
    )
    parser.add_argument("--resources-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=1.0,
        help="Sampling interval in seconds; defaults to 1.0.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive.")
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("supply a command after --")
    return args


def _read_process_stats() -> dict[int, dict]:
    """Return minimal stat fields for processes visible in procfs."""
    processes = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            after_name = stat.rsplit(")", maxsplit=1)[1].strip().split()
            processes[int(entry.name)] = {
                "ppid": int(after_name[1]),
                "pgrp": int(after_name[2]),
                "cpu_ticks": int(after_name[11]) + int(after_name[12]),
                "start_ticks": int(after_name[19]),
            }
        except (OSError, IndexError, ValueError):
            continue
    return processes


def _tree_process_ids(root_pid: int, processes: dict[int, dict]) -> set[int]:
    """Find the root, descendants, and its dedicated process group."""
    tree = {root_pid} if root_pid in processes else set()
    changed = True
    while changed:
        changed = False
        for pid, stats in processes.items():
            if stats["ppid"] in tree and pid not in tree:
                tree.add(pid)
                changed = True
    tree.update(
        pid for pid, stats in processes.items() if stats["pgrp"] == root_pid
    )
    return tree


def _status_bytes(pid: int, field_name: str) -> int:
    try:
        for line in (Path("/proc") / str(pid) / "status").read_text(
            encoding="utf-8"
        ).splitlines():
            if line.startswith(f"{field_name}:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _pss_bytes(pid: int) -> int | None:
    try:
        for line in (Path("/proc") / str(pid) / "smaps_rollup").read_text(
            encoding="utf-8"
        ).splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _system_swap_used_bytes() -> int:
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", maxsplit=1)
            values[key] = int(value.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return 0
    return max(0, values.get("SwapTotal", 0) - values.get("SwapFree", 0))


def _cumulative_tree_cpu_ticks(
    processes: dict[int, dict],
    process_ids: set[int],
    last_seen_cpu_ticks: dict[tuple[int, int], int],
) -> int:
    """Retain observed CPU totals after child exit, keyed by process identity.

    ``/proc/<pid>/stat`` CPU time is cumulative for one process but disappears
    when that process exits.  The start time distinguishes a reused PID from
    its previous process, allowing both completed CPU totals to remain counted.
    """
    for pid in process_ids:
        stats = processes[pid]
        process_identity = (pid, stats["start_ticks"])
        last_seen_cpu_ticks[process_identity] = max(
            last_seen_cpu_ticks.get(process_identity, 0),
            stats["cpu_ticks"],
        )
    return sum(last_seen_cpu_ticks.values())


def _sample(
    root_pid: int,
    started_at: float,
    previous_cpu_ticks: int | None,
    previous_sample_at: float | None,
    initial_system_swap_bytes: int,
    last_seen_cpu_ticks: dict[tuple[int, int], int],
) -> tuple[dict, int, float]:
    sampled_at = time.perf_counter()
    processes = _read_process_stats()
    process_ids = _tree_process_ids(root_pid, processes)
    cpu_ticks = _cumulative_tree_cpu_ticks(
        processes,
        process_ids,
        last_seen_cpu_ticks,
    )
    elapsed_since_last = (
        None if previous_sample_at is None else sampled_at - previous_sample_at
    )
    if previous_cpu_ticks is None or not elapsed_since_last:
        cpu_percent = 0.0
    else:
        cpu_percent = (
            (cpu_ticks - previous_cpu_ticks)
            / float(os.sysconf("SC_CLK_TCK"))
            / elapsed_since_last
            * 100.0
        )

    rss_bytes = sum(_status_bytes(pid, "VmRSS") for pid in process_ids)
    swap_bytes = sum(_status_bytes(pid, "VmSwap") for pid in process_ids)
    pss_values = [_pss_bytes(pid) for pid in process_ids]
    available_pss_values = [value for value in pss_values if value is not None]
    system_swap_bytes = _system_swap_used_bytes()
    return (
        {
            "elapsed_seconds": sampled_at - started_at,
            "process_count": len(process_ids),
            "tree_cpu_percent": cpu_percent,
            "tree_cpu_seconds": cpu_ticks / float(os.sysconf("SC_CLK_TCK")),
            "tree_rss_bytes": rss_bytes,
            "tree_pss_bytes": sum(available_pss_values) if available_pss_values else "",
            "tree_pss_process_count": len(available_pss_values),
            "tree_swap_bytes": swap_bytes,
            "system_swap_used_bytes": system_swap_bytes,
            "system_swap_delta_bytes": system_swap_bytes - initial_system_swap_bytes,
        },
        cpu_ticks,
        sampled_at,
    )


def _number_or_none(value):
    return None if value == "" else value


def _format_bytes(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) / (1024 ** 2):.1f} MiB"


def _build_summary(
    args,
    samples: list[dict],
    wall_clock_seconds: float,
    exit_code: int,
) -> dict:
    """Summarize process-tree samples without changing the output schema."""
    cpu_samples = [sample["tree_cpu_percent"] for sample in samples[1:]]
    pss_samples = [
        _number_or_none(sample["tree_pss_bytes"])
        for sample in samples
        if _number_or_none(sample["tree_pss_bytes"]) is not None
    ]
    cumulative_cpu_seconds = (
        float(samples[-1]["tree_cpu_seconds"]) if samples else 0.0
    )
    return {
        "schema_version": 1,
        "command": args.command,
        "exit_code": int(exit_code),
        "sampling_interval_seconds": float(args.interval_seconds),
        "sample_count": int(len(samples)),
        "wall_clock_seconds": wall_clock_seconds,
        "average_process_tree_cpu_percent": (
            cumulative_cpu_seconds / wall_clock_seconds * 100.0
            if wall_clock_seconds > 0.0
            else 0.0
        ),
        "peak_process_tree_cpu_percent": max(cpu_samples, default=0.0),
        "peak_process_tree_rss_bytes": max(
            (sample["tree_rss_bytes"] for sample in samples),
            default=0,
        ),
        "peak_process_tree_pss_bytes": max(pss_samples, default=None),
        "peak_process_tree_swap_bytes": max(
            (sample["tree_swap_bytes"] for sample in samples),
            default=0,
        ),
        "peak_system_swap_delta_bytes": max(
            (sample["system_swap_delta_bytes"] for sample in samples),
            default=0,
        ),
        "max_process_count": max(
            (sample["process_count"] for sample in samples),
            default=0,
        ),
    }


def monitor_command(args) -> int:
    resources_path = Path(args.resources_csv)
    summary_path = Path(args.summary_json)
    resources_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    process = subprocess.Popen(args.command, start_new_session=True)
    initial_system_swap_bytes = _system_swap_used_bytes()
    samples = []
    previous_cpu_ticks = None
    previous_sample_at = None
    last_seen_cpu_ticks = {}
    with resources_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        while True:
            sample, previous_cpu_ticks, previous_sample_at = _sample(
                process.pid,
                started_at,
                previous_cpu_ticks,
                previous_sample_at,
                initial_system_swap_bytes,
                last_seen_cpu_ticks,
            )
            writer.writerow(sample)
            output_file.flush()
            samples.append(sample)
            if process.poll() is not None:
                break
            time.sleep(args.interval_seconds)

    exit_code = process.wait()
    wall_clock_seconds = time.perf_counter() - started_at
    summary = _build_summary(args, samples, wall_clock_seconds, exit_code)
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    print(
        "Resource summary: "
        f"wall={summary['wall_clock_seconds']:.1f}s, "
        f"cpu avg/peak={summary['average_process_tree_cpu_percent']:.1f}%/"
        f"{summary['peak_process_tree_cpu_percent']:.1f}%, "
        f"RSS peak={_format_bytes(summary['peak_process_tree_rss_bytes'])}, "
        f"PSS peak={_format_bytes(summary['peak_process_tree_pss_bytes'])}, "
        f"tree swap peak={_format_bytes(summary['peak_process_tree_swap_bytes'])}, "
        f"system swap delta peak={_format_bytes(summary['peak_system_swap_delta_bytes'])}, "
        f"processes max={summary['max_process_count']}"
    )
    print(f"Saved resource samples to {resources_path}")
    print(f"Saved resource summary to {summary_path}")
    return exit_code


def main(argv=None) -> int:
    return monitor_command(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
