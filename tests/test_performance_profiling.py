"""Tests for opt-in phase timing and process-tree resource monitoring."""

import csv
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


MONITOR_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts/monitor_process_tree.py"
)
MONITOR_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "monitor_process_tree",
    MONITOR_SCRIPT_PATH,
)
monitor_process_tree = importlib.util.module_from_spec(MONITOR_SCRIPT_SPEC)
MONITOR_SCRIPT_SPEC.loader.exec_module(monitor_process_tree)


def test_process_tree_monitor_writes_structured_samples_and_summary(tmp_path):
    resources_path = tmp_path / "resources.csv"
    summary_path = tmp_path / "summary.json"
    command = (
        "import subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(0.1)']); "
        "time.sleep(0.15); child.wait()"
    )

    exit_code = monitor_process_tree.main(
        [
            "--resources-csv",
            str(resources_path),
            "--summary-json",
            str(summary_path),
            "--interval-seconds",
            "0.02",
            "--",
            sys.executable,
            "-c",
            command,
        ]
    )

    assert exit_code == 0
    with resources_path.open(newline="", encoding="utf-8") as input_file:
        samples = list(csv.DictReader(input_file))
    assert samples
    assert set(samples[0]) == set(monitor_process_tree.SAMPLE_FIELDS)
    assert max(int(sample["process_count"]) for sample in samples) >= 2
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == 1
    assert summary["exit_code"] == 0
    assert summary["sample_count"] == len(samples)
    assert summary["max_process_count"] >= 2
    assert summary["peak_process_tree_rss_bytes"] >= 0


def test_cpu_accounting_retains_exited_child_ticks_and_uses_linux_percentages():
    last_seen_cpu_ticks = {}
    first_processes = {
        100: {"start_ticks": 10, "cpu_ticks": 20},
        101: {"start_ticks": 20, "cpu_ticks": 80},
    }
    first_total = monitor_process_tree._cumulative_tree_cpu_ticks(
        first_processes,
        set(first_processes),
        last_seen_cpu_ticks,
    )

    # The child consumed CPU and exited before the next sample. Its 80 ticks
    # must remain in the cumulative tree total.
    second_processes = {
        100: {"start_ticks": 10, "cpu_ticks": 50},
    }
    second_total = monitor_process_tree._cumulative_tree_cpu_ticks(
        second_processes,
        set(second_processes),
        last_seen_cpu_ticks,
    )

    # PID 101 is reused, but a different start time makes it a new identity.
    third_processes = {
        100: {"start_ticks": 10, "cpu_ticks": 70},
        101: {"start_ticks": 30, "cpu_ticks": 40},
    }
    third_total = monitor_process_tree._cumulative_tree_cpu_ticks(
        third_processes,
        set(third_processes),
        last_seen_cpu_ticks,
    )

    assert [first_total, second_total, third_total] == [100, 130, 190]
    assert first_total <= second_total <= third_total
    summary = monitor_process_tree._build_summary(
        SimpleNamespace(command=["workload"], interval_seconds=1.0),
        [
            {
                "tree_cpu_seconds": 1.0,
                "tree_cpu_percent": 0.0,
                "tree_rss_bytes": 0,
                "tree_pss_bytes": "",
                "tree_swap_bytes": 0,
                "system_swap_delta_bytes": 0,
                "process_count": 1,
            },
            {
                "tree_cpu_seconds": 2.4,
                "tree_cpu_percent": 140.0,
                "tree_rss_bytes": 0,
                "tree_pss_bytes": "",
                "tree_swap_bytes": 0,
                "system_swap_delta_bytes": 0,
                "process_count": 2,
            },
        ],
        wall_clock_seconds=1.0,
        exit_code=0,
    )

    assert summary["average_process_tree_cpu_percent"] == 240.0
    assert summary["peak_process_tree_cpu_percent"] == 140.0
    assert summary["average_process_tree_cpu_percent"] >= 0.0
    assert summary["peak_process_tree_cpu_percent"] > 100.0
