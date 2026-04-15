#!/usr/bin/env python3
"""Generate comparison tables and charts from benchmark results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_results(results_dir: Path | None = None) -> list[dict]:
    """Load all JSON result files from a directory."""
    d = results_dir or RESULTS_DIR
    if not d.exists():
        return []
    results = []
    for f in sorted(d.glob("*.json")):
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def generate_comparison_table(results: list[dict]) -> str:
    """Generate a markdown comparison table from results."""
    if not results:
        return "No results to display."

    header = (
        "| Tool | Conda-forge | Cold Install (s) | Warm Install (s) | "
        "Lockfile (s) | Disk (MB) | Tests (s) |"
    )
    sep = "|------|-------------|------------------|------------------|" \
          "--------------|-----------|-----------|"
    rows = [header, sep]

    for r in sorted(results, key=lambda x: x.get("warm_install", {}).get("median", 999)):
        conda = "Yes" if r.get("supports_conda_forge") else "No"
        cold = r.get("cold_install", {}).get("median", "N/A")
        warm = r.get("warm_install", {}).get("median", "N/A")
        lock = r.get("lockfile_gen", {}).get("median", "N/A")
        disk = r.get("disk_footprint_mb", "N/A")
        tests = r.get("test_execution", {}).get("median", "N/A")

        cold_s = f"{cold:.1f}" if isinstance(cold, (int, float)) else str(cold)
        warm_s = f"{warm:.1f}" if isinstance(warm, (int, float)) else str(warm)
        lock_s = f"{lock:.1f}" if isinstance(lock, (int, float)) else str(lock)
        disk_s = f"{disk:.0f}" if isinstance(disk, (int, float)) else str(disk)
        tests_s = f"{tests:.1f}" if isinstance(tests, (int, float)) else str(tests)

        rows.append(
            f"| {r['tool']:<4} | {conda:<11} | {cold_s:>16} | {warm_s:>16} | "
            f"{lock_s:>12} | {disk_s:>9} | {tests_s:>9} |"
        )

    return "\n".join(rows)


def generate_charts(results: list[dict], output_dir: str | None = None) -> None:
    """Generate comparison bar charts from results."""
    out = Path(output_dir) if output_dir else RESULTS_DIR.parent / "charts"
    out.mkdir(parents=True, exist_ok=True)

    tools = [r["tool"] for r in results]
    x = np.arange(len(tools))
    bar_width = 0.35

    # --- Install comparison ---
    fig, ax = plt.subplots(figsize=(12, 6))
    cold = [r.get("cold_install", {}).get("median", 0) for r in results]
    warm = [r.get("warm_install", {}).get("median", 0) for r in results]
    ax.bar(x - bar_width / 2, cold, bar_width, label="Cold Install")
    ax.bar(x + bar_width / 2, warm, bar_width, label="Warm Install")
    ax.set_xlabel("Package Manager")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Install Time Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(tools)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "install_comparison.png", dpi=150)
    plt.close(fig)

    # --- Disk footprint ---
    fig, ax = plt.subplots(figsize=(10, 6))
    disk = [r.get("disk_footprint_mb", 0) for r in results]
    colors = ["#2ecc71" if r.get("supports_conda_forge") else "#3498db" for r in results]
    ax.bar(tools, disk, color=colors)
    ax.set_xlabel("Package Manager")
    ax.set_ylabel("Size (MB)")
    ax.set_title("Environment Disk Footprint")
    plt.tight_layout()
    plt.savefig(out / "disk_footprint.png", dpi=150)
    plt.close(fig)

    # --- Solver stress ---
    fig, ax = plt.subplots(figsize=(12, 6))
    conflict = []
    incremental = []
    bump = []
    for r in results:
        ss = r.get("solver_stress", {})
        conflict.append(ss.get("conflict_resolution", {}).get("time", 0))
        incremental.append(ss.get("incremental_add", {}).get("time", 0))
        bump.append(ss.get("version_bump", {}).get("time", 0))

    width = 0.25
    ax.bar(x - width, conflict, width, label="Conflict Resolution")
    ax.bar(x, incremental, width, label="Incremental Add")
    ax.bar(x + width, bump, width, label="Version Bump")
    ax.set_xlabel("Package Manager")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Solver Stress Test")
    ax.set_xticks(x)
    ax.set_xticklabels(tools)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "solver_stress.png", dpi=150)
    plt.close(fig)

    print(f"Charts saved to {out}/")


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR
    results = load_results(results_dir)
    if not results:
        print(f"No results found in {results_dir}")
        sys.exit(1)

    print("\n## Benchmark Results\n")
    print(generate_comparison_table(results))
    print()
    generate_charts(results)


if __name__ == "__main__":
    main()
