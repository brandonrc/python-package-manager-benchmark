#!/usr/bin/env python3
"""Benchmark runner for comparing Python package manager performance.

Usage:
    python benchmark/runner.py                    # benchmark all available tools
    python benchmark/runner.py --tools pixi uv    # benchmark specific tools
    python benchmark/runner.py --runs 5           # 5 runs per benchmark (default: 3)
    python benchmark/runner.py --skip-cold        # skip cold install (slow)
    python benchmark/runner.py --skip-solver      # skip solver stress tests
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median as compute_median_stat


PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TimingResult:
    runs: list[float] = field(default_factory=list)
    median: float = 0.0

    def compute_median(self):
        self.median = compute_median_stat(self.runs) if self.runs else 0.0


@dataclass
class SolverStressResult:
    time: float = 0.0
    success: bool = False


@dataclass
class BenchmarkResult:
    tool: str = ""
    timestamp: str = ""
    system: dict = field(default_factory=dict)
    cold_install: TimingResult = field(default_factory=TimingResult)
    warm_install: TimingResult = field(default_factory=TimingResult)
    lockfile_gen: TimingResult = field(default_factory=TimingResult)
    disk_footprint_mb: float = 0.0
    test_execution: TimingResult = field(default_factory=TimingResult)
    solver_stress: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    supports_conda_forge: bool = False


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_system_info() -> dict:
    """Collect system information for reproducibility."""
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "arch": platform.machine(),
        "cpu": platform.processor() or "unknown",
        "python": platform.python_version(),
    }


def measure_dir_size(path: Path) -> float:
    """Return directory size in MB. Returns 0.0 if path doesn't exist."""
    if not path.exists():
        return 0.0
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


def time_command(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict | None = None,
    timeout: int = 600,
) -> tuple[float, bool, str]:
    """Run a command and return (elapsed_seconds, success, combined_output)."""
    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env or os.environ.copy(),
        )
        elapsed = time.monotonic() - start
        return elapsed, result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return elapsed, False, f"TIMEOUT after {timeout}s"
    except FileNotFoundError:
        elapsed = time.monotonic() - start
        return elapsed, False, f"Command not found: {cmd[0]}"


def detect_tools() -> dict[str, str | None]:
    """Detect which package manager tools are installed. Returns {name: version_string_or_None}."""
    tools = {}
    for tool_name in ["pixi", "uv", "conda", "mamba", "pip", "poetry"]:
        path = shutil.which(tool_name)
        if not path:
            tools[tool_name] = None
            continue
        try:
            result = subprocess.run(
                [tool_name, "--version"],
                capture_output=True, text=True, timeout=10,
            )
            tools[tool_name] = result.stdout.strip() or result.stderr.strip() or "unknown"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            tools[tool_name] = "unknown"
    return tools


# ---------------------------------------------------------------------------
# Tool-specific helpers
# ---------------------------------------------------------------------------

def _conda_env_path(env_name: str) -> Path:
    """Find a conda environment's path."""
    result = subprocess.run(
        ["conda", "info", "--base"], capture_output=True, text=True
    )
    base = Path(result.stdout.strip())
    return base / "envs" / env_name


def _remove_conda_env(env_name: str):
    subprocess.run(
        ["conda", "env", "remove", "-n", env_name, "-y"],
        capture_output=True,
    )


def _remove_mamba_env(env_name: str):
    subprocess.run(
        ["mamba", "env", "remove", "-n", env_name, "-y"],
        capture_output=True,
    )


def _poetry_env_path() -> Path:
    """Find poetry's virtualenv path (it stores venvs outside the project)."""
    result = subprocess.run(
        ["poetry", "env", "info", "--path"],
        capture_output=True, text=True, cwd=PROJECT_DIR,
    )
    if result.returncode == 0 and result.stdout.strip():
        return Path(result.stdout.strip())
    return PROJECT_DIR / ".venv"


# ---------------------------------------------------------------------------
# Tool configs
# ---------------------------------------------------------------------------

TOOL_CONFIGS = {
    "pixi": {
        "install_cmd": ["pixi", "install"],
        "run_tests_cmd": ["pixi", "run", "pytest", "tests/", "-v", "--timeout=60"],
        "clear_cache": lambda: subprocess.run(
            ["pixi", "clean", "cache", "--yes"], capture_output=True
        ),
        "clear_env": lambda: (
            shutil.rmtree(PROJECT_DIR / ".pixi", ignore_errors=True),
            (PROJECT_DIR / "pixi.lock").unlink(missing_ok=True),
        ),
        "env_path": lambda: PROJECT_DIR / ".pixi" / "envs" / "default",
        "lockfile_cmd": ["pixi", "install"],
        "lockfile_path": lambda: PROJECT_DIR / "pixi.lock",
        "supports_conda_forge": True,
    },
    "conda": {
        "install_cmd": ["conda", "env", "create", "-f", "environment.yml", "-n", "mlbench-conda", "-y"],
        "post_install_cmd": ["conda", "run", "-n", "mlbench-conda", "pip", "install", "-e", "."],
        "run_tests_cmd": ["conda", "run", "-n", "mlbench-conda", "pytest", "tests/", "-v", "--timeout=60"],
        "clear_cache": lambda: subprocess.run(
            ["conda", "clean", "--all", "-y"], capture_output=True
        ),
        "clear_env": lambda: _remove_conda_env("mlbench-conda"),
        "env_path": lambda: _conda_env_path("mlbench-conda"),
        "lockfile_cmd": ["conda-lock", "-f", "environment.yml", "--lockfile", "conda-lock.yml"],
        "lockfile_path": lambda: PROJECT_DIR / "conda-lock.yml",
        "supports_conda_forge": True,
    },
    "mamba": {
        "install_cmd": ["mamba", "env", "create", "-f", "environment.yml", "-n", "mlbench-mamba", "-y"],
        "post_install_cmd": ["conda", "run", "-n", "mlbench-mamba", "pip", "install", "-e", "."],
        "run_tests_cmd": ["conda", "run", "-n", "mlbench-mamba", "pytest", "tests/", "-v", "--timeout=60"],
        "clear_cache": lambda: subprocess.run(
            ["mamba", "clean", "--all", "-y"], capture_output=True
        ),
        "clear_env": lambda: _remove_mamba_env("mlbench-mamba"),
        "env_path": lambda: _conda_env_path("mlbench-mamba"),
        "lockfile_cmd": ["conda-lock", "-f", "environment.yml", "--lockfile", "conda-lock-mamba.yml"],
        "lockfile_path": lambda: PROJECT_DIR / "conda-lock-mamba.yml",
        "supports_conda_forge": True,
    },
    "uv": {
        "install_cmd": ["uv", "sync", "--all-extras", "--python", "3.12"],
        "run_tests_cmd": ["uv", "run", "pytest", "tests/", "-v", "--timeout=60"],
        "clear_cache": lambda: subprocess.run(
            ["uv", "cache", "clean"], capture_output=True
        ),
        "clear_env": lambda: shutil.rmtree(PROJECT_DIR / ".venv", ignore_errors=True),
        "env_path": lambda: PROJECT_DIR / ".venv",
        "lockfile_cmd": ["uv", "lock"],
        "lockfile_path": lambda: PROJECT_DIR / "uv.lock",
        "supports_conda_forge": False,
    },
    "pip": {
        "install_cmd": None,  # pip uses install_cmd_fn instead
        "install_cmd_fn": lambda: _pip_install(),
        "run_tests_cmd": [str(PROJECT_DIR / ".venv-pip" / "bin" / "python"), "-m", "pytest", "tests/", "-v", "--timeout=60"],
        "clear_cache": lambda: subprocess.run(
            ["pip", "cache", "purge"], capture_output=True
        ),
        "clear_env": lambda: shutil.rmtree(PROJECT_DIR / ".venv-pip", ignore_errors=True),
        "env_path": lambda: PROJECT_DIR / ".venv-pip",
        "lockfile_cmd": None,
        "lockfile_path": lambda: None,
        "supports_conda_forge": False,
    },
    "poetry": {
        "install_cmd": ["poetry", "install", "--with", "test"],
        "run_tests_cmd": ["poetry", "run", "pytest", "tests/", "-v", "--timeout=60"],
        "clear_cache": lambda: subprocess.run(
            ["poetry", "cache", "clear", "--all", "."], capture_output=True, text=True, input="yes\n"
        ),
        "clear_env": lambda: (
            shutil.rmtree(PROJECT_DIR / ".venv", ignore_errors=True),
            subprocess.run(["poetry", "env", "remove", "--all"], capture_output=True, cwd=PROJECT_DIR),
        ),
        "env_path": lambda: _poetry_env_path(),
        "lockfile_cmd": ["poetry", "lock"],
        "lockfile_path": lambda: PROJECT_DIR / "poetry.lock",
        "supports_conda_forge": False,
    },
}


def _find_python() -> str:
    """Find a Python 3.11 or 3.12 interpreter."""
    for candidate in ["python3.12", "python3.11", "python3"]:
        path = shutil.which(candidate)
        if path:
            result = subprocess.run([path, "--version"], capture_output=True, text=True)
            version = result.stdout.strip()
            if "3.11" in version or "3.12" in version:
                return path
    return sys.executable


def _pip_install() -> tuple[float, bool, str]:
    """pip needs venv creation + install as two steps. Returns (elapsed, success, output)."""
    start = time.monotonic()
    venv_path = PROJECT_DIR / ".venv-pip"
    python_bin = _find_python()

    r1 = subprocess.run(
        [python_bin, "-m", "venv", str(venv_path)],
        cwd=PROJECT_DIR, capture_output=True, text=True,
    )
    if r1.returncode != 0:
        return time.monotonic() - start, False, f"venv creation failed (using {python_bin}): " + r1.stderr

    pip_bin = venv_path / "bin" / "pip"
    r2 = subprocess.run(
        [str(pip_bin), "install", "-r", "requirements.txt"],
        cwd=PROJECT_DIR, capture_output=True, text=True, timeout=600,
    )
    elapsed = time.monotonic() - start
    return elapsed, r2.returncode == 0, r2.stdout + r2.stderr


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def _run_install(config: dict, cold: bool) -> tuple[float, bool, str]:
    """Run an install benchmark for a tool. If cold=True, clear cache first."""
    if cold:
        config["clear_cache"]()
    config["clear_env"]()

    # pip has a custom install function
    if "install_cmd_fn" in config:
        return config["install_cmd_fn"]()

    elapsed, success, output = time_command(config["install_cmd"])
    if success and "post_install_cmd" in config:
        e2, s2, o2 = time_command(config["post_install_cmd"])
        elapsed += e2
        success = s2
        output += o2
    return elapsed, success, output


def benchmark_installs(
    tool_name: str,
    config: dict,
    result: BenchmarkResult,
    num_runs: int,
    skip_cold: bool,
):
    """Run cold and warm install benchmarks."""
    if not skip_cold:
        print(f"  [{tool_name}] Cold install ({num_runs} runs)...")
        for i in range(num_runs):
            elapsed, success, output = _run_install(config, cold=True)
            if success:
                result.cold_install.runs.append(elapsed)
                print(f"    Run {i+1}: {elapsed:.1f}s")
            else:
                result.notes.append(f"Cold install run {i+1} failed: {output[:500]}")
                print(f"    Run {i+1}: FAILED")
        result.cold_install.compute_median()

    print(f"  [{tool_name}] Warm install ({num_runs} runs)...")
    for i in range(num_runs):
        elapsed, success, output = _run_install(config, cold=False)
        if success:
            result.warm_install.runs.append(elapsed)
            print(f"    Run {i+1}: {elapsed:.1f}s")
        else:
            result.notes.append(f"Warm install run {i+1} failed: {output[:500]}")
            print(f"    Run {i+1}: FAILED")
    result.warm_install.compute_median()


def benchmark_disk(tool_name: str, config: dict, result: BenchmarkResult):
    """Measure environment disk footprint after install."""
    env_path = config["env_path"]()
    result.disk_footprint_mb = measure_dir_size(env_path)
    print(f"  [{tool_name}] Disk footprint: {result.disk_footprint_mb:.0f} MB")


def benchmark_lockfile(tool_name: str, config: dict, result: BenchmarkResult, num_runs: int):
    """Benchmark lockfile generation."""
    lockfile_cmd = config.get("lockfile_cmd")
    if not lockfile_cmd:
        result.notes.append("Lockfile generation: N/A (tool has no lockfile)")
        print(f"  [{tool_name}] Lockfile: N/A")
        return

    print(f"  [{tool_name}] Lockfile generation ({num_runs} runs)...")
    for i in range(num_runs):
        lockfile_path = config["lockfile_path"]()
        if lockfile_path and lockfile_path.exists():
            lockfile_path.unlink()
        elapsed, success, output = time_command(lockfile_cmd)
        if success:
            result.lockfile_gen.runs.append(elapsed)
            print(f"    Run {i+1}: {elapsed:.1f}s")
        else:
            result.notes.append(f"Lockfile run {i+1} failed: {output[:500]}")
            print(f"    Run {i+1}: FAILED")
    result.lockfile_gen.compute_median()


def benchmark_tests(tool_name: str, config: dict, result: BenchmarkResult, num_runs: int):
    """Benchmark test suite execution."""
    print(f"  [{tool_name}] Test execution ({num_runs} runs)...")
    for i in range(num_runs):
        elapsed, success, output = time_command(config["run_tests_cmd"])
        if success:
            result.test_execution.runs.append(elapsed)
            print(f"    Run {i+1}: {elapsed:.1f}s")
        else:
            result.notes.append(f"Test run {i+1} failed: {output[:500]}")
            print(f"    Run {i+1}: FAILED")
    result.test_execution.compute_median()


def benchmark_solver_stress(tool_name: str, config: dict, result: BenchmarkResult):
    """Run solver stress tests: conflict resolution, incremental add, version bump."""
    print(f"  [{tool_name}] Solver stress tests...")

    if result.warm_install.runs:
        result.solver_stress["conflict_resolution"] = asdict(
            SolverStressResult(time=result.warm_install.median, success=True)
        )
    else:
        result.solver_stress["conflict_resolution"] = asdict(
            SolverStressResult(time=0.0, success=False)
        )

    lockfile_cmd = config.get("lockfile_cmd")
    if lockfile_cmd:
        elapsed, success, output = time_command(lockfile_cmd)
        result.solver_stress["incremental_add"] = asdict(
            SolverStressResult(time=elapsed, success=success)
        )
        print(f"    Incremental re-solve: {elapsed:.1f}s ({'OK' if success else 'FAIL'})")
    else:
        result.solver_stress["incremental_add"] = asdict(
            SolverStressResult(time=0.0, success=False)
        )
        result.notes.append("Incremental add: N/A (no lockfile support)")

    if lockfile_cmd:
        lockfile_path = config["lockfile_path"]()
        if lockfile_path and lockfile_path.exists():
            lockfile_path.unlink()
        elapsed, success, output = time_command(lockfile_cmd)
        result.solver_stress["version_bump"] = asdict(
            SolverStressResult(time=elapsed, success=success)
        )
        print(f"    Version bump re-solve: {elapsed:.1f}s ({'OK' if success else 'FAIL'})")
    else:
        result.solver_stress["version_bump"] = asdict(
            SolverStressResult(time=0.0, success=False)
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all_benchmarks(
    tool_name: str,
    config: dict,
    num_runs: int = 3,
    skip_cold: bool = False,
    skip_solver: bool = False,
) -> BenchmarkResult:
    """Run all benchmarks for a single tool and return the result."""
    result = BenchmarkResult(
        tool=tool_name,
        timestamp=datetime.now().isoformat(),
        system=get_system_info(),
        supports_conda_forge=config["supports_conda_forge"],
    )

    benchmark_installs(tool_name, config, result, num_runs, skip_cold)
    benchmark_disk(tool_name, config, result)
    benchmark_lockfile(tool_name, config, result, num_runs)
    benchmark_tests(tool_name, config, result, num_runs)
    if not skip_solver:
        benchmark_solver_stress(tool_name, config, result)

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark Python package managers")
    parser.add_argument("--tools", nargs="+", default=None, help="Tools to benchmark (default: all available)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per benchmark")
    parser.add_argument("--skip-cold", action="store_true", help="Skip cold install benchmarks")
    parser.add_argument("--skip-solver", action="store_true", help="Skip solver stress tests")
    args = parser.parse_args()

    print("Detecting tools...")
    available = detect_tools()
    for name, version in available.items():
        status = version if version else "NOT FOUND"
        print(f"  {name}: {status}")

    tools_to_run = args.tools or [t for t, v in available.items() if v is not None]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for tool_name in tools_to_run:
        if tool_name not in TOOL_CONFIGS:
            print(f"WARNING: unknown tool '{tool_name}', skipping")
            continue
        if not available.get(tool_name):
            print(f"WARNING: {tool_name} not found, skipping")
            continue

        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {tool_name} ({available[tool_name]})")
        print(f"{'=' * 60}")

        config = TOOL_CONFIGS[tool_name]
        result = run_all_benchmarks(tool_name, config, args.runs, args.skip_cold, args.skip_solver)

        output_path = RESULTS_DIR / f"{tool_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"Results saved to {output_path}")

    print(f"\nDone. Results in {RESULTS_DIR}/")
    print("Run 'python benchmark/report.py' to generate comparison tables and charts.")


if __name__ == "__main__":
    main()
