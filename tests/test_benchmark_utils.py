import json
import os
from dataclasses import asdict
from pathlib import Path

from benchmark.runner import (
    get_system_info,
    measure_dir_size,
    time_command,
    detect_tools,
    BenchmarkResult,
    TimingResult,
    SolverStressResult,
    TOOL_CONFIGS,
)


def test_get_system_info_has_required_keys():
    info = get_system_info()
    assert "os" in info
    assert "arch" in info
    assert "python" in info
    assert "cpu" in info


def test_measure_dir_size_with_known_file(tmp_path):
    test_file = tmp_path / "test.bin"
    test_file.write_bytes(b"x" * 1024)
    size = measure_dir_size(tmp_path)
    assert 0.0009 < size < 0.002  # ~1KB in MB


def test_measure_dir_size_empty_dir(tmp_path):
    size = measure_dir_size(tmp_path)
    assert size == 0.0


def test_measure_dir_size_nonexistent():
    size = measure_dir_size(Path("/nonexistent/path"))
    assert size == 0.0


def test_time_command_success():
    elapsed, success, output = time_command(["echo", "hello"])
    assert success is True
    assert elapsed < 5.0
    assert "hello" in output


def test_time_command_failure():
    elapsed, success, output = time_command(["false"])
    assert success is False


def test_detect_tools_returns_dict():
    tools = detect_tools()
    assert isinstance(tools, dict)
    for name in ["pixi", "uv", "conda", "mamba", "pip", "poetry"]:
        assert name in tools


def test_detect_tools_finds_pip():
    tools = detect_tools()
    assert tools["pip"] is not None  # pip should always be available


def test_timing_result_compute_median():
    tr = TimingResult(runs=[1.0, 3.0, 2.0])
    tr.compute_median()
    assert tr.median == 2.0


def test_timing_result_empty():
    tr = TimingResult()
    tr.compute_median()
    assert tr.median == 0.0


def test_benchmark_result_json_roundtrip():
    result = BenchmarkResult(
        tool="test_tool",
        timestamp="2026-01-01T00:00:00",
        system={"os": "TestOS"},
        cold_install=TimingResult(runs=[1.0, 2.0, 3.0], median=2.0),
        warm_install=TimingResult(runs=[0.5, 0.6, 0.7], median=0.6),
        lockfile_gen=TimingResult(runs=[0.3], median=0.3),
        disk_footprint_mb=1234.5,
        test_execution=TimingResult(runs=[5.0], median=5.0),
        solver_stress={
            "conflict_resolution": asdict(SolverStressResult(time=10.0, success=True)),
        },
    )
    data = asdict(result)
    json_str = json.dumps(data, indent=2)
    parsed = json.loads(json_str)
    assert parsed["tool"] == "test_tool"
    assert parsed["cold_install"]["median"] == 2.0
    assert parsed["solver_stress"]["conflict_resolution"]["success"] is True


def test_tool_configs_has_all_tools():
    for name in ["pixi", "uv", "conda", "mamba", "pip", "poetry"]:
        assert name in TOOL_CONFIGS
        config = TOOL_CONFIGS[name]
        assert "install_cmd" in config
        assert "run_tests_cmd" in config
        assert "clear_env" in config
        assert "env_path" in config
        assert "supports_conda_forge" in config
