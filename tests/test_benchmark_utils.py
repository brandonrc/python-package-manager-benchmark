import json
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import benchmark.runner as runner
from benchmark.runner import (
    get_system_info,
    measure_dir_size,
    time_command,
    detect_tools,
    BenchmarkResult,
    TimingResult,
    SolverStressResult,
    TOOL_CONFIGS,
    _parse_conda_version,
    _version_at_least,
    _conda_pypi_preflight,
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


# --- conda-pypi support -----------------------------------------------------

def test_parse_conda_version():
    assert _parse_conda_version("conda 26.5.2") == (26, 5, 2)
    assert _parse_conda_version("conda 26.1.1") == (26, 1, 1)
    assert _parse_conda_version("26.5") == (26, 5, 0)


def test_version_at_least():
    assert _version_at_least((26, 5, 2), (26, 5)) is True
    assert _version_at_least((26, 5, 0), (26, 5)) is True
    assert _version_at_least((27, 0, 0), (26, 5)) is True
    assert _version_at_least((26, 4, 9), (26, 5)) is False
    assert _version_at_least((26, 1, 1), (26, 5)) is False


def _fake_run_factory(version_out, base_list_out):
    """Build a fake subprocess.run that answers conda --version and conda list."""
    def fake_run(cmd, *args, **kwargs):
        if cmd[:2] == ["conda", "--version"]:
            return SimpleNamespace(returncode=0, stdout=version_out, stderr="")
        if cmd[:3] == ["conda", "list", "-n"]:
            return SimpleNamespace(returncode=0, stdout=base_list_out, stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    return fake_run


def test_preflight_fails_on_old_conda(monkeypatch):
    monkeypatch.setattr(
        runner.subprocess, "run",
        _fake_run_factory("conda 26.1.1\n", "conda-pypi 0.9.0\nconda-rattler-solver 0.1.0\n"),
    )
    ok, reason = _conda_pypi_preflight()
    assert ok is False
    assert "26.1.1" in reason and "26.5" in reason


def test_preflight_fails_when_rattler_missing(monkeypatch):
    monkeypatch.setattr(
        runner.subprocess, "run",
        _fake_run_factory("conda 26.5.2\n", "conda-pypi 0.9.0\nlibmamba 2.5.0\n"),
    )
    ok, reason = _conda_pypi_preflight()
    assert ok is False
    assert "rattler" in reason.lower()


def test_preflight_passes_when_ready(monkeypatch):
    monkeypatch.setattr(
        runner.subprocess, "run",
        _fake_run_factory("conda 26.5.2\n", "conda-pypi 0.9.0\nconda-rattler-solver 0.1.0\n"),
    )
    ok, reason = _conda_pypi_preflight()
    assert ok is True


def test_conda_pypi_config_present_and_isolated():
    assert "conda-pypi" in TOOL_CONFIGS
    cfg = TOOL_CONFIGS["conda-pypi"]
    # Uses the rattler solver as a CLI flag (no global condarc mutation)
    assert "--solver" in cfg["install_cmd"]
    assert "rattler" in cfg["install_cmd"]
    # Uses its own environment file and env name, not the conda baseline
    assert "environment-pypi.yml" in cfg["install_cmd"]
    assert "mlbench-conda-pypi" in cfg["install_cmd"]
    # No global-config-mutating pre-steps
    assert "pre_install_cmds" not in cfg
    # Has a preflight gate
    assert callable(cfg.get("preflight"))


def test_conda_baseline_no_longer_mutates_global_config():
    # Regression: the merged PR added global config mutation to the conda entry.
    cfg = TOOL_CONFIGS["conda"]
    assert "pre_install_cmds" not in cfg
    assert cfg["install_cmd"][:3] == ["conda", "env", "create"]


def test_detect_tools_includes_conda_pypi():
    tools = detect_tools()
    assert "conda-pypi" in tools
