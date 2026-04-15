import json
from pathlib import Path

from benchmark.report import load_results, generate_comparison_table, generate_charts


FIXTURE_RESULT = {
    "tool": "pixi",
    "timestamp": "2026-01-01T00:00:00",
    "system": {"os": "Darwin", "arch": "arm64", "python": "3.11.0", "cpu": "arm", "os_version": ""},
    "cold_install": {"runs": [10.0, 11.0, 12.0], "median": 11.0},
    "warm_install": {"runs": [3.0, 3.5, 4.0], "median": 3.5},
    "lockfile_gen": {"runs": [2.0, 2.5, 3.0], "median": 2.5},
    "disk_footprint_mb": 2500.0,
    "test_execution": {"runs": [5.0, 5.5, 6.0], "median": 5.5},
    "solver_stress": {
        "conflict_resolution": {"time": 8.0, "success": True},
        "incremental_add": {"time": 2.0, "success": True},
        "version_bump": {"time": 3.0, "success": True},
    },
    "notes": [],
    "supports_conda_forge": True,
}


def _write_fixture(tmp_path: Path, tool_name: str = "pixi", overrides: dict | None = None) -> Path:
    data = {**FIXTURE_RESULT, "tool": tool_name}
    if overrides:
        data.update(overrides)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"{tool_name}_result.json"
    path.write_text(json.dumps(data))
    return path


def test_load_results_single_file(tmp_path):
    _write_fixture(tmp_path)
    results = load_results(tmp_path)
    assert len(results) == 1
    assert results[0]["tool"] == "pixi"


def test_load_results_multiple_files(tmp_path):
    _write_fixture(tmp_path, "pixi")
    _write_fixture(tmp_path, "conda", {"cold_install": {"runs": [40.0], "median": 40.0}})
    results = load_results(tmp_path)
    assert len(results) == 2
    tools = {r["tool"] for r in results}
    assert tools == {"pixi", "conda"}


def test_load_results_empty_dir(tmp_path):
    results = load_results(tmp_path)
    assert results == []


def test_generate_comparison_table(tmp_path):
    _write_fixture(tmp_path, "pixi")
    _write_fixture(tmp_path, "conda", {"cold_install": {"runs": [45.0], "median": 45.0}})
    results = load_results(tmp_path)
    table = generate_comparison_table(results)
    assert "pixi" in table
    assert "conda" in table
    assert "11.0" in table  # pixi cold install median
    assert "45.0" in table  # conda cold install median


def test_generate_charts_creates_files(tmp_path):
    _write_fixture(tmp_path / "data", "pixi")
    _write_fixture(tmp_path / "data", "conda", {"cold_install": {"runs": [45.0], "median": 45.0}})
    results = load_results(tmp_path / "data")
    chart_dir = tmp_path / "charts"
    generate_charts(results, output_dir=str(chart_dir))
    assert (chart_dir / "install_comparison.png").exists()
    assert (chart_dir / "disk_footprint.png").exists()
    assert (chart_dir / "solver_stress.png").exists()
