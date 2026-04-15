# pixi-benchmark

Benchmark comparing Python package manager performance for ML/AI workloads.

Compares **pixi**, **uv**, **conda**, **mamba**, **pip**, and **poetry** on:

- Cold and warm install time
- Lockfile generation speed
- Environment disk footprint
- Test suite execution time
- Solver stress (conflict resolution, incremental adds, version bumps)

## The Project

`mlbench` is a toy-but-functional ML pipeline built on CIFAR-10:

- **Data**: HuggingFace datasets + albumentations transforms
- **Model**: timm ResNet-18 in a PyTorch Lightning module
- **Train**: Lightning Trainer (1-2 epochs on a tiny subset)
- **Evaluate**: torchmetrics accuracy/F1, confusion matrix, ONNX export
- **Serve**: Gradio inference demo

The dependency list deliberately mixes conda-forge and PyPI packages (~25 total)
to stress the real-world pain of mixed-channel resolution.

## Tool Compatibility

| Tool | Conda-forge | PyPI | Mixed channels |
|------|-------------|------|----------------|
| pixi | Yes | Yes | Yes (native) |
| conda | Yes | Yes (via pip) | Partial |
| mamba | Yes | Yes (via pip) | Partial |
| uv | No | Yes | PyPI-only |
| pip | No | Yes | PyPI-only |
| poetry | No | Yes | PyPI-only |

## Running Benchmarks

Install at least one of the tools, then:

```bash
# All available tools
python benchmark/runner.py

# Specific tools
python benchmark/runner.py --tools pixi uv conda

# More runs for better statistics
python benchmark/runner.py --runs 5

# Skip slow cold-install benchmarks
python benchmark/runner.py --skip-cold

# Skip solver stress tests
python benchmark/runner.py --skip-solver
```

Results are saved as JSON in `benchmark/results/`.

## Generating Reports

```bash
python benchmark/report.py
```

Produces a markdown comparison table (stdout) and PNG charts in `benchmark/charts/`.

## Running the ML Pipeline

With pixi:
```bash
pixi run test          # run tests
pixi run train         # train the model
```

With other tools, activate the environment and run directly:
```bash
pytest tests/ -v
python -m mlbench.train
```
