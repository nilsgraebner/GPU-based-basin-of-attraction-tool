# Basin of Attraction Analysis Tool

This repository contains a Julia/CUDA backend and a Python/Tkinter GUI for basin-of-attraction analysis of nonlinear dynamical systems. The tool was developed to compute basin maps, parameter sweeps, and point-wise trajectory inspections with an extrema-based classification of the post-transient response.

The default configuration contains a Duffing-type electromechanical oscillator model, but the application also supports user-defined ODE systems with 2 to 8 states when the equations can be expressed in the restricted GPU-safe expression language.

## What The App Does

The application evaluates many initial conditions in parallel, integrates each trajectory through a transient phase, and classifies the steady-state response from extrema in a user-selected observable. The result is shown as a basin map in the initial-condition plane. The same workflow can be repeated over a varied model parameter to generate sweep diagrams and drill-down views.

Main analysis modes:

- **Single basin**: Computes one basin-of-attraction map for a selected initial-condition plane.
- **Parameter sweep**: Repeats basin computations over a selected parameter and stores representative class extrema, basin maps, and phase portraits.
- **Point inspection**: Computes one selected trajectory and compares it with a CPU Float64 adaptive reference solution.

The GUI can display and export:

- Basin maps with rectangle zoom and optional recomputation of the selected region.
- Phase portraits of representative trajectories.
- Sweep plots of extrema over a varied parameter.
- Image exports of the current plots.
- Sweep videos showing basin map, phase portrait, and sweep overview along the swept parameter.

## Numerical Workflow

For each initial condition, the backend uses a two-stage workflow:

1. **Stage A, transient integration**  
   The trajectory is integrated until the configured transient time. The final state is used as the starting point for the steady-state evaluation.

2. **Stage B, evaluation window**  
   The trajectory is integrated for a fixed number of evaluation periods. Extrema of the selected observable are extracted from sign changes of a selected zero-crossing state.

3. **Fingerprint classification**  
   The detected extrema are quantized with `fingerprint_tol`, sorted, truncated to `fingerprint_k`, and used as a response fingerprint. Initial conditions with the same fingerprint receive the same basin label.

4. **Visualization and export**  
   Basin labels, representative phase trajectories, class statistics, extrema rows, and summary metadata are written to the run folder.

More details on the implementation are documented in [docs/ALGORITHMS.md](docs/ALGORITHMS.md).

## Solver Modes

The GUI exposes solver setups as curated combinations rather than independent low-level options.

- **GPU fast - Float32 fixed Tsit5**  
  Fixed-step GPU integration using `DiffEqGPU` and `OrdinaryDiffEq`.

- **GPU memory saver - Float32 streaming extrema**  
  Keeps Stage A unchanged, then processes Stage B in chunks and stores only extrema instead of full trajectories.

- **GPU RK4 extrema Stage B - Float32**  
  Uses GPU Tsit5 for Stage A and a fused CUDA RK4 kernel for Stage B extrema detection.

- **GPU RK4 extrema Stage A+B - Float32**  
  Uses custom CUDA RK4 kernels for both transient integration and extrema detection for the built-in Duffing model.

- **GPU RK4 extrema Stage A+B Custom - Float32**  
  Generates specialized CUDA RK4 kernels from validated custom ODE expressions.

- **CPU fixed/adaptive modes**  
  CPU fallbacks and reference modes based on `OrdinaryDiffEq`, including Tsit5, Vern9, Rosenbrock23, and Rodas5P.

## Model Input

The model is entered as one right-hand side expression per state variable. Expressions may use:

- State names and parameter names.
- `t` and, in phase mode, `phase`.
- Numeric constants, `pi`, arithmetic operators, and parentheses.
- `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, `min`, `max`, and `ifelse`.

The expression validator rejects arrays, indexing, assignments, arbitrary Julia code, random numbers, and file access. This keeps custom models compatible with code generation and GPU execution.

In custom RK4 phase mode, equations can remain in the familiar form `cos(w*t)` when the period expression is `2*pi/w`; internally, the product `w*t` is replaced by the wrapped forcing phase.

## Repository Contents

- `run_gui.py`: Python GUI and plotting/export code.
- `basin_backend.jl`: Julia backend for integration, extrema extraction, classification, sweeps, and result I/O.
- `rk4_extrema_cuda.jl`: Custom CUDA kernels and kernel code generation for RK4 extrema detection.
- `default_config.toml`: Default Duffing/harvester configuration.
- `Project.toml`: Julia package environment.
- `requirements.txt`: Optional Python packages for image/video export.
- `examples/`: Small backend example configurations.
- `docs/ALGORITHMS.md`: Algorithm and implementation notes.
- `LICENSE`: MIT license.

Generated data are written to `runs/` and temporary GUI configurations to `work/`. Both folders are ignored by Git.

## Installation

Install Julia and Python first. CUDA-enabled GPU runs require a working NVIDIA driver and a CUDA-compatible Julia setup. CPU modes can be used without CUDA hardware, but the CUDA package is part of the Julia environment because the backend can load GPU kernels when requested.

From the repository root:

```powershell
julia --project=. -e "using Pkg; Pkg.instantiate()"
python -m pip install -r requirements.txt
```

The Python packages are only needed for full export functionality. The GUI itself uses Python's standard `tkinter` module.

## Start The GUI

On Windows:

```powershell
.\start_app.bat
```

Or directly:

```powershell
python run_gui.py
```

The GUI starts a persistent Julia backend process with the local Julia project. The first run can take longer because Julia packages and CUDA may need to initialize or precompile. If the Julia packages have not been installed yet, the GUI automatically runs `Pkg.instantiate()` before starting the backend.

## Command-Line Use

A configuration can also be run directly:

```powershell
julia --project=. basin_backend.jl default_config.toml
```

Small example configs are available in `examples/`, for example:

```powershell
julia --project=. basin_backend.jl examples/smoke_test_config.toml
```

## Citation

If this tool is used in a publication, please cite the associated paper and reference the repository URL in the code availability statement.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
