# Algorithms And Implementation Notes

This document summarizes the numerical workflow implemented by the basin-of-attraction analysis tool. It is intended as a technical companion to the GUI-oriented README.

## Overview

The program analyzes nonlinear ODE systems by integrating a grid of initial conditions and classifying the post-transient response. The classification is based on extrema of a selected observable rather than on the full saved trajectory. This reduces the amount of data that must be retained and gives a compact representation of periodic, multi-periodic, and many steady-state response classes.

The main workflow is:

1. Generate initial conditions on a two-dimensional plane.
2. Integrate every initial condition through a transient interval.
3. Integrate an evaluation window after the transient.
4. Extract extrema of a selected observable from zero crossings of another state.
5. Convert extrema to fingerprints and assign basin labels.
6. Save basin labels, class statistics, representative trajectories, and sweep data.

## Initial-Condition Grid

The basin plane is defined by two selected state variables. All other state components are fixed at the configured default initial values.

The grid can be specified directly by `nx` and `ny`, or generated from a target number of points while preserving the aspect ratio of the selected coordinate ranges. Each grid point defines one trajectory.

The final label matrix is reshaped back to the grid layout and written as `labels.csv` for visualization in the GUI.

## Time Settings

The backend supports two time modes:

- **Periodic mode**: The period is evaluated from `period_expression`, for example `2*pi/w`. Transient and evaluation durations are specified in periods. The fixed step size is derived from `samples_per_period`.
- **Direct-time mode**: Transient duration, evaluation duration, step size, and save interval are specified directly.

In periodic mode, `dt` and `save_dt` are equal. This is required by the custom RK4 extrema kernels because extrema are detected on the same discrete time grid used by the integration.

## Stage A And Stage B

Most runs are split into two stages:

**Stage A** integrates the transient. Only the final state after the transient is needed for basin classification.

**Stage B** integrates the evaluation window. In the standard path, the evaluation trajectory is saved and extrema are extracted afterwards. In streaming and RK4-extrema paths, extrema are collected during Stage B so that full trajectories do not need to be stored for all initial conditions.

For result plots, the program stores or recomputes a small set of representative phase trajectories. This keeps the basin computation memory-efficient while still allowing visual inspection.

## Extrema Detection

Extrema are detected from sign changes in a configured zero-crossing state. For the default Duffing-type model, the observable is typically displacement `x` and the zero-crossing state is velocity `v`.

For each pair of consecutive saved states, the backend checks whether the zero-crossing state changes sign. If a crossing is found, the observable value at the crossing is linearly interpolated between the two samples. These values are stored as extrema of the steady-state response.

The parameters controlling this step are:

- `observable_state`: state whose extrema define the response.
- `zero_cross_state`: state whose sign changes indicate extrema.
- `extrema_eps`: small threshold below which zero-crossing values are treated as zero.
- `max_extrema`: maximum number of extrema retained per trajectory.

## Fingerprint Classification

Each trajectory is represented by a fingerprint derived from its extrema:

1. Extrema are rounded by `fingerprint_tol`.
2. The rounded extrema values are sorted.
3. Only the first `fingerprint_k` values are retained.
4. Equal fingerprints are assigned the same class label.

This intentionally ignores the order in which extrema occur and focuses on the set of characteristic response amplitudes. It is useful for basin maps where the goal is to separate distinct steady-state response families.

The class count and class fractions are written to `class_stats.csv`. The basin image uses the label matrix, while sweep plots can use either label colors or feature-based colors derived from extrema statistics.

## Solver Backends

### Standard Fixed-Step Solvers

CPU and GPU fixed-step paths use `OrdinaryDiffEq` solvers. GPU ensemble computations are executed through `DiffEqGPU`. These paths are the most general and work with the validated custom ODE functions generated from the GUI expressions.

### Adaptive CPU Solvers

Adaptive CPU solvers are available for reference and point inspection. They are useful for checking numerical sensitivity, but they are slower for large basin grids.

### Streaming Extrema

The streaming-extrema mode avoids retaining full Stage B trajectories. It processes the evaluation window in chunks, updates extrema from consecutive states, and stores only the extrema values needed for classification. If the configured chunk covers the full evaluation window, the backend can use the faster standard full-window path.

### CUDA RK4 Extrema Kernels

The experimental RK4 extrema modes use custom CUDA kernels to integrate fixed-step RK4 and detect extrema directly on the GPU.

There are two main variants:

- **Stage B RK4 extrema**: Stage A is computed by the standard GPU solver. Stage B is computed by a CUDA RK4 kernel that returns extrema.
- **Stage A+B RK4 extrema**: Both transient integration and extrema detection are performed by CUDA RK4 kernels.

For the built-in Duffing model, hand-written kernels are still available internally for compatibility with older configurations. The GUI exposes the generated custom Stage A+B path instead, so the same solver setup can be used for both the standard model and user-defined systems.

For generated Stage A+B kernels, the backend validates the expressions and generates a specialized CUDA RHS and RK4 kernel for the selected state and parameter names. This currently supports 2 to 8 states and Float32 or Float64 GPU runs.

## Phase Mode For Periodic Forcing

Long integrations in Float32 can lose resolution when the forcing is evaluated as `cos(w*t)` with very large `t`. For periodic forcing, the RK4 phase mode advances a wrapped phase in `[0, 2*pi)` and evaluates the forcing from that phase.

For custom RK4 phase mode, the user may keep the equation as `cos(w*t)` when `period_expression = 2*pi/w`. The backend rewrites the product `w*t` to the wrapped internal `phase` before generating the runtime ODE function and CUDA kernel. Periodic custom Stage A+B RK4 runs select this wrapped phase form automatically when every physical-time occurrence can be rewritten safely; otherwise the explicitly configured time mode is kept.

Representative phase trajectories are selected from the most frequent classes, not from the lowest numeric labels. The GUI can then filter those saved trajectories by minimum class fraction without recomputing the ODE integrations; the same filter is used during sweep video export.

## Parameter Sweeps

Parameter sweeps repeat the operating-point workflow for a sequence of values of one selected parameter. For each value, the backend can save:

- Basin labels.
- Class statistics.
- Representative phase trajectories.
- Extrema rows for the sweep overview.
- Per-value timing information.

The sweep plot does not need to show every extremum from every trajectory. Instead, classes can be filtered by minimum class fraction, maximum number of classes per sweep value, and maximum point budget. These controls redraw the sweep visualization locally without rerunning the ODE integrations.

## Point Inspection

Point inspection computes one selected trajectory with the currently selected method and compares it with a CPU Float64 adaptive Vern9 reference trajectory. This is intended for diagnosing solver sensitivity, transient settling, and differences between fixed-step GPU paths and high-accuracy CPU reference integration.

## Output Files

Each run creates a folder under `runs/`. Typical outputs include:

- `summary.toml`: metadata and paths to result files.
- `config_snapshot.toml`: full configuration used for the run.
- `labels.csv`: basin label matrix.
- `class_stats.csv`: class counts, fractions, and representative extrema statistics.
- `phase_samples.csv` or binary phase-sample files: representative phase trajectories.
- Sweep-specific CSV files for extrema, timings, class statistics, and drill-down manifests.

Temporary GUI configurations are written to `work/`. Both `runs/` and `work/` are intentionally ignored by Git.
