# Update Notes

## Current Release Update

This update improves result handling, visualization control, GPU solver options, and sweep video export.

- Added **Load Results** to open any previous `summary.toml` result file.
- Added minimum class-fraction filtering for phase portraits. The same filter is used in sweep video exports.
- Changed representative phase-sample selection so that trajectories are selected from the most frequent classes instead of the lowest numeric labels.
- Reintroduced GPU Float64 fixed-step Tsit5 as a solver setup.
- Added generated custom RK4 Stage A+B extrema runs for Float64.
- Allowed the generated custom RK4 Stage A+B path to run the standard Duffing configuration as well as user-defined ODE systems.
- Removed the separate built-in-only RK4 Stage A+B solver from the GUI solver list.
- Added a 5-second intro page to exported sweep videos with model equations, parameters, solver setup, and integration/classification settings.

Example videos:

- Bistable Duffing oscillator sweep: https://youtu.be/6RrQ9ALEsMU
- Bistable nonlinear energy harvester sweep: https://youtu.be/gjpNIxjhMkw
