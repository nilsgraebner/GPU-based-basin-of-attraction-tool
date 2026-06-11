from __future__ import annotations

import ast
import csv
import colorsys
import io
import math
import os
import queue
import random
import re
import struct
import subprocess
import threading
import time
import traceback
import tomllib
import tkinter as tk
from fractions import Fraction
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - optional export dependency
    Image = ImageDraw = ImageFont = None


BASE_DIR = Path(__file__).resolve().parent
BACKEND_PATH = BASE_DIR / "basin_backend.jl"
DEFAULT_CONFIG_PATH = BASE_DIR / "default_config.toml"
WORK_DIR = BASE_DIR / "work"
WORK_DIR.mkdir(exist_ok=True)
PHASE_BINARY_MAGIC = b"BASPH01\x00"
MAX_SWEEP_POINTS_IN_MEMORY = 1_000_000
SWEEP_MIN_FRACTION_DEFAULT = "0.0002"
SWEEP_MAX_CLASSES_DEFAULT = "50"
SWEEP_MAX_POINTS_DEFAULT = "1000000"
SWEEP_COLOR_MODE_DEFAULT = "Feature"
SWEEP_POINT_SIZE_DEFAULT = 0.9
PHASE_MIN_FRACTION_DEFAULT = "0.0"
EXPORT_IMAGE_SIZE = (1000, 720)
VIDEO_FRAME_SIZE = (1920, 1080)
VIDEO_HEADER_HEIGHT = 360
VIDEO_MARGIN = 36
VIDEO_PANEL_GAP = 24
VIDEO_FPS = 24
VIDEO_INTRO_SECONDS = 5
VIDEO_INTRO_FRAMES = VIDEO_FPS * VIDEO_INTRO_SECONDS
JULIA_LOG_FRAME_TRANSLATION = str.maketrans(
    {
        "\u2500": "-",
        "\u2502": "|",
        "\u250c": "+",
        "\u2510": "+",
        "\u2514": "+",
        "\u2518": "+",
    }
)
GPU_SAFE_FUNCTIONS = {
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "exp",
    "log",
    "log10",
    "sqrt",
    "abs",
    "min",
    "max",
    "ifelse",
}
RESERVED_MODEL_NAMES = GPU_SAFE_FUNCTIONS | {"t", "phase", "pi", "π", "T", "u", "p"}

SOLVER_SETUPS = [
    {
        "label": "GPU fast - Float32 fixed Tsit5",
        "device": "gpu",
        "precision": "Float32",
        "solver_mode": "fixed",
        "solver": "Tsit5",
        "description": "Fastest standard GPU path. Uses fixed-step Tsit5 and stores the evaluation window for classification and phase plotting.",
    },
    {
        "label": "GPU accurate - Float64 fixed Tsit5",
        "device": "gpu",
        "precision": "Float64",
        "solver_mode": "fixed",
        "solver": "Tsit5",
        "description": "Standard GPU fixed-step Tsit5 path with Float64. Slower and more memory intensive than Float32, but useful for precision checks when the GPU supports Float64.",
    },
    {
        "label": "GPU memory saver - Float32 streaming extrema",
        "device": "gpu",
        "precision": "Float32",
        "solver_mode": "streaming_extrema",
        "solver": "Tsit5",
        "description": "GPU fixed-step path that keeps only extrema when Stage B is chunked. If one chunk covers the full evaluation window, it automatically uses the fast full-window path.",
    },
    {
        "label": "GPU RK4 extrema Stage B - Float32",
        "device": "gpu",
        "precision": "Float32",
        "solver_mode": "rk4_extrema",
        "solver": "Tsit5",
        "description": "Experimental fused CUDA path: Stage A uses GPU Tsit5, Stage B uses a custom fixed-step RK4 kernel that detects extrema directly on the GPU. The built-in Duffing model keeps its phase-wrapped optimized path; custom GPU-safe ODEs with 2 to 8 states use the generic RK4 kernel.",
    },
    {
        "label": "GPU RK4 extrema Stage A+B Custom - Float32",
        "device": "gpu",
        "precision": "Float32",
        "solver_mode": "rk4_full_extrema_custom",
        "solver": "Tsit5",
        "description": "Experimental all-RK4 CUDA path for GPU-safe ODEs with 2 to 8 states, including the standard Duffing model. Generates a specialized CUDA RHS and RK4 kernel from the model equations, then detects extrema directly on the GPU.",
    },
    {
        "label": "GPU RK4 extrema Stage A+B Custom - Float64",
        "device": "gpu",
        "precision": "Float64",
        "solver_mode": "rk4_full_extrema_custom",
        "solver": "Tsit5",
        "description": "Same generated all-RK4 CUDA A+B path as the custom Float32 setup, but with Float64 state integration on GPUs that support double precision.",
    },
    {
        "label": "CPU fast - Float32 fixed Tsit5",
        "device": "cpu",
        "precision": "Float32",
        "solver_mode": "fixed",
        "solver": "Tsit5",
        "description": "Threaded CPU fixed-step Tsit5 with Float32. Useful as a CPU fallback that stays close to the GPU precision.",
    },
    {
        "label": "CPU accurate - Float64 fixed Tsit5",
        "device": "cpu",
        "precision": "Float64",
        "solver_mode": "fixed",
        "solver": "Tsit5",
        "description": "Threaded CPU fixed-step Tsit5 with Float64. Good for reproducible fixed-step checks with higher precision.",
    },
    {
        "label": "CPU adaptive - Float64 Tsit5",
        "device": "cpu",
        "precision": "Float64",
        "solver_mode": "adaptive",
        "solver": "Tsit5",
        "description": "General-purpose adaptive CPU solver. Slower than fixed-step GPU, but useful when error tolerances matter.",
    },
    {
        "label": "CPU high accuracy - Float64 Vern9",
        "device": "cpu",
        "precision": "Float64",
        "solver_mode": "adaptive",
        "solver": "Vern9",
        "description": "Higher-order adaptive CPU solver for smooth non-stiff systems when accuracy is more important than speed.",
    },
    {
        "label": "CPU stiff - Float64 Rosenbrock23",
        "device": "cpu",
        "precision": "Float64",
        "solver_mode": "adaptive",
        "solver": "Rosenbrock23",
        "description": "Adaptive CPU solver for mildly stiff systems or quick stiffness checks.",
    },
    {
        "label": "CPU stiff accurate - Float64 Rodas5P",
        "device": "cpu",
        "precision": "Float64",
        "solver_mode": "adaptive",
        "solver": "Rodas5P",
        "description": "More accurate stiff CPU solver. Usually slower, but a good cross-check for difficult parameter regions.",
    },
]
SOLVER_SETUP_LABELS = [setup["label"] for setup in SOLVER_SETUPS]
SOLVER_SETUP_BY_LABEL = {setup["label"]: setup for setup in SOLVER_SETUPS}
SOLVER_SETUP_BY_FIELDS = {
    (setup["device"], setup["precision"], setup["solver_mode"], setup["solver"]): setup
    for setup in SOLVER_SETUPS
}


def parse_csv_strings(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_csv_numbers(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        token = part.strip()
        if token:
            values.append(float(token))
    return values


def parse_multiline(raw: str) -> list[str]:
    return [line.strip() for line in raw.splitlines() if line.strip()]


def normalize_expr_string(text: str) -> str:
    return re.sub(r"\s+", "", text.strip())


def matches_default_duffing_config(state_names: list[str], parameter_names: list[str], equations: list[str]) -> bool:
    return (
        state_names == ["x", "v", "U"]
        and parameter_names == ["w", "F0", "d", "r1", "r2", "r3"]
        and [normalize_expr_string(equation) for equation in equations]
        == [
            "v",
            "-d*v-r1*U+0.5*x-0.5*x^3+F0*cos(w*t)",
            "r2*v-r3*U",
        ]
    )


def toml_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{toml_escape(value)}"'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        inner = ", ".join(toml_value(item) for item in value)
        return f"[{inner}]"
    raise TypeError(f"Unsupported TOML value type: {type(value)!r}")


def format_numeric_value(value: float) -> str:
    return f"{value:.12g}"


def dump_toml(config: dict, path: Path) -> None:
    ordered_sections = [
        "analysis",
        "probe",
        "sweep",
        "model",
        "initial_condition_plane",
        "integration",
        "classification",
        "output",
    ]
    lines: list[str] = []
    for section in ordered_sections:
        if section not in config:
            continue
        values = config[section]
        lines.append(f"[{section}]")
        for key, value in values.items():
            if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
                if key == "equations":
                    lines.append(f"{key} = [")
                    for item in value:
                        lines.append(f'  "{toml_escape(item)}",')
                    lines.append("]")
                else:
                    lines.append(f"{key} = {toml_value(value)}")
            else:
                lines.append(f"{key} = {toml_value(value)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


PARAM_HELP = {
    "analysis_mode": "Selects what the app computes. Single Basin computes one basin map and phase plot. Parameter Sweep varies one model parameter and stores drill-down details. Inspection opens an interactive initial-condition plane without computing until you click a point.",
    "sweep_parameter": "Name of the model parameter to vary during a parameter sweep. It must be one of the parameter names listed in the model tab.",
    "sweep_min_value": "First value of the swept parameter.",
    "sweep_max_value": "Last value of the swept parameter.",
    "sweep_num_values": "Number of parameter values sampled between the sweep minimum and maximum.",
    "state_names": "Comma-separated state variable names. The order defines the state vector u used by the Julia ODE.",
    "parameter_names": "Comma-separated parameter names available inside the equation expressions.",
    "parameter_values": "Numeric values for the parameters, in exactly the same order as the parameter names.",
    "state_defaults": "Default initial state. Components not varied on the basin plane keep these values.",
    "equations": "Right-hand sides of the ODE. Enter one scalar expression per state, for example dx/dt = v and dv/dt = -d*v - x^3 + F0*cos(w*t). GPU-safe expressions may use state names, parameter names, t, phase, numeric constants, pi/π, + - * / ^, parentheses, comparisons inside ifelse, and functions such as sin, cos, exp, log, sqrt, abs, min, max. Arrays, indexing, assignments, custom functions, and arbitrary Julia code are not allowed.",
    "x_state": "State variable used for the horizontal axis of the initial-condition plane.",
    "y_state": "State variable used for the vertical axis of the initial-condition plane.",
    "x_min": "Lower bound of the horizontal initial-condition range.",
    "x_max": "Upper bound of the horizontal initial-condition range.",
    "y_min": "Lower bound of the vertical initial-condition range.",
    "y_max": "Upper bound of the vertical initial-condition range.",
    "grid_mode": "auto derives nx and ny from the target sample count and aspect ratio; manual uses nx and ny directly.",
    "n_target": "Approximate total number of initial conditions when grid mode is auto.",
    "nx": "Number of grid points along the horizontal initial-condition axis in manual mode.",
    "ny": "Number of grid points along the vertical initial-condition axis in manual mode.",
    "solver_setup": "Selects a valid backend solver combination. The Julia config still stores device, precision, solver_mode, and solver separately for compatibility.",
    "device": "Execution target. GPU uses DiffEqGPU fixed-step integration. Custom equations are accepted when they pass the GPU-safe expression validator; otherwise the run stops with a clear error instead of falling back to CPU.",
    "precision": "Floating-point precision used by the Julia backend. Float32 is faster and matches the GPU workflow; Float64 is more accurate on CPU.",
    "solver_mode": "fixed uses a constant time step; streaming_extrema uses fixed-step integration but collects only extrema during the evaluation window; rk4_extrema uses a fused CUDA RK4 Stage-B kernel; rk4_full_extrema is the optimized built-in Duffing A+B kernel; rk4_full_extrema_custom generates a specialized A+B CUDA kernel for custom GPU-safe ODEs with 2 to 8 states; adaptive lets the CPU solver choose steps from error tolerances.",
    "solver": "Time integrator used by the backend. GPU currently supports Tsit5 only; CPU supports the listed solvers.",
    "period_mode": "periodic computes times from a forcing period; direct_time uses the explicit time and step fields.",
    "custom_time_argument": "For custom RK4 A+B only: time passes physical t to the generated RHS; phase passes a wrapped phase variable from 0 to 2*pi. In phase mode you may keep expressions such as F0*cos(w*t) when Period Expression is 2*pi/w; internally w*t is replaced by phase. Periodic custom A+B runs use phase automatically when the forcing can be rewritten safely.",
    "period_expression": "Julia expression for one forcing period, evaluated with the current parameters. The default 2*pi/w follows the harmonic forcing.",
    "transient_periods": "Number of forcing periods integrated and discarded before classification in periodic mode.",
    "evaluation_periods": "Number of forcing periods recorded after the transient for extrema classification and phase plotting.",
    "samples_per_period": "Number of fixed integration steps per forcing period in periodic fixed-step mode.",
    "t_transient": "Transient duration used in direct_time mode.",
    "t_evaluation": "Evaluation-window duration used in direct_time mode.",
    "dt": "Fixed solver step size used in direct_time mode. In periodic mode it is derived from period/samples_per_period.",
    "save_dt": "Sampling interval for saved trajectory points in direct_time mode. In periodic mode it matches dt.",
    "streaming_chunk_periods": "Number of forcing periods processed per streaming-extrema chunk. Larger chunks reduce repeated solver launches; if the chunk covers the full evaluation window, GPU runs use the standard fast full-window path. Smaller chunks reduce peak memory during Stage B.",
    "abstol": "Absolute error tolerance for adaptive CPU integration.",
    "reltol": "Relative error tolerance for adaptive CPU integration.",
    "observable_state": "State whose extrema are used to identify the post-transient response class.",
    "zero_cross_state": "State whose sign changes define extrema of the observable state. For Duffing this is usually velocity v.",
    "fingerprint_tol": "Quantization tolerance for extrema values before comparing response signatures. Larger values merge more classes.",
    "fingerprint_k": "Maximum number of sorted extrema values used in the response fingerprint.",
    "extrema_eps": "Small zero-crossing threshold. Values of the zero-crossing state below this magnitude are treated as zero.",
    "max_extrema": "Maximum number of extrema stored per trajectory during classification.",
    "run_name": "Prefix used for the output folder name in runs/.",
    "write_labels_csv": "Write the basin label matrix as labels.csv. Keep enabled for GUI basin rendering.",
    "write_basin_image": "Write an additional basin_map.ppm image file beside the CSV results.",
    "write_sweep_details": "Store one basin label map and representative phase trajectories for every sweep value. This enables instant drill-down by clicking in the sweep plot.",
}


class Tooltip:
    def __init__(self, widget: tk.Widget, text: str, *, delay_ms: int = 450, wraplength: int = 380) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.wraplength = wraplength
        self.after_id: str | None = None
        self.window: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress-1>", self._toggle, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self.after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self) -> None:
        if self.after_id is not None:
            self.widget.after_cancel(self.after_id)
            self.after_id = None

    def _show(self) -> None:
        if self.window is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(
            self.window,
            text=self.text,
            justify="left",
            wraplength=self.wraplength,
            padding=(10, 7),
            relief="solid",
            borderwidth=1,
        )
        label.pack()

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self.window is not None:
            self.window.destroy()
            self.window = None

    def _toggle(self, event=None) -> None:
        if self.window is None:
            self._cancel()
            self._show()
        else:
            self._hide()
        if event is not None:
            return "break"


class BasinGuiApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Basin of Attraction GUI")
        self.geometry("1500x920")
        self.minsize(1200, 760)

        self.vars: dict[str, tk.StringVar] = {}
        self.text_widgets: dict[str, ScrolledText] = {}
        self.equation_frame: ttk.Frame | None = None
        self.equation_vars: list[tk.StringVar] = []
        self.equation_status_label: ttk.Label | None = None
        self.solver_setup_description_label: ttk.Label | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.backend_process: subprocess.Popen[str] | None = None
        self.julia_environment_ready = False
        self._cuda_version_notice_suppressed = False
        self._cuda_version_notice_filter_remaining = 0
        self.current_summary_path: Path | None = None
        self.current_result_dir: Path | None = None
        self.current_summary_data: dict | None = None
        self.current_result_mode = ""
        self.current_image: tk.PhotoImage | None = None
        self.basin_rows: list[list[int]] = []
        self.basin_bounds: tuple[float, float, float, float] | None = None
        self.current_label_feature_colors: dict[int, str] = {}
        self.result_plane_bounds: tuple[float, float, float, float] | None = None
        self.basin_zoom_history: list[dict[str, str]] = []
        self.phase_series: list[dict] = []
        self.phase_bounds: tuple[float, float, float, float] | None = None
        self.class_fractions: dict[int, float] = {}
        self.sweep_points: list[dict] = []
        self.sweep_points_total = 0
        self.sweep_bounds: tuple[float, float, float, float] | None = None
        self.sweep_feature_colors_by_value: dict[float, dict[int, str]] = {}
        self.sweep_class_fractions_by_value: dict[float, dict[int, float]] = {}
        self.sweep_details: list[dict] = []
        self.selected_sweep_value: float | None = None
        self.sweep_min_fraction_var = tk.StringVar(value=SWEEP_MIN_FRACTION_DEFAULT)
        self.sweep_max_classes_var = tk.StringVar(value=SWEEP_MAX_CLASSES_DEFAULT)
        self.sweep_max_points_var = tk.StringVar(value=SWEEP_MAX_POINTS_DEFAULT)
        self.sweep_color_mode_var = tk.StringVar(value=SWEEP_COLOR_MODE_DEFAULT)
        self.sweep_point_size_var = tk.DoubleVar(value=SWEEP_POINT_SIZE_DEFAULT)
        self.phase_min_fraction_var = tk.StringVar(value=PHASE_MIN_FRACTION_DEFAULT)
        self.phase_filter_status_label: ttk.Label | None = None
        self.sweep_filter_status_label: ttk.Label | None = None
        self.sweep_point_size_label: ttk.Label | None = None
        self._sweep_redraw_after_id: str | None = None
        self.inspection_selected_point: tuple[float, float] | None = None
        self.probe_state_names: list[str] = []
        self.probe_x_state = ""
        self.probe_y_state = ""
        self.probe_selected: list[dict] = []
        self.probe_benchmark: list[dict] = []
        self.probe_phase_start: float | None = None
        self.probe_phase_end: float | None = None
        self.zoom_views: dict[str, tuple[float, float, float, float]] = {}
        self.plot_viewports: dict[str, tuple[int, int, int, int, float, float, float, float]] = {}
        self.zoom_drag_start: dict[str, tuple[int, int]] = {}
        self.zoom_rect_items: dict[str, int] = {}
        self.main_pane: ttk.PanedWindow | None = None
        self.run_in_progress = False

        self._build_variable_store()
        self._build_ui()
        self.load_config(DEFAULT_CONFIG_PATH)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(120, self._poll_log_queue)

    def _build_variable_store(self) -> None:
        keys = [
            "analysis_mode",
            "sweep_parameter",
            "sweep_min_value",
            "sweep_max_value",
            "sweep_num_values",
            "state_names",
            "parameter_names",
            "parameter_values",
            "state_defaults",
            "x_state",
            "y_state",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "grid_mode",
            "n_target",
            "nx",
            "ny",
            "solver_setup",
            "device",
            "precision",
            "solver_mode",
            "solver",
            "period_mode",
            "custom_time_argument",
            "period_expression",
            "transient_periods",
            "evaluation_periods",
            "samples_per_period",
            "t_transient",
            "t_evaluation",
            "dt",
            "save_dt",
            "streaming_chunk_periods",
            "abstol",
            "reltol",
            "observable_state",
            "zero_cross_state",
            "fingerprint_tol",
            "fingerprint_k",
            "extrema_eps",
            "max_extrema",
            "run_name",
            "write_labels_csv",
            "write_basin_image",
            "write_sweep_details",
        ]
        for key in keys:
            self.vars[key] = tk.StringVar()
        self.vars["state_names"].trace_add("write", lambda *_args: self._sync_equation_rows())
        self.vars["solver_setup"].trace_add("write", lambda *_args: self._on_solver_setup_changed())
        self.vars["parameter_names"].trace_add("write", lambda *_args: self._schedule_equation_validation())
        self.vars["device"].trace_add("write", lambda *_args: self._schedule_equation_validation())

    def _build_ui(self) -> None:
        root = ttk.PanedWindow(self, orient="horizontal")
        self.main_pane = root
        root.pack(fill="both", expand=True, padx=10, pady=10)

        left = ttk.Frame(root, padding=8)
        right = ttk.Frame(root, padding=8)
        root.add(left, weight=1)
        root.add(right, weight=2)

        notebook = ttk.Notebook(left)
        notebook.pack(fill="both", expand=True)

        analysis_tab = ttk.Frame(notebook, padding=10)
        model_tab = ttk.Frame(notebook, padding=10)
        grid_tab = ttk.Frame(notebook, padding=10)
        integration_tab = ttk.Frame(notebook, padding=10)
        class_tab = ttk.Frame(notebook, padding=10)

        notebook.add(analysis_tab, text="Analysis")
        notebook.add(model_tab, text="Model")
        notebook.add(grid_tab, text="Grid")
        notebook.add(integration_tab, text="Integration")
        notebook.add(class_tab, text="Classification")

        self._build_analysis_tab(analysis_tab)
        self._build_model_tab(model_tab)
        self._build_grid_tab(grid_tab)
        self._build_integration_tab(integration_tab)
        self._build_classification_tab(class_tab)

        action_frame = ttk.Frame(right)
        action_frame.pack(fill="x", pady=(0, 10))

        self.run_button = ttk.Button(action_frame, text="Run", command=self.run_backend)
        self.run_button.pack(side="left", padx=(0, 8))

        self.cancel_button = ttk.Button(action_frame, text="Cancel", command=self.cancel_run, state="disabled")
        self.cancel_button.pack(side="left", padx=(0, 8))

        ttk.Button(action_frame, text="Load Defaults", command=lambda: self.load_config(DEFAULT_CONFIG_PATH)).pack(side="left", padx=(0, 8))
        ttk.Button(action_frame, text="Load Config", command=self.load_config_dialog).pack(side="left", padx=(0, 8))
        ttk.Button(action_frame, text="Save Config", command=self.save_config_dialog).pack(side="left", padx=(0, 8))
        ttk.Button(action_frame, text="Load Results", command=self.load_result_dialog).pack(side="left", padx=(0, 8))
        ttk.Button(action_frame, text="Open Result Folder", command=self.open_result_dir).pack(side="left", padx=(0, 8))
        ttk.Button(action_frame, text="Restart Julia", command=self.restart_backend).pack(side="left")

        right_notebook = ttk.Notebook(right)
        right_notebook.pack(fill="both", expand=True)

        log_tab = ttk.Frame(right_notebook, padding=8)
        results_tab = ttk.Frame(right_notebook, padding=8)
        right_notebook.add(log_tab, text="Log")
        right_notebook.add(results_tab, text="Results")

        self.log_text = ScrolledText(log_tab, height=18, wrap="word", font=("Consolas", 10))
        self.log_text.pack(fill="both", expand=True)

        summary_label = ttk.Label(results_tab, text="Summary", font=("Segoe UI", 11, "bold"))
        summary_label.pack(anchor="w")

        self.summary_text = ScrolledText(results_tab, height=16, wrap="word", font=("Consolas", 10))
        self.summary_text.pack(fill="x", expand=False, pady=(6, 10))

        self.plot_notebook = ttk.Notebook(results_tab)
        self.plot_notebook.pack(fill="both", expand=True)

        single_tab = ttk.Frame(self.plot_notebook, padding=6)
        sweep_tab = ttk.Frame(self.plot_notebook, padding=6)
        inspection_tab = ttk.Frame(self.plot_notebook, padding=6)
        self.plot_notebook.add(single_tab, text="Single Basin")
        self.plot_notebook.add(sweep_tab, text="Parameter Sweep")
        self.plot_notebook.add(inspection_tab, text="Point Inspection")

        single_pane = ttk.PanedWindow(single_tab, orient="horizontal")
        single_pane.pack(fill="both", expand=True)

        basin_tab = ttk.Frame(single_pane, padding=4)
        phase_tab = ttk.Frame(single_pane, padding=4)
        single_pane.add(basin_tab, weight=1)
        single_pane.add(phase_tab, weight=1)

        single_controls = ttk.Frame(basin_tab)
        single_controls.pack(anchor="w", fill="x", pady=(0, 6))
        ttk.Button(single_controls, text="Export Images", command=self.export_current_images).pack(side="left")
        self.basin_info = ttk.Label(basin_tab, text="No results loaded yet.")
        self.basin_info.pack(anchor="w", pady=(0, 6))
        self.basin_canvas = tk.Canvas(basin_tab, background="#ffffff", highlightthickness=0)
        self.basin_canvas.pack(fill="both", expand=True)
        self.basin_canvas.bind("<Configure>", lambda _event: self._redraw_basin_canvas())
        self._bind_basin_recompute_zoom(self.basin_canvas)

        phase_controls = ttk.Frame(phase_tab)
        phase_controls.pack(anchor="w", fill="x", pady=(0, 6))
        ttk.Label(phase_controls, text="Min class fraction").pack(side="left")
        phase_min_entry = ttk.Entry(phase_controls, textvariable=self.phase_min_fraction_var, width=8)
        phase_min_entry.pack(side="left", padx=(4, 8))
        ttk.Button(phase_controls, text="Redraw", command=self._redraw_phase_canvas).pack(side="left")
        self.phase_filter_status_label = ttk.Label(phase_controls, text="", foreground="#555555")
        self.phase_filter_status_label.pack(side="left", padx=(10, 0))
        phase_min_entry.bind("<Return>", lambda _event: self._redraw_phase_canvas())
        self.phase_info = ttk.Label(phase_tab, text="No phase data loaded yet.")
        self.phase_info.pack(anchor="w", pady=(0, 6))
        self.phase_canvas = tk.Canvas(phase_tab, background="#ffffff", highlightthickness=0)
        self.phase_canvas.pack(fill="both", expand=True)
        self.phase_canvas.bind("<Configure>", lambda _event: self._redraw_phase_canvas())
        self._bind_plot_zoom(self.phase_canvas, "phase", self._redraw_phase_canvas)

        self.sweep_info = ttk.Label(sweep_tab, text="No sweep data loaded yet.")
        self.sweep_info.pack(anchor="w", pady=(0, 6))
        sweep_controls = ttk.Frame(sweep_tab)
        sweep_controls.pack(fill="x", pady=(0, 6))
        ttk.Label(sweep_controls, text="Min fraction").pack(side="left")
        min_fraction_entry = ttk.Entry(sweep_controls, textvariable=self.sweep_min_fraction_var, width=8)
        min_fraction_entry.pack(side="left", padx=(4, 12))
        ttk.Label(sweep_controls, text="Max classes / w").pack(side="left")
        max_classes_entry = ttk.Entry(sweep_controls, textvariable=self.sweep_max_classes_var, width=6)
        max_classes_entry.pack(side="left", padx=(4, 12))
        ttk.Label(sweep_controls, text="Max points").pack(side="left")
        max_points_entry = ttk.Entry(sweep_controls, textvariable=self.sweep_max_points_var, width=8)
        max_points_entry.pack(side="left", padx=(4, 12))
        ttk.Label(sweep_controls, text="Color").pack(side="left")
        color_combo = ttk.Combobox(
            sweep_controls,
            textvariable=self.sweep_color_mode_var,
            values=["Label", "Feature"],
            width=8,
            state="readonly",
        )
        color_combo.pack(side="left", padx=(4, 12))
        ttk.Label(sweep_controls, text="Point size").pack(side="left")
        point_size_scale = ttk.Scale(
            sweep_controls,
            from_=0.3,
            to=4.0,
            variable=self.sweep_point_size_var,
            command=self._on_sweep_point_size_changed,
            length=110,
        )
        point_size_scale.pack(side="left", padx=(4, 4))
        self.sweep_point_size_label = ttk.Label(
            sweep_controls,
            text=f"{SWEEP_POINT_SIZE_DEFAULT:.1f}",
            width=4,
            foreground="#555555",
        )
        self.sweep_point_size_label.pack(side="left", padx=(0, 12))
        ttk.Button(sweep_controls, text="Redraw", command=self._redraw_sweep_canvas).pack(side="left")
        ttk.Button(sweep_controls, text="Export Images", command=self.export_current_images).pack(side="left", padx=(8, 0))
        ttk.Button(sweep_controls, text="Export Video", command=self.export_sweep_video).pack(side="left", padx=(4, 0))
        self.sweep_filter_status_label = ttk.Label(sweep_controls, text="", foreground="#555555")
        self.sweep_filter_status_label.pack(side="left", padx=(12, 0))
        for widget in (min_fraction_entry, max_classes_entry, max_points_entry):
            widget.bind("<Return>", lambda _event: self._redraw_sweep_canvas())
        color_combo.bind("<<ComboboxSelected>>", lambda _event: self._redraw_color_sensitive_plots())
        self.sweep_canvas = tk.Canvas(sweep_tab, background="#ffffff", highlightthickness=0)
        self.sweep_canvas.pack(fill="both", expand=True)
        self.sweep_canvas.bind("<Configure>", lambda _event: self._redraw_sweep_canvas())
        self._bind_plot_zoom(self.sweep_canvas, "sweep", self._redraw_sweep_canvas, self._on_sweep_click)

        inspection_pane = ttk.PanedWindow(inspection_tab, orient="horizontal")
        inspection_pane.pack(fill="both", expand=True)

        inspection_left = ttk.Frame(inspection_pane, padding=4)
        inspection_right = ttk.PanedWindow(inspection_pane, orient="vertical")
        inspection_pane.add(inspection_left, weight=1)
        inspection_pane.add(inspection_right, weight=2)

        self.inspection_info = ttk.Label(
            inspection_left,
            text="Select Analysis Mode = inspection, press Run, then click an initial condition.",
            wraplength=320,
        )
        self.inspection_info.pack(anchor="w", pady=(0, 6))
        self.inspection_canvas = tk.Canvas(inspection_left, background="#ffffff", highlightthickness=0)
        self.inspection_canvas.pack(fill="both", expand=True)
        self.inspection_canvas.bind("<Configure>", lambda _event: self._redraw_inspection_canvas())
        self._bind_plot_zoom(self.inspection_canvas, "inspection", self._redraw_inspection_canvas, self._on_inspection_click)

        time_tab = ttk.Frame(inspection_right, padding=4)
        phase_probe_tab = ttk.Frame(inspection_right, padding=4)
        inspection_right.add(time_tab, weight=1)
        inspection_right.add(phase_probe_tab, weight=1)

        self.probe_time_info = ttk.Label(time_tab, text="No point trajectory loaded yet.")
        self.probe_time_info.pack(anchor="w", pady=(0, 6))
        self.probe_time_canvas = tk.Canvas(time_tab, background="#ffffff", highlightthickness=0)
        self.probe_time_canvas.pack(fill="both", expand=True)
        self.probe_time_canvas.bind("<Configure>", lambda _event: self._redraw_probe_time_canvas())
        self._bind_plot_zoom(self.probe_time_canvas, "probe_time", self._redraw_probe_time_canvas)

        self.probe_phase_info = ttk.Label(phase_probe_tab, text="No point phase trajectory loaded yet.")
        self.probe_phase_info.pack(anchor="w", pady=(0, 6))
        self.probe_phase_canvas = tk.Canvas(phase_probe_tab, background="#ffffff", highlightthickness=0)
        self.probe_phase_canvas.pack(fill="both", expand=True)
        self.probe_phase_canvas.bind("<Configure>", lambda _event: self._redraw_probe_phase_canvas())
        self._bind_plot_zoom(self.probe_phase_canvas, "probe_phase", self._redraw_probe_phase_canvas)

        self.image_info = self.basin_info
        self.after(250, self._set_default_main_split)

    def _set_default_main_split(self) -> None:
        if self.main_pane is None:
            return
        width = self.main_pane.winfo_width()
        if width <= 1:
            self.after(250, self._set_default_main_split)
            return
        self.main_pane.sashpos(0, max(280, width // 3))

    def _build_analysis_tab(self, parent: ttk.Frame) -> None:
        self._combo(parent, "Analysis Mode", "analysis_mode", 0, ["single_basin", "parameter_sweep", "inspection"])
        self._entry(parent, "Sweep Parameter", "sweep_parameter", 1)
        self._entry(parent, "Sweep Minimum", "sweep_min_value", 2)
        self._entry(parent, "Sweep Maximum", "sweep_max_value", 3)
        self._entry(parent, "Sweep Values", "sweep_num_values", 4)

    def _build_model_tab(self, parent: ttk.Frame) -> None:
        form = ttk.Frame(parent)
        form.grid(row=0, column=0, sticky="ew")
        self._entry(form, "States (CSV)", "state_names", 0, width=34)
        self._entry(form, "Parameters (CSV)", "parameter_names", 1, width=34)
        self._entry(form, "Parameter Values (CSV)", "parameter_values", 2, width=34)
        self._entry(form, "Default Initial State (CSV)", "state_defaults", 3, width=34)

        self._label_with_help(parent, "Equations (Julia syntax)", "equations", 1, pady=(14, 4))
        guide = (
            "One RHS expression per state. Example: dx/dt = v, dv/dt = -d*v - x^3 + F0*cos(w*t). "
            "Allowed for GPU: states, parameters, t, phase, numbers, pi, + - * / ^, parentheses, "
            "sin/cos/tan/exp/log/sqrt/abs/min/max/ifelse. No arrays, indexing, assignments, or custom functions."
        )
        guide_label = ttk.Label(parent, text=guide, wraplength=460, foreground="#555555", justify="left")
        guide_label.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.equation_frame = ttk.Frame(parent)
        self.equation_frame.grid(row=3, column=0, sticky="nsew")
        self.equation_frame.columnconfigure(1, weight=1)
        self.equation_status_label = ttk.Label(parent, text="", wraplength=460, justify="left")
        self.equation_status_label.grid(row=4, column=0, sticky="ew", pady=(6, 0))

        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(3, weight=1)
        self._sync_equation_rows()

    def _build_grid_tab(self, parent: ttk.Frame) -> None:
        self._entry(parent, "X Axis State", "x_state", 0)
        self._entry(parent, "Y Axis State", "y_state", 1)
        self._entry(parent, "x_min", "x_min", 2)
        self._entry(parent, "x_max", "x_max", 3)
        self._entry(parent, "y_min", "y_min", 4)
        self._entry(parent, "y_max", "y_max", 5)
        self._combo(parent, "Grid Mode", "grid_mode", 6, ["auto", "manual"])
        self._entry(parent, "Target Points N_target", "n_target", 7)
        self._entry(parent, "nx (manual)", "nx", 8)
        self._entry(parent, "ny (manual)", "ny", 9)
    def _build_integration_tab(self, parent: ttk.Frame) -> None:
        self._solver_setup_control(parent, 0)
        self._combo(parent, "Time Mode", "period_mode", 2, ["periodic", "direct_time"])
        self._combo(parent, "Custom Time Arg", "custom_time_argument", 3, ["time", "phase"])
        self._entry(parent, "Period Expression", "period_expression", 4, width=40)
        self._entry(parent, "Transient Periods", "transient_periods", 5)
        self._entry(parent, "Evaluation Periods", "evaluation_periods", 6)
        self._entry(parent, "Samples per Period", "samples_per_period", 7)
        self._entry(parent, "t_transient (direct_time)", "t_transient", 8)
        self._entry(parent, "t_evaluation (direct_time)", "t_evaluation", 9)
        self._entry(parent, "dt", "dt", 10)
        self._entry(parent, "save_dt", "save_dt", 11)
        self._entry(parent, "Streaming Chunk Periods", "streaming_chunk_periods", 12)
        self._entry(parent, "abstol", "abstol", 13)
        self._entry(parent, "reltol", "reltol", 14)

    def _build_classification_tab(self, parent: ttk.Frame) -> None:
        self._entry(parent, "Observable State", "observable_state", 0)
        self._entry(parent, "Zero-Crossing State", "zero_cross_state", 1)
        self._entry(parent, "Fingerprint Tolerance", "fingerprint_tol", 2)
        self._entry(parent, "Fingerprint-K", "fingerprint_k", 3)
        self._entry(parent, "Extrema Epsilon", "extrema_eps", 4)
        self._entry(parent, "Maximum Extrema", "max_extrema", 5)
        self._entry(parent, "Run Name", "run_name", 6)
        self._combo(parent, "Write Labels CSV", "write_labels_csv", 7, ["true", "false"])
        self._combo(parent, "Write Basin Image", "write_basin_image", 8, ["true", "false"])
        self._combo(parent, "Write Sweep Details", "write_sweep_details", 9, ["true", "false"])

    def _label_with_help(self, parent: ttk.Frame, label: str, key: str, row: int, *, pady=4) -> None:
        label_frame = ttk.Frame(parent)
        label_frame.grid(row=row, column=0, sticky="w", pady=pady, padx=(0, 10))
        text_label = ttk.Label(label_frame, text=label)
        text_label.pack(side="left")
        help_text = PARAM_HELP.get(key)
        if help_text:
            Tooltip(text_label, help_text)
            help_label = ttk.Label(label_frame, text="?", foreground="#1f5aa6", cursor="hand2")
            help_label.pack(side="left", padx=(6, 0))
            Tooltip(help_label, help_text)

    def _entry(self, parent: ttk.Frame, label: str, key: str, row: int, width: int = 26) -> None:
        self._label_with_help(parent, label, key, row)
        ttk.Entry(parent, textvariable=self.vars[key], width=width).grid(row=row, column=1, sticky="ew", pady=4)
        parent.columnconfigure(1, weight=1)

    def _combo(self, parent: ttk.Frame, label: str, key: str, row: int, values: list[str]) -> None:
        self._label_with_help(parent, label, key, row)
        combo = ttk.Combobox(parent, textvariable=self.vars[key], values=values, state="readonly", width=24)
        combo.grid(row=row, column=1, sticky="w", pady=4)

    def _solver_setup_control(self, parent: ttk.Frame, row: int) -> None:
        self._label_with_help(parent, "Solver Setup", "solver_setup", row)
        combo = ttk.Combobox(
            parent,
            textvariable=self.vars["solver_setup"],
            values=SOLVER_SETUP_LABELS,
            state="readonly",
            width=44,
        )
        combo.grid(row=row, column=1, sticky="ew", pady=4)
        self.solver_setup_description_label = ttk.Label(
            parent,
            text="",
            wraplength=520,
            justify="left",
            foreground="#555555",
        )
        self.solver_setup_description_label.grid(row=row + 1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        parent.columnconfigure(1, weight=1)

    def _on_solver_setup_changed(self, *, sync_time_argument: bool = True) -> None:
        setup = SOLVER_SETUP_BY_LABEL.get(self.vars["solver_setup"].get())
        if setup:
            for key in ("device", "precision", "solver_mode", "solver"):
                self.vars[key].set(setup[key])
            if sync_time_argument:
                if setup["solver_mode"] == "rk4_full_extrema_custom":
                    self.vars["custom_time_argument"].set("phase")
                elif self.vars["custom_time_argument"].get() == "phase":
                    self.vars["custom_time_argument"].set("time")
        self._update_solver_setup_description()

    def _solver_setup_label_for_fields(self, device: str, precision: str, solver_mode: str, solver: str) -> str:
        setup = SOLVER_SETUP_BY_FIELDS.get((device, precision, solver_mode, solver))
        if setup:
            return setup["label"]
        return f"Custom - {device} / {precision} / {solver_mode} / {solver}"

    def _sync_solver_setup_from_fields(self) -> None:
        label = self._solver_setup_label_for_fields(
            self.vars["device"].get().strip(),
            self.vars["precision"].get().strip(),
            self.vars["solver_mode"].get().strip(),
            self.vars["solver"].get().strip(),
        )
        self.vars["solver_setup"].set(label)
        self._update_solver_setup_description()

    def _update_solver_setup_description(self) -> None:
        if self.solver_setup_description_label is None:
            return
        label = self.vars["solver_setup"].get()
        setup = SOLVER_SETUP_BY_LABEL.get(label)
        if setup:
            backend = f"Backend: {setup['device']} / {setup['precision']} / {setup['solver_mode']} / {setup['solver']}."
            self.solver_setup_description_label.configure(text=f"{setup['description']} {backend}", foreground="#555555")
            return
        device = self.vars["device"].get().strip()
        precision = self.vars["precision"].get().strip()
        solver_mode = self.vars["solver_mode"].get().strip()
        solver = self.vars["solver"].get().strip()
        text = (
            "Custom combination loaded from the config. "
            f"Backend: {device} / {precision} / {solver_mode} / {solver}. "
            "Select one of the listed setups to switch back to a known valid combination."
        )
        self.solver_setup_description_label.configure(text=text, foreground="#8a5a00")

    def _current_equations(self) -> list[str]:
        return [var.get().strip() for var in self.equation_vars]

    def _set_equation_values(self, equations: list[str]) -> None:
        self._sync_equation_rows(equations)

    def _schedule_equation_validation(self) -> None:
        self.after_idle(self._update_equation_status)

    def _validate_identifier_list(self, names: list[str], kind: str) -> None:
        seen: set[str] = set()
        for name in names:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(f"{kind} name '{name}' is invalid. Use ASCII names starting with a letter or underscore.")
            if name in RESERVED_MODEL_NAMES:
                raise ValueError(f"{kind} name '{name}' is reserved for the expression language.")
            if name in seen:
                raise ValueError(f"{kind} name '{name}' is duplicated.")
            seen.add(name)

    def _validate_equations_for_gui(self, *, raise_on_error: bool = False) -> tuple[bool, str]:
        try:
            state_names = parse_csv_strings(self.vars["state_names"].get())
            parameter_names = parse_csv_strings(self.vars["parameter_names"].get())
            equations = self._current_equations()
            if not state_names:
                raise ValueError("Define at least one state.")
            if len(state_names) != len(equations):
                raise ValueError("The number of equations must match the number of states.")
            if any(not equation for equation in equations):
                raise ValueError("Every state needs a non-empty equation.")
            self._validate_identifier_list(state_names, "State")
            self._validate_identifier_list(parameter_names, "Parameter")
            overlap = sorted(set(state_names) & set(parameter_names))
            if overlap:
                raise ValueError(f"State and parameter names must not overlap: {', '.join(overlap)}.")

            allowed_names = set(state_names) | set(parameter_names) | {"t", "phase", "pi", "π"} | GPU_SAFE_FUNCTIONS
            for equation in equations:
                bad_char = re.search(r"[^A-Za-z0-9_π\s+\-*/^().,<>=!&|]", equation)
                if bad_char:
                    raise ValueError(f"Unsupported character '{bad_char.group(0)}' in '{equation}'.")
                if any(char in equation for char in "[]{};"):
                    raise ValueError(f"Unsupported syntax in '{equation}'. Arrays, indexing, braces, and semicolons are not allowed.")
                if "=" in equation and not any(op in equation for op in ("==", "<=", ">=", "!=")):
                    raise ValueError(f"Unsupported assignment in '{equation}'. Enter only the right-hand side expression.")
                for token in re.findall(r"(?<![\d.])\b[A-Za-z_][A-Za-z0-9_]*\b|π", equation):
                    if token not in allowed_names:
                        raise ValueError(f"Unknown or unsupported symbol '{token}' in '{equation}'.")
                for call_name in re.findall(r"([A-Za-z_][A-Za-z0-9_]*|π)\s*\(", equation):
                    if call_name not in GPU_SAFE_FUNCTIONS:
                        raise ValueError(f"Unsupported function '{call_name}' in '{equation}'.")
            return True, "Equation syntax looks GPU-safe."
        except Exception as exc:
            if raise_on_error:
                raise
            return False, str(exc)

    def _update_equation_status(self) -> None:
        if self.equation_status_label is None:
            return
        ok, message = self._validate_equations_for_gui()
        if ok:
            self.equation_status_label.configure(text=message, foreground="#1d6f42")
        else:
            self.equation_status_label.configure(text=f"Equation issue: {message}", foreground="#a12622")

    def _sync_equation_rows(self, equations: list[str] | None = None) -> None:
        if self.equation_frame is None:
            return

        state_names = parse_csv_strings(self.vars["state_names"].get())
        existing = equations if equations is not None else self._current_equations()
        row_count = len(state_names) if state_names else max(len(existing), 1)

        for child in self.equation_frame.winfo_children():
            child.destroy()

        self.equation_vars = []
        for row in range(row_count):
            state_name = state_names[row] if row < len(state_names) else f"state{row + 1}"
            equation_value = existing[row] if row < len(existing) else ""
            variable = tk.StringVar(value=equation_value)
            variable.trace_add("write", lambda *_args: self._schedule_equation_validation())
            self.equation_vars.append(variable)

            label = ttk.Label(self.equation_frame, text=f"d{state_name}/dt =")
            label.grid(row=row, column=0, sticky="e", padx=(0, 8), pady=4)
            entry = ttk.Entry(self.equation_frame, textvariable=variable, font=("Consolas", 10))
            entry.grid(row=row, column=1, sticky="ew", pady=4)
        self._schedule_equation_validation()

    def _bind_plot_zoom(self, canvas: tk.Canvas, plot_key: str, redraw_callback, click_callback=None) -> None:
        canvas.bind("<ButtonPress-1>", lambda event: self._start_rect_zoom(event, plot_key), add="+")
        canvas.bind("<B1-Motion>", lambda event: self._drag_rect_zoom(event, plot_key), add="+")
        canvas.bind("<ButtonRelease-1>", lambda event: self._finish_rect_zoom(event, plot_key, redraw_callback, click_callback), add="+")
        canvas.bind("<Button-3>", lambda _event: self._reset_plot_zoom(plot_key, redraw_callback), add="+")

    def _bind_basin_recompute_zoom(self, canvas: tk.Canvas) -> None:
        canvas.bind("<ButtonPress-1>", lambda event: self._start_rect_zoom(event, "basin"), add="+")
        canvas.bind("<B1-Motion>", lambda event: self._drag_rect_zoom(event, "basin"), add="+")
        canvas.bind("<ButtonRelease-1>", self._finish_basin_recompute_zoom, add="+")
        canvas.bind("<Button-3>", self._restore_previous_basin_zoom, add="+")

    def _apply_zoom_view(self, plot_key: str, bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return self.zoom_views.get(plot_key, bounds)

    def _remember_plot_view(
        self,
        plot_key: str,
        margin_left: int,
        margin_top: int,
        plot_w: int,
        plot_h: int,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ) -> None:
        self.plot_viewports[plot_key] = (margin_left, margin_top, plot_w, plot_h, xmin, xmax, ymin, ymax)

    def _clamp_to_viewport(self, plot_key: str, x: int, y: int) -> tuple[int, int] | None:
        viewport = self.plot_viewports.get(plot_key)
        if viewport is None:
            return None

        margin_left, margin_top, plot_w, plot_h, _xmin, _xmax, _ymin, _ymax = viewport
        clamped_x = min(max(int(x), margin_left), margin_left + plot_w)
        clamped_y = min(max(int(y), margin_top), margin_top + plot_h)
        return clamped_x, clamped_y

    def _start_rect_zoom(self, event: tk.Event, plot_key: str):
        viewport = self.plot_viewports.get(plot_key)
        if viewport is None:
            return "break"

        margin_left, margin_top, plot_w, plot_h, _xmin, _xmax, _ymin, _ymax = viewport
        if not (margin_left <= event.x <= margin_left + plot_w and margin_top <= event.y <= margin_top + plot_h):
            return "break"

        self.zoom_drag_start[plot_key] = (int(event.x), int(event.y))
        self._delete_zoom_rect(plot_key, event.widget)
        return "break"

    def _drag_rect_zoom(self, event: tk.Event, plot_key: str):
        start = self.zoom_drag_start.get(plot_key)
        if start is None:
            return "break"

        clamped = self._clamp_to_viewport(plot_key, int(event.x), int(event.y))
        if clamped is None:
            return "break"

        canvas = event.widget
        self._delete_zoom_rect(plot_key, canvas)
        x0, y0 = start
        x1, y1 = clamped
        self.zoom_rect_items[plot_key] = canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            outline="#111111",
            width=1,
            dash=(4, 3),
        )
        return "break"

    def _finish_rect_zoom(self, event: tk.Event, plot_key: str, redraw_callback, click_callback=None):
        start = self.zoom_drag_start.pop(plot_key, None)
        if start is None:
            return "break"

        canvas = event.widget
        self._delete_zoom_rect(plot_key, canvas)
        clamped = self._clamp_to_viewport(plot_key, int(event.x), int(event.y))
        viewport = self.plot_viewports.get(plot_key)
        if clamped is None or viewport is None:
            return "break"

        x0, y0 = start
        x1, y1 = clamped
        if abs(x1 - x0) < 8 or abs(y1 - y0) < 8:
            if click_callback is not None:
                click_callback(event)
            return "break"

        margin_left, margin_top, plot_w, plot_h, xmin, xmax, ymin, ymax = viewport
        px_min, px_max = sorted((x0, x1))
        py_min, py_max = sorted((y0, y1))
        new_xmin = xmin + (px_min - margin_left) / plot_w * (xmax - xmin)
        new_xmax = xmin + (px_max - margin_left) / plot_w * (xmax - xmin)
        y_at_top = ymax - (py_min - margin_top) / plot_h * (ymax - ymin)
        y_at_bottom = ymax - (py_max - margin_top) / plot_h * (ymax - ymin)
        new_ymin, new_ymax = sorted((y_at_bottom, y_at_top))
        if new_xmin < new_xmax and new_ymin < new_ymax:
            self.zoom_views[plot_key] = (new_xmin, new_xmax, new_ymin, new_ymax)
            redraw_callback()
        return "break"

    def _delete_zoom_rect(self, plot_key: str, canvas: tk.Canvas) -> None:
        item = self.zoom_rect_items.pop(plot_key, None)
        if item is not None:
            canvas.delete(item)

    def _reset_plot_zoom(self, plot_key: str, redraw_callback):
        self.zoom_views.pop(plot_key, None)
        self.zoom_drag_start.pop(plot_key, None)
        self.zoom_rect_items.pop(plot_key, None)
        redraw_callback()
        return "break"

    def _clear_plot_zoom(self, *plot_keys: str) -> None:
        if plot_keys:
            for key in plot_keys:
                self.zoom_views.pop(key, None)
                self.plot_viewports.pop(key, None)
                self.zoom_drag_start.pop(key, None)
                self.zoom_rect_items.pop(key, None)
        else:
            self.zoom_views.clear()
            self.plot_viewports.clear()
            self.zoom_drag_start.clear()
            self.zoom_rect_items.clear()

    def _current_grid_snapshot(self) -> dict[str, str]:
        keys = [
            "analysis_mode",
            "parameter_values",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "grid_mode",
            "n_target",
            "nx",
            "ny",
        ]
        return {key: self.vars[key].get() for key in keys}

    def _restore_grid_snapshot(self, snapshot: dict[str, str]) -> None:
        for key, value in snapshot.items():
            if key in self.vars:
                self.vars[key].set(value)

    def _current_plane_bounds(self) -> tuple[float, float, float, float] | None:
        try:
            bounds = (
                float(self.vars["x_min"].get()),
                float(self.vars["x_max"].get()),
                float(self.vars["y_min"].get()),
                float(self.vars["y_max"].get()),
            )
        except ValueError:
            return None

        xmin, xmax, ymin, ymax = bounds
        if xmin == xmax or ymin == ymax:
            return None
        return bounds

    def _plane_bounds_from_dict(self, plane: dict) -> tuple[float, float, float, float] | None:
        try:
            bounds = (
                float(plane["x_min"]),
                float(plane["x_max"]),
                float(plane["y_min"]),
                float(plane["y_max"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

        xmin, xmax, ymin, ymax = bounds
        if xmin == xmax or ymin == ymax:
            return None
        return bounds

    def _summary_plane_bounds(self, summary: dict) -> tuple[float, float, float, float] | None:
        bounds = self._plane_bounds_from_dict(summary.get("initial_condition_plane", {}))
        if bounds is not None:
            return bounds

        config_snapshot = summary.get("run", {}).get("config_snapshot", "")
        if config_snapshot:
            try:
                snapshot_path = Path(str(config_snapshot))
                if snapshot_path.is_file():
                    with snapshot_path.open("rb") as handle:
                        config = tomllib.load(handle)
                    bounds = self._plane_bounds_from_dict(config.get("initial_condition_plane", {}))
                    if bounds is not None:
                        return bounds
            except Exception:
                pass
        return self._current_plane_bounds()

    def _apply_basin_zoom_bounds_to_config(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        nx: int,
        ny: int,
    ) -> None:
        self.vars["analysis_mode"].set("single_basin")
        self.vars["x_min"].set(format_numeric_value(xmin))
        self.vars["x_max"].set(format_numeric_value(xmax))
        self.vars["y_min"].set(format_numeric_value(ymin))
        self.vars["y_max"].set(format_numeric_value(ymax))
        self.vars["grid_mode"].set("manual")
        self.vars["n_target"].set(str(nx * ny))
        self.vars["nx"].set(str(nx))
        self.vars["ny"].set(str(ny))

    def _set_selected_sweep_parameter_for_basin_zoom(self) -> None:
        if self.current_result_mode != "parameter_sweep" or self.selected_sweep_value is None:
            return

        summary = self.current_summary_data or {}
        sweep_parameter = str(summary.get("sweep", {}).get("parameter", "")).strip()
        if not sweep_parameter:
            return

        parameter_names = parse_csv_strings(self.vars["parameter_names"].get())
        parameter_values = parse_csv_numbers(self.vars["parameter_values"].get())
        if len(parameter_names) != len(parameter_values) or sweep_parameter not in parameter_names:
            return

        parameter_values[parameter_names.index(sweep_parameter)] = float(self.selected_sweep_value)
        self.vars["parameter_values"].set(", ".join(format_numeric_value(value) for value in parameter_values))

    def _snapshot_bounds(self, snapshot: dict[str, str]) -> tuple[float, float, float, float] | None:
        try:
            bounds = (
                float(snapshot["x_min"]),
                float(snapshot["x_max"]),
                float(snapshot["y_min"]),
                float(snapshot["y_max"]),
            )
        except (KeyError, ValueError):
            return None

        xmin, xmax, ymin, ymax = bounds
        if xmin == xmax or ymin == ymax:
            return None
        return bounds

    def _basin_zoom_factor_label(self, bounds: tuple[float, float, float, float]) -> str:
        reference = self._snapshot_bounds(self.basin_zoom_history[0]) if self.basin_zoom_history else None
        if reference is None:
            return "Zoom: 1x"

        xmin, xmax, ymin, ymax = bounds
        ref_xmin, ref_xmax, ref_ymin, ref_ymax = reference
        x_factor = abs((ref_xmax - ref_xmin) / (xmax - xmin))
        y_factor = abs((ref_ymax - ref_ymin) / (ymax - ymin))
        area_factor = x_factor * y_factor
        return (
            f"Zoom: {self._format_factor(area_factor)} area | "
            f"x {self._format_factor(x_factor)}, y {self._format_factor(y_factor)}"
        )

    def _format_factor(self, value: float) -> str:
        if not math.isfinite(value) or value <= 0:
            return "?"
        if value < 10:
            text = f"{value:.2f}".rstrip("0").rstrip(".")
            return f"{text}x"
        if value < 100:
            text = f"{value:.1f}".rstrip("0").rstrip(".")
            return f"{text}x"
        return f"{value:.0f}x"

    def _finish_basin_recompute_zoom(self, event: tk.Event):
        start = self.zoom_drag_start.pop("basin", None)
        if start is None:
            return "break"

        canvas = event.widget
        self._delete_zoom_rect("basin", canvas)
        if self.run_in_progress:
            self.basin_info.configure(text="A computation is already running.")
            return "break"

        clamped = self._clamp_to_viewport("basin", int(event.x), int(event.y))
        viewport = self.plot_viewports.get("basin")
        if clamped is None or viewport is None:
            return "break"

        x0, y0 = start
        x1, y1 = clamped
        if abs(x1 - x0) < 8 or abs(y1 - y0) < 8:
            return "break"

        nrows = len(self.basin_rows)
        ncols = max((len(row) for row in self.basin_rows), default=0)
        if nrows <= 0 or ncols <= 0:
            return "break"

        margin_left, margin_top, plot_w, plot_h, xmin, xmax, ymin, ymax = viewport
        px_min, px_max = sorted((x0, x1))
        py_min, py_max = sorted((y0, y1))
        new_xmin = xmin + (px_min - margin_left) / plot_w * (xmax - xmin)
        new_xmax = xmin + (px_max - margin_left) / plot_w * (xmax - xmin)
        y_at_top = ymax - (py_min - margin_top) / plot_h * (ymax - ymin)
        y_at_bottom = ymax - (py_max - margin_top) / plot_h * (ymax - ymin)
        new_ymin, new_ymax = sorted((y_at_bottom, y_at_top))
        if not (new_xmin < new_xmax and new_ymin < new_ymax):
            return "break"

        self._set_selected_sweep_parameter_for_basin_zoom()
        previous_extent = self._current_grid_snapshot()
        previous_extent["analysis_mode"] = "single_basin"
        self.basin_zoom_history.append(previous_extent)
        self._apply_basin_zoom_bounds_to_config(new_xmin, new_xmax, new_ymin, new_ymax, ncols, nrows)
        self.basin_bounds = (new_xmin, new_xmax, new_ymin, new_ymax)
        self.basin_info.configure(
            text=(
                f"Recomputing zoom: x={new_xmin:.6g}..{new_xmax:.6g}, "
                f"y={new_ymin:.6g}..{new_ymax:.6g} | Grid: {ncols} x {nrows}"
            )
        )
        self.after_idle(lambda: self.run_backend(preserve_basin_zoom_history=True))
        return "break"

    def _restore_previous_basin_zoom(self, _event=None):
        if self.run_in_progress:
            self.basin_info.configure(text="A computation is already running.")
            return "break"
        if not self.basin_zoom_history:
            self.basin_info.configure(text="No previous basin zoom extent is available.")
            return "break"

        snapshot = self.basin_zoom_history.pop()
        self._restore_grid_snapshot(snapshot)
        self.basin_info.configure(text="Recomputing previous basin extent...")
        self.after_idle(lambda: self.run_backend(preserve_basin_zoom_history=True))
        return "break"

    def load_config_dialog(self) -> None:
        path = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("TOML", "*.toml"), ("All Files", "*.*")],
            initialdir=BASE_DIR,
        )
        if path:
            self.load_config(Path(path))

    def save_config_dialog(self) -> None:
        try:
            config = self.collect_config()
        except Exception as exc:
            messagebox.showerror("Invalid Configuration", str(exc))
            return

        path = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".toml",
            filetypes=[("TOML", "*.toml"), ("All Files", "*.*")],
            initialdir=BASE_DIR,
        )
        if not path:
            return
        dump_toml(config, Path(path))
        messagebox.showinfo("Saved", f"Configuration saved:\n{path}")

    def load_config(self, path: Path) -> None:
        with path.open("rb") as handle:
            config = tomllib.load(handle)

        analysis = config.get("analysis", {})
        sweep = config.get("sweep", {})
        model = config["model"]
        plane = config["initial_condition_plane"]
        integration = config["integration"]
        classification = config["classification"]
        output = config["output"]

        self.vars["analysis_mode"].set(str(analysis.get("mode", "single_basin")))
        self.vars["sweep_parameter"].set(str(sweep.get("parameter", model["parameter_names"][0] if model["parameter_names"] else "")))
        self.vars["sweep_min_value"].set(str(sweep.get("min_value", 0.0)))
        self.vars["sweep_max_value"].set(str(sweep.get("max_value", 1.0)))
        self.vars["sweep_num_values"].set(str(sweep.get("num_values", 25)))

        self.vars["state_names"].set(", ".join(model["state_names"]))
        self.vars["parameter_names"].set(", ".join(model["parameter_names"]))
        self.vars["parameter_values"].set(", ".join(str(value) for value in model["parameter_values"]))
        self.vars["state_defaults"].set(", ".join(str(value) for value in model["state_defaults"]))
        self._set_equation_values(list(model["equations"]))

        for key in ["x_state", "y_state", "x_min", "x_max", "y_min", "y_max", "grid_mode", "n_target", "nx", "ny"]:
            self.vars[key].set(str(plane[key]))

        for key in [
            "device",
            "precision",
            "solver_mode",
            "solver",
            "period_mode",
            "custom_time_argument",
            "period_expression",
            "transient_periods",
            "evaluation_periods",
            "samples_per_period",
            "t_transient",
            "t_evaluation",
            "dt",
            "save_dt",
        ]:
            self.vars[key].set(str(integration.get(key, "time") if key == "custom_time_argument" else integration[key]))
        self.vars["streaming_chunk_periods"].set(str(integration.get("streaming_chunk_periods", 10.0)))
        for key in ["abstol", "reltol"]:
            self.vars[key].set(str(integration[key]))
        self._sync_solver_setup_from_fields()

        for key in [
            "observable_state",
            "zero_cross_state",
            "fingerprint_tol",
            "fingerprint_k",
            "extrema_eps",
            "max_extrema",
        ]:
            self.vars[key].set(str(classification[key]))

        self.vars["run_name"].set(str(output["run_name"]))
        self.vars["write_labels_csv"].set("true" if output["write_labels_csv"] else "false")
        self.vars["write_basin_image"].set("true" if output["write_basin_image"] else "false")
        self.vars["write_sweep_details"].set("true" if output.get("write_sweep_details", True) else "false")
        self.basin_zoom_history.clear()

        self._append_log(f"Configuration loaded: {path}")

    def collect_config(self) -> dict:
        state_names = parse_csv_strings(self.vars["state_names"].get())
        equations = self._current_equations()
        parameter_names = parse_csv_strings(self.vars["parameter_names"].get())
        parameter_values = parse_csv_numbers(self.vars["parameter_values"].get())
        state_defaults = parse_csv_numbers(self.vars["state_defaults"].get())

        if not state_names:
            raise ValueError("At least one state and one equation must be defined.")
        if any(not equation for equation in equations):
            raise ValueError("Every state needs a non-empty equation.")
        if len(state_names) != len(equations):
            raise ValueError("The number of equations must match the number of states.")
        self._validate_equations_for_gui(raise_on_error=True)
        if len(parameter_names) != len(parameter_values):
            raise ValueError("The number of parameter values must match the number of parameter names.")
        if len(state_defaults) != len(state_names):
            raise ValueError("The number of default initial values must match the number of states.")

        named_states = {
            self.vars["x_state"].get().strip(),
            self.vars["y_state"].get().strip(),
            self.vars["observable_state"].get().strip(),
            self.vars["zero_cross_state"].get().strip(),
        }
        missing = sorted(name for name in named_states if name and name not in state_names)
        if missing:
            raise ValueError(f"These state names are not defined: {', '.join(missing)}")

        analysis_mode = self.vars["analysis_mode"].get().strip()
        if analysis_mode not in {"single_basin", "parameter_sweep", "inspection"}:
            raise ValueError("Analysis mode must be single_basin, parameter_sweep, or inspection.")

        self._on_solver_setup_changed(sync_time_argument=False)
        device = self.vars["device"].get().strip()
        precision = self.vars["precision"].get().strip()
        solver_mode = self.vars["solver_mode"].get().strip()
        solver = self.vars["solver"].get().strip()
        custom_time_argument = self.vars["custom_time_argument"].get().strip()
        if device not in {"gpu", "cpu"}:
            raise ValueError("Device must be gpu or cpu.")
        if precision not in {"Float32", "Float64"}:
            raise ValueError("Precision must be Float32 or Float64.")
        valid_solver_modes = {"fixed", "streaming_extrema", "rk4_extrema", "rk4_full_extrema", "rk4_full_extrema_custom", "adaptive"}
        if solver_mode not in valid_solver_modes:
            raise ValueError("Solver mode must be fixed, streaming_extrema, rk4_extrema, rk4_full_extrema, rk4_full_extrema_custom, or adaptive.")
        if solver not in {"Tsit5", "Vern9", "Rosenbrock23", "Rodas5P"}:
            raise ValueError("Solver must be Tsit5, Vern9, Rosenbrock23, or Rodas5P.")
        if custom_time_argument not in {"time", "phase"}:
            raise ValueError("Custom Time Arg must be time or phase.")
        if device == "gpu" and solver_mode == "adaptive":
            raise ValueError("GPU supports fixed, streaming_extrema, rk4_extrema, rk4_full_extrema, and rk4_full_extrema_custom setups only.")
        if device == "gpu" and solver != "Tsit5":
            raise ValueError("GPU supports Tsit5 only.")
        if solver_mode in {"rk4_extrema", "rk4_full_extrema", "rk4_full_extrema_custom"}:
            if device != "gpu" or solver != "Tsit5":
                raise ValueError("RK4 extrema solver setups require GPU and Tsit5.")
            if solver_mode in {"rk4_extrema", "rk4_full_extrema"} and precision != "Float32":
                raise ValueError("The Stage-B RK4 and optimized built-in A+B RK4 setups currently require Float32.")
            if solver_mode == "rk4_full_extrema_custom" and precision not in {"Float32", "Float64"}:
                raise ValueError("The custom RK4 A+B setup requires Float32 or Float64.")
            if not 2 <= len(state_names) <= 8:
                raise ValueError("GPU fused RK4 extrema currently supports custom ODEs with 2 to 8 states.")
            is_default_duffing = matches_default_duffing_config(state_names, parameter_names, equations)
            if solver_mode == "rk4_full_extrema" and not is_default_duffing:
                raise ValueError("GPU RK4 Stage A+B is the optimized built-in Duffing path. Select 'GPU RK4 extrema Stage A+B Custom - Float32' for custom models.")
        uses_phase_symbol = any(re.search(r"(?<![\d.])\bphase\b", equation) for equation in equations)
        if uses_phase_symbol and custom_time_argument != "phase":
            raise ValueError("Equations using 'phase' require Custom Time Arg = phase.")
        if custom_time_argument == "phase":
            if solver_mode != "rk4_full_extrema_custom":
                raise ValueError("Custom Time Arg = phase currently requires 'GPU RK4 extrema Stage A+B Custom - Float32'.")
            if self.vars["period_mode"].get().strip() != "periodic":
                raise ValueError("Custom Time Arg = phase requires Time Mode = periodic.")

        sweep_parameter = self.vars["sweep_parameter"].get().strip()
        if analysis_mode == "parameter_sweep" and sweep_parameter not in parameter_names:
            raise ValueError(f"Sweep parameter must be one of: {', '.join(parameter_names)}")

        sweep_num_values = int(float(self.vars["sweep_num_values"].get()))
        if sweep_num_values < 2:
            raise ValueError("Sweep values must be at least 2.")

        config = {
            "analysis": {
                "mode": analysis_mode,
            },
            "sweep": {
                "parameter": sweep_parameter,
                "min_value": float(self.vars["sweep_min_value"].get()),
                "max_value": float(self.vars["sweep_max_value"].get()),
                "num_values": sweep_num_values,
            },
            "model": {
                "state_names": state_names,
                "equations": equations,
                "parameter_names": parameter_names,
                "parameter_values": parameter_values,
                "state_defaults": state_defaults,
            },
            "initial_condition_plane": {
                "x_state": self.vars["x_state"].get().strip(),
                "y_state": self.vars["y_state"].get().strip(),
                "x_min": float(self.vars["x_min"].get()),
                "x_max": float(self.vars["x_max"].get()),
                "y_min": float(self.vars["y_min"].get()),
                "y_max": float(self.vars["y_max"].get()),
                "grid_mode": self.vars["grid_mode"].get().strip(),
                "n_target": int(float(self.vars["n_target"].get())),
                "nx": int(float(self.vars["nx"].get())),
                "ny": int(float(self.vars["ny"].get())),
            },
            "integration": {
                "device": device,
                "precision": precision,
                "solver_mode": solver_mode,
                "solver": solver,
                "period_mode": self.vars["period_mode"].get().strip(),
                "custom_time_argument": custom_time_argument,
                "period_expression": self.vars["period_expression"].get().strip(),
                "transient_periods": float(self.vars["transient_periods"].get()),
                "evaluation_periods": float(self.vars["evaluation_periods"].get()),
                "samples_per_period": int(float(self.vars["samples_per_period"].get())),
                "t_transient": float(self.vars["t_transient"].get()),
                "t_evaluation": float(self.vars["t_evaluation"].get()),
                "dt": float(self.vars["dt"].get()),
                "save_dt": float(self.vars["save_dt"].get()),
                "streaming_chunk_periods": float(self.vars["streaming_chunk_periods"].get()),
                "abstol": float(self.vars["abstol"].get()),
                "reltol": float(self.vars["reltol"].get()),
            },
            "classification": {
                "observable_state": self.vars["observable_state"].get().strip(),
                "zero_cross_state": self.vars["zero_cross_state"].get().strip(),
                "fingerprint_tol": float(self.vars["fingerprint_tol"].get()),
                "fingerprint_k": int(float(self.vars["fingerprint_k"].get())),
                "extrema_eps": float(self.vars["extrema_eps"].get()),
                "max_extrema": int(float(self.vars["max_extrema"].get())),
            },
            "output": {
                "run_name": self.vars["run_name"].get().strip() or "run",
                "write_labels_csv": self.vars["write_labels_csv"].get().strip().lower() == "true",
                "write_basin_image": self.vars["write_basin_image"].get().strip().lower() == "true",
                "write_sweep_details": self.vars["write_sweep_details"].get().strip().lower() == "true",
            },
        }
        return config

    def _ensure_backend_server(self) -> bool:
        if self.backend_process is not None and self.backend_process.poll() is None:
            return True

        if not self._ensure_julia_environment():
            return False

        cmd = ["julia", f"--project={BASE_DIR}", str(BACKEND_PATH), "--server"]
        self._append_log("Starting persistent Julia backend:")
        self._append_log(" ".join(cmd))

        try:
            self.backend_process = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except Exception as exc:
            messagebox.showerror("Start Failed", str(exc))
            self.backend_process = None
            return False

        thread = threading.Thread(target=self._backend_server_reader_thread, args=(self.backend_process,), daemon=True)
        thread.start()
        return True

    def _ensure_julia_environment(self) -> bool:
        if self.julia_environment_ready:
            return True

        cmd = [
            "julia",
            f"--project={BASE_DIR}",
            "--startup-file=no",
            "--compile=min",
            "-e",
            "using Pkg; Pkg.instantiate()",
        ]
        self._append_log("Checking Julia package environment:")
        self._append_log(" ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except Exception as exc:
            messagebox.showerror("Julia Setup Failed", str(exc))
            return False

        output = result.stdout.strip()
        if output:
            for line in output.splitlines():
                self._append_log(self._normalize_backend_log_line(line))

        if result.returncode != 0:
            messagebox.showerror(
                "Julia Setup Failed",
                "Julia could not instantiate the local project environment. See the log for details.",
            )
            return False

        self.julia_environment_ready = True
        self._append_log("Julia package environment is ready.")
        return True

    def _submit_backend_job(self, config_path: Path, intro: str) -> bool:
        if not self._ensure_backend_server() or self.backend_process is None or self.backend_process.stdin is None:
            return False

        self._append_log(intro)
        self._append_log(str(config_path))
        try:
            self.backend_process.stdin.write(str(config_path) + "\n")
            self.backend_process.stdin.flush()
        except Exception as exc:
            messagebox.showerror("Backend Failed", f"The persistent Julia backend could not receive the job:\n{exc}")
            self._terminate_backend_server(force=True)
            return False

        self.process = self.backend_process
        self.run_in_progress = True
        self.run_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        return True

    def run_backend(self, *, preserve_basin_zoom_history: bool = False) -> None:
        if self.run_in_progress:
            return

        try:
            config = self.collect_config()
        except Exception as exc:
            messagebox.showerror("Invalid Configuration", str(exc))
            return

        if not preserve_basin_zoom_history:
            self.basin_zoom_history.clear()

        if config["analysis"]["mode"] == "inspection":
            self._enter_inspection_mode()
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        config_path = WORK_DIR / f"current_config_{timestamp}.toml"
        dump_toml(config, config_path)

        self.log_text.delete("1.0", "end")
        self.summary_text.delete("1.0", "end")
        self.basin_canvas.delete("all")
        self.phase_canvas.delete("all")
        self.sweep_canvas.delete("all")
        self.basin_info.configure(text="Computation running...")
        self.phase_info.configure(text="Computation running...")
        self.sweep_info.configure(text="Computation running...")
        self.current_image = None
        self.basin_rows = []
        self.current_label_feature_colors = {}
        self.phase_series = []
        self.phase_bounds = None
        self.sweep_points = []
        self.sweep_points_total = 0
        self.sweep_bounds = None
        self.sweep_feature_colors_by_value = {}
        self.sweep_details = []
        self.selected_sweep_value = None
        self.probe_selected = []
        self.probe_benchmark = []
        self.probe_phase_start = None
        self.probe_phase_end = None
        self._clear_plot_zoom()
        self.current_summary_path = None
        self.current_result_dir = None
        self.current_summary_data = None
        self.current_result_mode = ""
        self.result_plane_bounds = None
        self.basin_bounds = None

        self._submit_backend_job(config_path, "Sending run to persistent Julia backend:")

    def cancel_run(self) -> None:
        if self.run_in_progress:
            self._append_log("Cancellation requested. Restarting the persistent Julia backend to stop the current job.")
            self._terminate_backend_server(force=True)
            self.run_in_progress = False
            self.run_button.configure(state="normal")
            self.cancel_button.configure(state="disabled")
            self.process = None

    def _kill_if_still_running(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is None:
            process.kill()
            self._append_log("Julia process was killed because it was still running after cancellation.")

    def restart_backend(self) -> None:
        if self.run_in_progress:
            self.cancel_run()
            return
        self._terminate_backend_server(force=False)
        self._append_log("Persistent Julia backend will be restarted on the next run.")

    def _terminate_backend_server(self, *, force: bool) -> None:
        process = self.backend_process
        self.backend_process = None
        if process is None or process.poll() is not None:
            return

        if not force and process.stdin is not None:
            try:
                process.stdin.write("__quit__\n")
                process.stdin.flush()
            except Exception:
                force = True

        if force:
            process.terminate()
            self.after(3000, lambda: self._kill_if_still_running(process))

    def _backend_server_reader_thread(self, process: subprocess.Popen[str]) -> None:
        for line in process.stdout or []:
            self.log_queue.put(line.rstrip("\n"))
        return_code = process.wait()
        self.log_queue.put(f"__BACKEND_EXITED__:{process.pid}:{return_code}")

    def _reader_thread(self) -> None:
        assert self.process is not None
        for line in self.process.stdout or []:
            self.log_queue.put(line.rstrip("\n"))
        return_code = self.process.wait()
        self.log_queue.put(f"__PROCESS_DONE__:{return_code}")

    @staticmethod
    def _normalize_backend_log_line(line: str) -> str:
        text = line.rstrip("\r")
        if "â" in text or "Ã" in text:
            try:
                text = text.encode("cp1252").decode("utf-8")
            except UnicodeError:
                pass
        return text.translate(JULIA_LOG_FRAME_TRANSLATION)

    def _hide_cuda_version_notice(self, line: str) -> bool:
        text = line.lstrip("+| ").strip()
        if "You are using CUDA " in text and "CUDA.jl was precompiled for CUDA " in text:
            self._cuda_version_notice_filter_remaining = 2
            if not self._cuda_version_notice_suppressed:
                self._append_log(
                    "CUDA.jl version notice hidden: runtime CUDA differs from the version used during precompilation; GPU runs continue."
                )
                self._cuda_version_notice_suppressed = True
            return True

        if self._cuda_version_notice_filter_remaining > 0:
            if "This is unexpected; please file an issue." in text or (
                "@ CUDA " in text and "initialization.jl:148" in text
            ):
                self._cuda_version_notice_filter_remaining -= 1
                return True
            self._cuda_version_notice_filter_remaining = 0

        return False

    def _poll_log_queue(self) -> None:
        try:
            try:
                while True:
                    line = self._normalize_backend_log_line(self.log_queue.get_nowait())
                    try:
                        if line.startswith("RESULT_SUMMARY="):
                            self.current_summary_path = Path(line.split("=", 1)[1].strip())
                        elif line.startswith("RESULT_DIR="):
                            self.current_result_dir = Path(line.split("=", 1)[1].strip())
                        elif line == "__BACKEND_READY__":
                            self._append_log("Persistent Julia backend is ready.")
                        elif line.startswith("__JOB_STARTED__="):
                            self._append_log("Julia job started.")
                        elif line.startswith("__JOB_DONE__="):
                            self._finish_run(int(line.split("=", 1)[1]))
                        elif line.startswith("__BACKEND_EXITED__:"):
                            self._handle_backend_exit(line)
                        elif line == "__BACKEND_EXITING__":
                            self._append_log("Persistent Julia backend is shutting down.")
                        elif line.startswith("__PROCESS_DONE__:"):
                            self._finish_run(int(line.split(":", 1)[1]))
                        elif self._hide_cuda_version_notice(line):
                            pass
                        else:
                            self._append_log(line)
                    except Exception as exc:
                        self._handle_gui_error("Error while processing backend output", exc)
            except queue.Empty:
                pass
        finally:
            self.after(120, self._poll_log_queue)

    def _handle_backend_exit(self, line: str) -> None:
        _marker, pid_text, code_text = line.split(":", 2)
        return_code = int(code_text)
        pid = int(pid_text)
        if self.backend_process is not None and self.backend_process.pid == pid:
            self.backend_process = None
        if self.run_in_progress:
            self._finish_run(return_code if return_code != 0 else 1)
        elif return_code != 0:
            self._append_log(f"Persistent Julia backend exited with code {return_code}.")

    def _handle_gui_error(self, title: str, exc: Exception) -> None:
        self.run_in_progress = False
        self.run_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self.process = None
        self.image_info.configure(text=f"{title}: {exc}")
        self._append_log(title)
        self._append_log(str(exc))
        self._append_log(traceback.format_exc())

    def _finish_run(self, return_code: int) -> None:
        self.run_in_progress = False
        self.run_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        self.process = None

        if return_code != 0:
            self.image_info.configure(text="Computation failed. See the log for details.")
            messagebox.showerror("Computation Failed", "The Julia backend exited with an error.")
            return

        if not self.current_summary_path or not self.current_summary_path.exists():
            self.image_info.configure(text="Computation finished, but no summary file was found.")
            messagebox.showwarning("Incomplete Result", "No result summary was found.")
            return

        try:
            self.load_results(self.current_summary_path)
            self._append_log("Computation completed successfully.")
        except Exception as exc:
            self._handle_gui_error("Results were created but could not be displayed", exc)

    def load_results(self, summary_path: Path) -> None:
        with summary_path.open("rb") as handle:
            summary = tomllib.load(handle)

        self.current_summary_path = summary_path
        self.current_result_dir = Path(summary["run"]["run_dir"])
        self.current_summary_data = summary
        self._clear_plot_zoom()

        mode = summary.get("analysis", {}).get("mode", "single_basin")
        self.current_result_mode = str(mode)
        self.result_plane_bounds = self._summary_plane_bounds(summary)
        self.basin_bounds = self.result_plane_bounds
        paths = summary.get("paths", {})
        basin_image_raw = paths.get("basin_image", "")
        labels_csv_raw = paths.get("labels_csv", "")
        phase_samples_raw = paths.get("phase_samples_csv", "")
        sweep_extrema_raw = paths.get("sweep_extrema_csv", "")
        sweep_details_raw = paths.get("sweep_details_csv", "")
        trajectory_selected_raw = paths.get("trajectory_selected_csv", "")
        trajectory_benchmark_raw = paths.get("trajectory_benchmark_csv", "")
        basin_image = Path(basin_image_raw) if basin_image_raw else None
        labels_csv = Path(labels_csv_raw) if labels_csv_raw else None
        phase_samples_csv = Path(phase_samples_raw) if phase_samples_raw else None
        sweep_extrema_csv = Path(sweep_extrema_raw) if sweep_extrema_raw else None
        sweep_details_csv = Path(sweep_details_raw) if sweep_details_raw else None
        trajectory_selected_csv = Path(trajectory_selected_raw) if trajectory_selected_raw else None
        trajectory_benchmark_csv = Path(trajectory_benchmark_raw) if trajectory_benchmark_raw else None
        class_stats_raw = paths.get("class_stats_csv", "")
        class_stats_path = Path(class_stats_raw) if class_stats_raw else None
        self.current_label_feature_colors = {}
        self.class_fractions = {}
        self.sweep_class_fractions_by_value = {}

        lines = [f"Run directory: {summary['run']['run_dir']}", f"Analysis mode: {mode}"]
        if mode == "parameter_sweep":
            sweep_meta = summary.get("sweep", {})
            results = summary.get("results", {})
            lines.extend(
                [
                    f"Sweep parameter: {sweep_meta.get('parameter', '')}",
                    f"Sweep range: {sweep_meta.get('min_value', '')} to {sweep_meta.get('max_value', '')}",
                    f"Sweep values: {sweep_meta.get('num_values', results.get('num_values', ''))}",
                    f"Total trajectories: {results.get('n_trajectories_total', '')}",
                    f"Extrema points: {results.get('num_extrema', '')}",
                    f"Runtime [s]: {summary['run']['elapsed_seconds']}",
                ]
            )
        elif mode == "point_probe":
            probe_meta = summary.get("probe", {})
            integration_meta = summary.get("integration", {})
            effective_device = integration_meta.get("effective_device", integration_meta.get("device", ""))
            lines.extend(
                [
                    f"Initial point: {probe_meta.get('x_state', '')}={probe_meta.get('x_value', '')}, {probe_meta.get('y_state', '')}={probe_meta.get('y_value', '')}",
                    f"Selected method: {effective_device} / {integration_meta.get('solver_mode', '')} / {integration_meta.get('solver', '')} / {integration_meta.get('precision', '')}",
                    f"Benchmark: CPU / adaptive / Vern9 / Float64",
                    f"Runtime [s]: {summary['run']['elapsed_seconds']}",
                ]
            )
            if effective_device != integration_meta.get("device", ""):
                lines.append(f"Requested device: {integration_meta.get('device', '')}; effective device: {effective_device}")
        else:
            lines.extend(
                [
                    f"Number of classes: {summary['results']['num_classes']}",
                    f"Grid: nx={summary['results']['nx']}, ny={summary['results']['ny']}",
                    f"Total points: {summary['results']['n_trajectories']}",
                    f"Solver: {summary['integration']['device']} / {summary['integration']['solver_mode']} / {summary['integration']['solver']}",
                    f"Precision: {summary['integration']['precision']}",
                    f"dt: {summary['integration']['dt']}",
                    f"save_dt: {summary['integration']['save_dt']}",
                    f"Transient time: {summary['integration']['t_transient']}",
                    f"Evaluation time: {summary['integration']['t_evaluation']}",
                    f"Runtime [s]: {summary['run']['elapsed_seconds']}",
                    "",
                    "Class statistics:",
                ]
            )

        if mode == "parameter_sweep" and class_stats_path and class_stats_path.exists():
            lines.extend(["", f"Class statistics CSV: {class_stats_path}"])
        elif class_stats_path and class_stats_path.exists():
            with class_stats_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    lines.append(
                        f"Label {row['label']}: count={row['count']}, fraction={row['fraction']}"
                    )

        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", "\n".join(lines))

        if mode == "point_probe":
            if not trajectory_selected_csv or not trajectory_benchmark_csv:
                self.probe_time_info.configure(text="Point probe result is missing trajectory paths.")
                return
            self.plot_notebook.select(2)
            self._load_point_probe_result(summary, trajectory_selected_csv, trajectory_benchmark_csv)
            return

        if mode == "parameter_sweep":
            self.plot_notebook.select(1)
            self.basin_canvas.delete("all")
            self.phase_canvas.delete("all")
            self.basin_rows = []
            self.basin_bounds = self.result_plane_bounds
            self.phase_series = []
            self.phase_bounds = None
            self.selected_sweep_value = None
            self.current_label_feature_colors = {}
            self.sweep_feature_colors_by_value = {}
            self.sweep_class_fractions_by_value = self._load_sweep_class_fractions(class_stats_path) if class_stats_path and class_stats_path.is_file() else {}
            self.basin_info.configure(text="Click the sweep plot to load the saved basin for that parameter value.")
            self.phase_info.configure(text="Click the sweep plot to load representative phase trajectories.")
            self.sweep_details = self._load_sweep_details(sweep_details_csv) if sweep_details_csv and sweep_details_csv.is_file() else []
            if sweep_extrema_csv and sweep_extrema_csv.is_file():
                self.sweep_points, self.sweep_bounds = self._load_sweep_points(sweep_extrema_csv)
                detail_text = f" | Drill-down values: {len(self.sweep_details)}" if self.sweep_details else " | No drill-down details saved"
                if self.sweep_points_total > len(self.sweep_points):
                    point_text = f"Displayed points: {len(self.sweep_points)} of {self.sweep_points_total}"
                else:
                    point_text = f"Points: {len(self.sweep_points)}"
                self.sweep_info.configure(
                    text=f"Source: {sweep_extrema_csv} | {point_text}{detail_text}"
                )
                self._redraw_sweep_canvas()
            else:
                self.sweep_points = []
                self.sweep_points_total = 0
                self.sweep_bounds = None
                self.sweep_canvas.delete("all")
                self.sweep_info.configure(text="No sweep extrema data available.")
            return

        self.plot_notebook.select(0)
        self.sweep_points = []
        self.sweep_points_total = 0
        self.sweep_bounds = None
        self.sweep_feature_colors_by_value = {}
        self.sweep_details = []
        self.selected_sweep_value = None
        self.sweep_canvas.delete("all")
        self.sweep_info.configure(text="No sweep data loaded for this single-basin result.")
        if class_stats_path and class_stats_path.is_file():
            self.current_label_feature_colors = self._load_class_feature_colors(class_stats_path)
            self.class_fractions = self._load_class_fractions(class_stats_path)

        if labels_csv and labels_csv.is_file():
            self.basin_rows = self._load_label_rows(labels_csv)
            self.basin_bounds = self.result_plane_bounds
            self.basin_info.configure(
                text=(
                    f"Source: {labels_csv} | Grid: {summary['results']['nx']} x {summary['results']['ny']} | "
                    "Drag rectangle to recompute zoom; right-click goes back"
                )
            )
            self._redraw_basin_canvas()
        elif basin_image and basin_image.is_file():
            self.basin_rows = []
            self.basin_bounds = self.result_plane_bounds
            self.basin_info.configure(text=f"Only an image file is available: {basin_image}")
            self._draw_image_fallback(basin_image)
        else:
            self.basin_canvas.delete("all")
            self.basin_rows = []
            self.basin_bounds = None
            self.current_image = None
            self.basin_info.configure(text="No basin data available.")

        if phase_samples_csv and phase_samples_csv.is_file():
            self.phase_series, self.phase_bounds = self._load_phase_series(phase_samples_csv)
            self._annotate_phase_series(self.phase_series, self.class_fractions)
            phase_meta = summary.get("phase", {})
            x_state = phase_meta.get("x_state", "x")
            y_state = phase_meta.get("y_state", "y")
            shown = len(self._phase_series_for_display())
            self.phase_info.configure(
                text=f"Source: {phase_samples_csv} | Axes: {x_state} / {y_state} | Trajectories: {shown}/{len(self.phase_series)}"
            )
            self._redraw_phase_canvas()
        else:
            self.phase_series = []
            self.phase_bounds = None
            self.phase_canvas.delete("all")
            self._update_phase_filter_status(0, 0)
            self.phase_info.configure(text="No phase plot available. Recompute the result with the current backend version.")

    def load_result_dialog(self) -> None:
        runs_dir = BASE_DIR / "runs"
        initial_dir = runs_dir if runs_dir.exists() else BASE_DIR
        path_raw = filedialog.askopenfilename(
            title="Load Result Summary",
            initialdir=str(initial_dir),
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
        )
        if not path_raw:
            return
        try:
            self.basin_zoom_history.clear()
            summary_path = Path(path_raw)
            self.load_results(summary_path)
            self._append_log(f"Result loaded: {summary_path}")
        except Exception as exc:
            self._handle_gui_error("Result could not be loaded", exc)

    def load_latest_result(self) -> None:
        summaries = sorted(
            (BASE_DIR / "runs").glob("*/summary.toml"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not summaries:
            messagebox.showinfo("No Result", "No result folder was found yet.")
            return
        try:
            self.basin_zoom_history.clear()
            self.load_results(summaries[0])
            self._append_log(f"Latest result loaded: {summaries[0]}")
        except Exception as exc:
            self._handle_gui_error("Latest result could not be loaded", exc)

    def _load_label_rows(self, labels_csv: Path) -> list[list[int]]:
        rows: list[list[int]] = []
        with labels_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                values = [int(float(token)) for token in row if token.strip()]
                if values:
                    rows.append(values)

        if not rows:
            raise ValueError(f"Label file is empty: {labels_csv}")
        return rows

    def _load_class_feature_colors(self, class_stats_csv: Path) -> dict[int, str]:
        colors: dict[int, str] = {}
        with class_stats_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            if not {"label", "color_r", "color_g", "color_b"}.issubset(fieldnames):
                return colors
            for row in reader:
                try:
                    label = int(float(row["label"]))
                    red = int(float(row["color_r"]))
                    green = int(float(row["color_g"]))
                    blue = int(float(row["color_b"]))
                except (TypeError, ValueError):
                    continue
                if all(0 <= value <= 255 for value in (red, green, blue)):
                    colors[label] = f"#{red:02x}{green:02x}{blue:02x}"
        return colors

    def _load_class_fractions(self, class_stats_csv: Path) -> dict[int, float]:
        fractions: dict[int, float] = {}
        with class_stats_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            if not {"label", "fraction"}.issubset(fieldnames) or "parameter_value" in fieldnames:
                return fractions
            for row in reader:
                try:
                    fractions[int(float(row["label"]))] = float(row["fraction"])
                except (TypeError, ValueError):
                    continue
        return fractions

    def _load_sweep_class_fractions(self, class_stats_csv: Path) -> dict[float, dict[int, float]]:
        fractions_by_value: dict[float, dict[int, float]] = {}
        with class_stats_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            if not {"parameter_value", "label", "fraction"}.issubset(fieldnames):
                return fractions_by_value
            for row in reader:
                try:
                    parameter_value = float(row["parameter_value"])
                    label = int(float(row["label"]))
                    fraction = float(row["fraction"])
                except (TypeError, ValueError):
                    continue
                fractions_by_value.setdefault(parameter_value, {})[label] = fraction
        return fractions_by_value

    def _class_fractions_for_sweep_value(self, parameter_value: float) -> dict[int, float]:
        if not self.sweep_class_fractions_by_value:
            return {}
        nearest_value = min(self.sweep_class_fractions_by_value, key=lambda value: abs(value - parameter_value))
        return dict(self.sweep_class_fractions_by_value.get(nearest_value, {}))

    @staticmethod
    def _class_fractions_for_sweep_value_map(fractions_by_value: dict[float, dict[int, float]], parameter_value: float) -> dict[int, float]:
        if not fractions_by_value:
            return {}
        nearest_value = min(fractions_by_value, key=lambda value: abs(value - parameter_value))
        return dict(fractions_by_value.get(nearest_value, {}))

    @staticmethod
    def _annotate_phase_series(series: list[dict], class_fractions: dict[int, float]) -> None:
        for item in series:
            label = int(item.get("label", 0))
            item["class_fraction"] = class_fractions.get(label, item.get("class_fraction", 1.0))

    def _phase_min_fraction(self) -> float:
        try:
            return max(0.0, float(self.phase_min_fraction_var.get().strip() or PHASE_MIN_FRACTION_DEFAULT))
        except ValueError:
            return float(PHASE_MIN_FRACTION_DEFAULT)

    def _phase_series_for_display(self, series: list[dict] | None = None, *, min_fraction: float | None = None) -> list[dict]:
        source = self.phase_series if series is None else series
        threshold = self._phase_min_fraction() if min_fraction is None else max(0.0, float(min_fraction))
        return [item for item in source if float(item.get("class_fraction", 1.0)) >= threshold]

    @staticmethod
    def _phase_bounds_for_series(series: list[dict]) -> tuple[float, float, float, float] | None:
        xmin = ymin = float("inf")
        xmax = ymax = float("-inf")
        for item in series:
            for x, y in item.get("points", []):
                xmin = min(xmin, x)
                xmax = max(xmax, x)
                ymin = min(ymin, y)
                ymax = max(ymax, y)
        if not math.isfinite(xmin) or not math.isfinite(ymin):
            return None
        if xmin == xmax:
            xmin -= 1.0
            xmax += 1.0
        if ymin == ymax:
            ymin -= 1.0
            ymax += 1.0
        return xmin, xmax, ymin, ymax

    def _update_phase_filter_status(self, displayed: int, total: int) -> None:
        if self.phase_filter_status_label is not None:
            self.phase_filter_status_label.configure(text=f"shown {displayed}/{total} | min frac {self._phase_min_fraction():g}")

    def _load_phase_series(self, phase_samples_csv: Path) -> tuple[list[dict], tuple[float, float, float, float] | None]:
        if phase_samples_csv.suffix.lower() == ".bin":
            return self._load_phase_series_binary(phase_samples_csv)

        grouped: dict[int, dict] = {}
        xmin = ymin = float("inf")
        xmax = ymax = float("-inf")

        with phase_samples_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                sample_id = int(row["sample_id"])
                label = int(row["label"])
                x = float(row["x"])
                y = float(row["y"])
                item = grouped.setdefault(sample_id, {"label": label, "points": []})
                item["points"].append((x, y))
                xmin = min(xmin, x)
                xmax = max(xmax, x)
                ymin = min(ymin, y)
                ymax = max(ymax, y)

        series = [item for _sample_id, item in sorted(grouped.items()) if item["points"]]
        if not series:
            return [], None

        if xmin == xmax:
            xmin -= 1.0
            xmax += 1.0
        if ymin == ymax:
            ymin -= 1.0
            ymax += 1.0
        return series, (xmin, xmax, ymin, ymax)

    def _load_phase_series_binary(self, phase_samples_bin: Path) -> tuple[list[dict], tuple[float, float, float, float] | None]:
        series: list[dict] = []
        xmin = ymin = float("inf")
        xmax = ymax = float("-inf")

        with phase_samples_bin.open("rb") as handle:
            magic = handle.read(len(PHASE_BINARY_MAGIC))
            if magic != PHASE_BINARY_MAGIC:
                raise ValueError(f"Unsupported phase-sample binary format: {phase_samples_bin}")

            sample_count_raw = handle.read(4)
            if len(sample_count_raw) != 4:
                raise ValueError(f"Truncated phase-sample binary header: {phase_samples_bin}")
            sample_count = struct.unpack("<I", sample_count_raw)[0]

            for _ in range(sample_count):
                header = handle.read(12)
                if len(header) != 12:
                    raise ValueError(f"Truncated phase-sample record header: {phase_samples_bin}")
                _sample_id, label, n_points = struct.unpack("<iiI", header)
                payload = handle.read(n_points * 8)
                if len(payload) != n_points * 8:
                    raise ValueError(f"Truncated phase-sample point payload: {phase_samples_bin}")

                points: list[tuple[float, float]] = []
                for x, y in struct.iter_unpack("<ff", payload):
                    points.append((x, y))
                    xmin = min(xmin, x)
                    xmax = max(xmax, x)
                    ymin = min(ymin, y)
                    ymax = max(ymax, y)
                if points:
                    series.append({"label": label, "points": points})

        if not series:
            return [], None

        if xmin == xmax:
            xmin -= 1.0
            xmax += 1.0
        if ymin == ymax:
            ymin -= 1.0
            ymax += 1.0
        return series, (xmin, xmax, ymin, ymax)

    def _load_sweep_details(self, sweep_details_csv: Path) -> list[dict]:
        details: list[dict] = []
        with sweep_details_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                labels_path = Path(row["labels_csv"])
                phase_path = Path(row["phase_samples_csv"])
                if not labels_path.is_file() or not phase_path.is_file():
                    continue
                details.append(
                    {
                        "index": int(row["index"]),
                        "parameter_value": float(row["parameter_value"]),
                        "labels_csv": labels_path,
                        "phase_samples_csv": phase_path,
                        "phase_samples_format": row.get("phase_samples_format", "csv"),
                        "num_classes": int(row.get("num_classes", 0)),
                        "n_phase_samples": int(row.get("n_phase_samples", 0)),
                    }
                )
        return sorted(details, key=lambda item: item["parameter_value"])

    def _load_sweep_points(self, sweep_extrema_csv: Path) -> tuple[list[dict], tuple[float, float, float, float] | None]:
        value_counts: dict[str, int] = {}
        value_numbers: dict[str, float] = {}
        total_points = 0
        xmin = ymin = float("inf")
        xmax = ymax = float("-inf")
        self.sweep_feature_colors_by_value = {}

        with sweep_extrema_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                parameter_key = row["parameter_value"]
                parameter_value = float(parameter_key)
                x_extreme = float(row["x_extreme"])
                value_counts[parameter_key] = value_counts.get(parameter_key, 0) + 1
                value_numbers[parameter_key] = parameter_value
                total_points += 1
                xmin = min(xmin, parameter_value)
                xmax = max(xmax, parameter_value)
                ymin = min(ymin, x_extreme)
                ymax = max(ymax, x_extreme)

        if total_points == 0:
            self.sweep_points_total = 0
            return [], None

        if total_points <= MAX_SWEEP_POINTS_IN_MEMORY:
            per_value_limit = max(value_counts.values())
        else:
            per_value_limit = max(1, MAX_SWEEP_POINTS_IN_MEMORY // max(1, len(value_counts)))
        rng = random.Random(24681357)
        buckets: dict[str, list[dict]] = {key: [] for key in value_counts}
        seen_per_value: dict[str, int] = {key: 0 for key in value_counts}

        with sweep_extrema_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                parameter_key = row["parameter_value"]
                parameter_value = value_numbers[parameter_key]
                x_extreme = float(row["x_extreme"])
                label = int(row.get("label", 0))
                label_color = self._label_color(label)
                feature_color = label_color
                try:
                    red = int(float(row.get("color_r", "")))
                    green = int(float(row.get("color_g", "")))
                    blue = int(float(row.get("color_b", "")))
                    if all(0 <= value <= 255 for value in (red, green, blue)):
                        feature_color = f"#{red:02x}{green:02x}{blue:02x}"
                except (TypeError, ValueError):
                    pass
                try:
                    class_count = int(float(row.get("class_count", "1")))
                except ValueError:
                    class_count = 1
                try:
                    class_fraction = float(row.get("class_fraction", "1"))
                except ValueError:
                    class_fraction = 1.0
                try:
                    class_rank = int(float(row.get("class_rank", str(max(label, 1)))))
                except ValueError:
                    class_rank = 1
                if "class_rank" not in row:
                    class_rank = 1
                self.sweep_feature_colors_by_value.setdefault(parameter_value, {}).setdefault(label, feature_color)

                point = {
                    "parameter_value": parameter_value,
                    "x_extreme": x_extreme,
                    "label": label,
                    "label_color": label_color,
                    "feature_color": feature_color,
                    "class_count": class_count,
                    "class_fraction": class_fraction,
                    "class_rank": class_rank,
                }
                seen_per_value[parameter_key] += 1
                seen_count = seen_per_value[parameter_key]
                bucket = buckets[parameter_key]
                if len(bucket) < per_value_limit:
                    bucket.append(point)
                else:
                    replace_idx = rng.randrange(seen_count)
                    if replace_idx < per_value_limit:
                        bucket[replace_idx] = point

        points: list[dict] = []
        for parameter_key in sorted(value_counts, key=lambda key: value_numbers[key]):
            points.extend(buckets[parameter_key])

        self.sweep_points_total = total_points
        if not points:
            return [], None
        if xmin == xmax:
            xmin -= 1.0
            xmax += 1.0
        if ymin == ymax:
            ymin -= 1.0
            ymax += 1.0
        return points, (xmin, xmax, ymin, ymax)

    def _load_trajectory_csv(self, path: Path) -> tuple[list[str], list[dict]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            state_names = [name for name in (reader.fieldnames or []) if name != "t"]
            rows = []
            for row in reader:
                rows.append(
                    {
                        "t": float(row["t"]),
                        "values": [float(row[name]) for name in state_names],
                    }
                )
        return state_names, rows

    def _load_point_probe_result(self, summary: dict, selected_csv: Path, benchmark_csv: Path) -> None:
        self._clear_plot_zoom("probe_time", "probe_phase")
        self.probe_state_names, self.probe_selected = self._load_trajectory_csv(selected_csv)
        benchmark_names, self.probe_benchmark = self._load_trajectory_csv(benchmark_csv)
        if benchmark_names != self.probe_state_names:
            raise ValueError("Selected and benchmark trajectories use different state columns.")

        probe_meta = summary.get("probe", {})
        integration_meta = summary.get("integration", {})
        self.probe_x_state = str(probe_meta.get("x_state", self.probe_state_names[0] if self.probe_state_names else ""))
        self.probe_y_state = str(probe_meta.get("y_state", self.probe_state_names[1] if len(self.probe_state_names) > 1 else self.probe_x_state))
        self.inspection_selected_point = (
            float(probe_meta.get("x_value", 0.0)),
            float(probe_meta.get("y_value", 0.0)),
        )
        self.probe_phase_start = float(integration_meta.get("t_transient", 0.0))
        self.probe_phase_end = float(integration_meta.get("t_end", self.probe_selected[-1]["t"] if self.probe_selected else 0.0))

        self.inspection_info.configure(
            text=f"Selected point: {self.probe_x_state}={self.inspection_selected_point[0]:.6g}, {self.probe_y_state}={self.inspection_selected_point[1]:.6g}"
        )
        self.probe_time_info.configure(
            text=f"Time series: selected method ({len(self.probe_selected)} samples) vs benchmark ({len(self.probe_benchmark)} samples)"
        )
        self.probe_phase_info.configure(
            text=f"Phase comparison: {self.probe_x_state} / {self.probe_y_state} | final window t={self.probe_phase_start:.6g}..{self.probe_phase_end:.6g} | drag rectangle to zoom, right-click reset"
        )
        self._redraw_inspection_canvas()
        self._redraw_probe_time_canvas()
        self._redraw_probe_phase_canvas()

    def _enter_inspection_mode(self) -> None:
        self.plot_notebook.select(2)
        self.current_summary_path = None
        self.inspection_selected_point = None
        self.probe_selected = []
        self.probe_benchmark = []
        self.probe_phase_start = None
        self.probe_phase_end = None
        self._clear_plot_zoom("inspection", "probe_time", "probe_phase")
        self.probe_state_names = parse_csv_strings(self.vars["state_names"].get())
        self.probe_x_state = self.vars["x_state"].get().strip()
        self.probe_y_state = self.vars["y_state"].get().strip()
        self.inspection_info.configure(text="Inspection mode ready. Click an initial condition in the plane.")
        self.probe_time_info.configure(text="Click a point to compute the full time history.")
        self.probe_phase_info.configure(text="Click a point to compute the phase comparison.")
        self.probe_time_canvas.delete("all")
        self.probe_phase_canvas.delete("all")
        self._redraw_inspection_canvas()

    def _inspection_plot_geometry(self) -> tuple[int, int, int, int, int, int, float, float, float, float] | None:
        try:
            xmin = float(self.vars["x_min"].get())
            xmax = float(self.vars["x_max"].get())
            ymin = float(self.vars["y_min"].get())
            ymax = float(self.vars["y_max"].get())
        except ValueError:
            return None
        if xmin == xmax or ymin == ymax:
            return None

        width = max(self.inspection_canvas.winfo_width(), 1)
        height = max(self.inspection_canvas.winfo_height(), 1)
        margin_left = 48
        margin_top = 18
        margin_right = 18
        margin_bottom = 36
        plot_w = max(width - margin_left - margin_right, 1)
        plot_h = max(height - margin_top - margin_bottom, 1)
        xmin, xmax, ymin, ymax = self._apply_zoom_view("inspection", (xmin, xmax, ymin, ymax))
        return margin_left, margin_top, margin_right, margin_bottom, plot_w, plot_h, xmin, xmax, ymin, ymax

    def _redraw_inspection_canvas(self) -> None:
        canvas = getattr(self, "inspection_canvas", None)
        if canvas is None:
            return
        canvas.delete("all")
        geometry = self._inspection_plot_geometry()
        if geometry is None:
            return
        margin_left, margin_top, _margin_right, _margin_bottom, plot_w, plot_h, xmin, xmax, ymin, ymax = geometry
        self._remember_plot_view("inspection", margin_left, margin_top, plot_w, plot_h, xmin, xmax, ymin, ymax)

        canvas.create_rectangle(margin_left, margin_top, margin_left + plot_w, margin_top + plot_h, outline="#555555")
        for tick in range(6):
            frac = tick / 5
            x = margin_left + frac * plot_w
            y = margin_top + frac * plot_h
            canvas.create_line(x, margin_top, x, margin_top + plot_h, fill="#eeeeee")
            canvas.create_line(margin_left, y, margin_left + plot_w, y, fill="#eeeeee")

        if self.inspection_selected_point is not None:
            x_value, y_value = self.inspection_selected_point
            px = margin_left + (x_value - xmin) / (xmax - xmin) * plot_w
            py = margin_top + (ymax - y_value) / (ymax - ymin) * plot_h
            canvas.create_line(px - 8, py, px + 8, py, fill="#111111", width=2)
            canvas.create_line(px, py - 8, px, py + 8, fill="#111111", width=2)
            canvas.create_oval(px - 4, py - 4, px + 4, py + 4, fill="#d1495b", outline="")

        self._draw_axis_labels(
            canvas,
            margin_left,
            margin_top,
            plot_w,
            plot_h,
            xmin,
            xmax,
            ymin,
            ymax,
            self.vars["x_state"].get().strip() or "x",
            self.vars["y_state"].get().strip() or "y",
        )

    def _on_inspection_click(self, event: tk.Event) -> None:
        if self.run_in_progress:
            return
        if self.vars["analysis_mode"].get().strip() != "inspection":
            self.inspection_info.configure(text="Switch Analysis Mode to inspection and press Run before selecting a point.")
            return

        geometry = self._inspection_plot_geometry()
        if geometry is None:
            return
        margin_left, margin_top, _margin_right, _margin_bottom, plot_w, plot_h, xmin, xmax, ymin, ymax = geometry
        px = min(max(float(event.x), float(margin_left)), float(margin_left + plot_w))
        py = min(max(float(event.y), float(margin_top)), float(margin_top + plot_h))
        x_value = xmin + (px - margin_left) / plot_w * (xmax - xmin)
        y_value = ymax - (py - margin_top) / plot_h * (ymax - ymin)

        config = self.collect_config()
        state_names = config["model"]["state_names"]
        initial_state = list(config["model"]["state_defaults"])
        x_idx = state_names.index(config["initial_condition_plane"]["x_state"])
        y_idx = state_names.index(config["initial_condition_plane"]["y_state"])
        initial_state[x_idx] = x_value
        initial_state[y_idx] = y_value
        config["analysis"]["mode"] = "point_probe"
        config["probe"] = {"initial_state": initial_state}

        self.inspection_selected_point = (x_value, y_value)
        self._redraw_inspection_canvas()
        self._start_point_probe(config)

    def _start_point_probe(self, config: dict) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        config_path = WORK_DIR / f"point_probe_config_{timestamp}.toml"
        dump_toml(config, config_path)

        self.probe_selected = []
        self.probe_benchmark = []
        self.probe_phase_start = None
        self.probe_phase_end = None
        self._clear_plot_zoom("probe_time", "probe_phase")
        self.probe_time_canvas.delete("all")
        self.probe_phase_canvas.delete("all")
        self.probe_time_info.configure(text="Computing selected method and benchmark...")
        self.probe_phase_info.configure(text="Computing selected method and benchmark...")
        self.current_summary_path = None
        self.current_result_dir = None

        self._submit_backend_job(config_path, "Sending point inspection job to persistent Julia backend:")

    def _uses_feature_colors(self) -> bool:
        return self.sweep_color_mode_var.get().strip().lower().startswith("feature")

    def _color_for_label(self, label: int) -> str:
        if self._uses_feature_colors():
            return self.current_label_feature_colors.get(label, self._label_color(label))
        return self._label_color(label)

    def _redraw_color_sensitive_plots(self) -> None:
        self._redraw_sweep_canvas()
        self._redraw_basin_canvas()
        self._redraw_phase_canvas()

    def _sweep_point_radius(self) -> float:
        try:
            return min(8.0, max(0.1, float(self.sweep_point_size_var.get())))
        except (tk.TclError, ValueError):
            return SWEEP_POINT_SIZE_DEFAULT

    def _on_sweep_point_size_changed(self, _value=None) -> None:
        radius = self._sweep_point_radius()
        if self.sweep_point_size_label is not None:
            self.sweep_point_size_label.configure(text=f"{radius:.1f}")
        if self._sweep_redraw_after_id is not None:
            self.after_cancel(self._sweep_redraw_after_id)
        self._sweep_redraw_after_id = self.after(80, self._redraw_sweep_canvas_after_size_change)

    def _redraw_sweep_canvas_after_size_change(self) -> None:
        self._sweep_redraw_after_id = None
        self._redraw_sweep_canvas()

    def _feature_colors_for_sweep_value(self, parameter_value: float) -> dict[int, str]:
        if not self.sweep_feature_colors_by_value:
            return {}
        nearest_value = min(self.sweep_feature_colors_by_value, key=lambda value: abs(value - parameter_value))
        return dict(self.sweep_feature_colors_by_value.get(nearest_value, {}))

    def _redraw_basin_canvas(self) -> None:
        canvas = getattr(self, "basin_canvas", None)
        if canvas is None:
            return
        canvas.delete("all")
        if not self.basin_rows:
            self.plot_viewports.pop("basin", None)
            return

        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        margin_left = 54
        margin_top = 24
        margin_right = 18
        margin_bottom = 44
        plot_w = max(width - margin_left - margin_right, 1)
        plot_h = max(height - margin_top - margin_bottom, 1)
        nrows = len(self.basin_rows)
        ncols = max(len(row) for row in self.basin_rows)
        bounds = self.basin_bounds or self._current_plane_bounds()
        if bounds is not None:
            self._remember_plot_view("basin", margin_left, margin_top, plot_w, plot_h, *bounds)
        else:
            self.plot_viewports.pop("basin", None)
        cell_w = plot_w / ncols
        cell_h = plot_h / nrows

        for display_row, row in enumerate(reversed(self.basin_rows)):
            y0 = margin_top + round(display_row * cell_h)
            y1 = margin_top + round((display_row + 1) * cell_h)
            for col, label in enumerate(row):
                x0 = margin_left + round(col * cell_w)
                x1 = margin_left + round((col + 1) * cell_w)
                canvas.create_rectangle(
                    x0,
                    y0,
                    x1,
                    y1,
                    outline="",
                    fill=self._color_for_label(label),
                )

        canvas.create_rectangle(
            margin_left,
            margin_top,
            margin_left + plot_w,
            margin_top + plot_h,
            outline="#555555",
            width=1,
        )
        if bounds is not None:
            xmin, xmax, ymin, ymax = bounds
            self._draw_axis_labels(canvas, margin_left, margin_top, plot_w, plot_h, xmin, xmax, ymin, ymax)
            canvas.create_text(
                margin_left + plot_w / 2,
                max(10, margin_top - 10),
                text=self._basin_zoom_factor_label(bounds),
                fill="#333333",
                font=("Segoe UI", 8, "bold"),
            )
            x_label = self.vars["x_state"].get().strip() or "x"
            y_label = self.vars["y_state"].get().strip() or "y"
            canvas.create_text(
                margin_left + plot_w,
                margin_top + plot_h + 34,
                text=x_label,
                fill="#333333",
                font=("Segoe UI", 8, "bold"),
                anchor="e",
            )
            canvas.create_text(
                8,
                margin_top + 4,
                text=y_label,
                fill="#333333",
                font=("Segoe UI", 8, "bold"),
                anchor="w",
            )

    def _redraw_phase_canvas(self) -> None:
        canvas = getattr(self, "phase_canvas", None)
        if canvas is None:
            return
        canvas.delete("all")
        if not self.phase_series or self.phase_bounds is None:
            self._update_phase_filter_status(0, 0)
            return
        display_series = self._phase_series_for_display()
        display_bounds = self._phase_bounds_for_series(display_series)
        self._update_phase_filter_status(len(display_series), len(self.phase_series))
        if not display_series or display_bounds is None:
            canvas.create_text(
                max(canvas.winfo_width(), 1) / 2,
                max(canvas.winfo_height(), 1) / 2,
                text="No phase trajectories meet the current minimum class fraction.",
                fill="#555555",
                font=("Segoe UI", 10),
                anchor="center",
            )
            return

        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        margin_left = 48
        margin_top = 18
        margin_right = 18
        margin_bottom = 36
        plot_w = max(width - margin_left - margin_right, 1)
        plot_h = max(height - margin_top - margin_bottom, 1)
        xmin, xmax, ymin, ymax = display_bounds
        pad_x = 0.04 * (xmax - xmin)
        pad_y = 0.04 * (ymax - ymin)
        xmin -= pad_x
        xmax += pad_x
        ymin -= pad_y
        ymax += pad_y
        xmin, xmax, ymin, ymax = self._apply_zoom_view("phase", (xmin, xmax, ymin, ymax))
        self._remember_plot_view("phase", margin_left, margin_top, plot_w, plot_h, xmin, xmax, ymin, ymax)

        def map_point(x_value: float, y_value: float) -> tuple[float, float]:
            px = margin_left + (x_value - xmin) / (xmax - xmin) * plot_w
            py = margin_top + (ymax - y_value) / (ymax - ymin) * plot_h
            return px, py

        canvas.create_rectangle(
            margin_left,
            margin_top,
            margin_left + plot_w,
            margin_top + plot_h,
            outline="#555555",
            width=1,
        )

        for tick in range(6):
            frac = tick / 5
            x = margin_left + frac * plot_w
            y = margin_top + frac * plot_h
            canvas.create_line(x, margin_top, x, margin_top + plot_h, fill="#eeeeee")
            canvas.create_line(margin_left, y, margin_left + plot_w, y, fill="#eeeeee")

        for item in display_series:
            points = item["points"]
            if len(points) < 2:
                continue
            step = max(1, len(points) // 1200)
            coords: list[float] = []
            for x_value, y_value in points[::step]:
                px, py = map_point(x_value, y_value)
                coords.extend([px, py])
            if len(coords) >= 4:
                canvas.create_line(
                    *coords,
                    fill=self._color_for_label(int(item["label"])),
                    width=1.5,
                    smooth=False,
                )

        phase_x_label, phase_y_label = self._current_phase_axis_labels()
        self._draw_axis_labels(canvas, margin_left, margin_top, plot_w, plot_h, xmin, xmax, ymin, ymax, phase_x_label, phase_y_label)

    def _sweep_plot_geometry(self) -> tuple[int, int, int, int, int, int, float, float, float, float] | None:
        canvas = getattr(self, "sweep_canvas", None)
        if canvas is None or self.sweep_bounds is None:
            return None

        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        margin_left = 58
        margin_top = 18
        margin_right = 22
        margin_bottom = 42
        plot_w = max(width - margin_left - margin_right, 1)
        plot_h = max(height - margin_top - margin_bottom, 1)
        xmin, xmax, ymin, ymax = self.sweep_bounds
        pad_x = 0.03 * (xmax - xmin)
        pad_y = 0.05 * (ymax - ymin)
        xmin, xmax, ymin, ymax = self._apply_zoom_view(
            "sweep",
            (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y),
        )
        return (
            margin_left,
            margin_top,
            margin_right,
            margin_bottom,
            plot_w,
            plot_h,
            xmin,
            xmax,
            ymin,
            ymax,
        )

    def _redraw_sweep_canvas(self) -> None:
        canvas = getattr(self, "sweep_canvas", None)
        if canvas is None:
            return
        canvas.delete("all")
        if not self.sweep_points or self.sweep_bounds is None:
            return

        geometry = self._sweep_plot_geometry()
        if geometry is None:
            return
        margin_left, margin_top, _margin_right, _margin_bottom, plot_w, plot_h, xmin, xmax, ymin, ymax = geometry
        self._remember_plot_view("sweep", margin_left, margin_top, plot_w, plot_h, xmin, xmax, ymin, ymax)

        canvas.create_rectangle(
            margin_left,
            margin_top,
            margin_left + plot_w,
            margin_top + plot_h,
            outline="#555555",
            width=1,
        )

        for tick in range(6):
            frac = tick / 5
            x = margin_left + frac * plot_w
            y = margin_top + frac * plot_h
            canvas.create_line(x, margin_top, x, margin_top + plot_h, fill="#eeeeee")
            canvas.create_line(margin_left, y, margin_left + plot_w, y, fill="#eeeeee")

        display_points = self._sweep_points_for_display(xmin, xmax, ymin, ymax)
        radius = self._sweep_point_radius()
        color_key = "feature_color" if self.sweep_color_mode_var.get().lower().startswith("feature") else "label_color"
        for point in display_points:
            parameter_value = point["parameter_value"]
            x_extreme = point["x_extreme"]
            color = point[color_key]
            px = margin_left + (parameter_value - xmin) / (xmax - xmin) * plot_w
            py = margin_top + (ymax - x_extreme) / (ymax - ymin) * plot_h
            canvas.create_rectangle(
                px - radius,
                py - radius,
                px + radius,
                py + radius,
                outline="",
                fill=color,
            )

        if self.selected_sweep_value is not None and xmin <= self.selected_sweep_value <= xmax:
            cursor_x = margin_left + (self.selected_sweep_value - xmin) / (xmax - xmin) * plot_w
            canvas.create_line(cursor_x, margin_top, cursor_x, margin_top + plot_h, fill="#111111", width=2)
            canvas.create_text(
                cursor_x,
                margin_top + 10,
                text=f"{self.selected_sweep_value:.5g}",
                fill="#111111",
                font=("Segoe UI", 8, "bold"),
                anchor="n",
            )

        sweep_x_label, sweep_y_label = self._current_sweep_axis_labels()
        self._draw_axis_labels(canvas, margin_left, margin_top, plot_w, plot_h, xmin, xmax, ymin, ymax, sweep_x_label, sweep_y_label)
        self._update_sweep_filter_status(len(display_points))

    def _sweep_points_for_display(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ) -> list[dict]:
        min_fraction, max_classes, max_points = self._sweep_filter_values()
        groups: dict[float, list[dict]] = {}
        for point in self.sweep_points:
            parameter_value = point["parameter_value"]
            x_extreme = point["x_extreme"]
            if xmin <= parameter_value <= xmax and ymin <= x_extreme <= ymax:
                if point.get("class_fraction", 1.0) < min_fraction:
                    continue
                if point.get("class_rank", 1) > max_classes:
                    continue
                groups.setdefault(parameter_value, []).append(point)

        if not groups:
            return []

        per_value_limit = max(1, max_points // max(1, len(groups)))
        display_points: list[dict] = []
        for parameter_value in sorted(groups):
            group = sorted(groups[parameter_value], key=lambda item: (item.get("class_rank", 1), item["x_extreme"]))
            if len(group) <= per_value_limit:
                display_points.extend(group)
            elif per_value_limit == 1:
                display_points.append(group[len(group) // 2])
            else:
                last_index = len(group) - 1
                for sample_index in range(per_value_limit):
                    idx = round(sample_index * last_index / (per_value_limit - 1))
                    display_points.append(group[idx])
        return display_points

    def _sweep_filter_values(self) -> tuple[float, int, int]:
        try:
            min_fraction = max(0.0, float(self.sweep_min_fraction_var.get().strip() or SWEEP_MIN_FRACTION_DEFAULT))
        except ValueError:
            min_fraction = float(SWEEP_MIN_FRACTION_DEFAULT)
        try:
            max_classes = max(1, int(float(self.sweep_max_classes_var.get().strip() or SWEEP_MAX_CLASSES_DEFAULT)))
        except ValueError:
            max_classes = int(SWEEP_MAX_CLASSES_DEFAULT)
        try:
            max_points = max(500, int(float(self.sweep_max_points_var.get().strip() or SWEEP_MAX_POINTS_DEFAULT)))
        except ValueError:
            max_points = int(SWEEP_MAX_POINTS_DEFAULT)
        return min_fraction, max_classes, max_points

    def _update_sweep_filter_status(self, displayed_points: int) -> None:
        if self.sweep_filter_status_label is None:
            return
        min_fraction, max_classes, max_points = self._sweep_filter_values()
        radius = self._sweep_point_radius()
        self.sweep_filter_status_label.configure(
            text=(
                f"drawn {displayed_points}/{len(self.sweep_points)} loaded"
                f" | min frac {min_fraction:g}, top {max_classes}, cap {max_points}, size {radius:.1f}"
            )
        )

    def _on_sweep_click(self, event: tk.Event) -> None:
        if not self.sweep_details:
            self.sweep_info.configure(text="This sweep result has no saved drill-down details. Recompute with Write Sweep Details = true.")
            return

        geometry = self._sweep_plot_geometry()
        if geometry is None:
            return
        margin_left, _margin_top, _margin_right, _margin_bottom, plot_w, _plot_h, xmin, xmax, _ymin, _ymax = geometry
        x_pixel = min(max(float(event.x), float(margin_left)), float(margin_left + plot_w))
        parameter_value = xmin + (x_pixel - margin_left) / plot_w * (xmax - xmin)
        detail = min(self.sweep_details, key=lambda item: abs(item["parameter_value"] - parameter_value))
        self._load_sweep_detail(detail)

    def _load_sweep_detail(self, detail: dict) -> None:
        self._clear_plot_zoom("phase")
        self.selected_sweep_value = float(detail["parameter_value"])
        self.current_image = None
        self.current_label_feature_colors = self._feature_colors_for_sweep_value(self.selected_sweep_value)
        self.basin_rows = self._load_label_rows(detail["labels_csv"])
        self.basin_bounds = self.result_plane_bounds or self._current_plane_bounds()
        nrows = len(self.basin_rows)
        ncols = max((len(row) for row in self.basin_rows), default=0)
        self.basin_info.configure(
            text=(
                f"Sweep value {self.selected_sweep_value:.8g} | "
                f"Classes: {detail['num_classes']} | Grid: {ncols} x {nrows} | "
                "Drag rectangle to recompute as single basin"
            )
        )
        self._redraw_basin_canvas()

        self.phase_series, self.phase_bounds = self._load_phase_series(detail["phase_samples_csv"])
        self._annotate_phase_series(self.phase_series, self._class_fractions_for_sweep_value(self.selected_sweep_value))
        shown = len(self._phase_series_for_display())
        self.phase_info.configure(
            text=(
                f"Sweep value {self.selected_sweep_value:.8g} | "
                f"Representative trajectories: {shown}/{len(self.phase_series)}"
            )
        )
        self._redraw_phase_canvas()
        self._redraw_sweep_canvas()
        self.plot_notebook.select(0)

    def _require_export_tools(self) -> bool:
        if Image is None or ImageDraw is None or ImageFont is None:
            messagebox.showerror("Export unavailable", "Pillow is required for image export, but it is not available.")
            return False
        return True

    def _pil_font(self, size: int = 12, *, bold: bool = False):
        if ImageFont is None:
            return None
        candidates = ["arialbd.ttf", "segoeuib.ttf"] if bold else ["arial.ttf", "segoeui.ttf"]
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
        return ImageFont.load_default()

    @staticmethod
    def _safe_export_name(text: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
        return cleaned.strip("_.") or "basin_export"

    def _default_export_stem(self) -> str:
        if self.current_result_dir is not None:
            return self._safe_export_name(self.current_result_dir.name)
        return f"basin_export_{time.strftime('%Y%m%d_%H%M%S')}"

    def _pil_label_color(self, label: int, feature_colors: dict[int, str] | None, *, feature_mode: bool | None = None) -> str:
        if feature_mode is None:
            feature_mode = self._uses_feature_colors()
        if feature_mode and feature_colors:
            return feature_colors.get(label, self._label_color(label))
        return self._label_color(label)

    def _draw_pil_axes(
        self,
        image,
        draw,
        margin_left: int,
        margin_top: int,
        plot_w: int,
        plot_h: int,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        x_label: str = "",
        y_label: str = "",
    ) -> None:
        font = self._pil_font(12)
        label_font = self._pil_font(13, bold=True)
        x_span = xmax - xmin
        y_span = ymax - ymin
        text_color = "#333333"
        for tick in range(3):
            frac = tick / 2
            x_value = xmin + frac * x_span
            x = margin_left + frac * plot_w
            y_value = ymax - frac * y_span
            y = margin_top + frac * plot_h
            draw.line([(x, margin_top + plot_h), (x, margin_top + plot_h + 5)], fill=text_color, width=1)
            draw.text((x, margin_top + plot_h + 14), self._format_axis_tick(x_value, x_span), fill=text_color, font=font, anchor="mm")
            draw.line([(margin_left - 5, y), (margin_left, y)], fill=text_color, width=1)
            draw.text((margin_left - 10, y), self._format_axis_tick(y_value, y_span), fill=text_color, font=font, anchor="rm")
        if x_label:
            draw.text(
                (margin_left + plot_w, margin_top + plot_h + 42),
                x_label,
                fill=text_color,
                font=label_font,
                anchor="rm",
            )
        if y_label:
            bbox = draw.textbbox((0, 0), y_label, font=label_font)
            label_w = max(1, bbox[2] - bbox[0] + 6)
            label_h = max(1, bbox[3] - bbox[1] + 6)
            label_image = Image.new("RGBA", (label_w, label_h), (255, 255, 255, 0))
            label_draw = ImageDraw.Draw(label_image)
            label_draw.text((3, 3), y_label, fill=text_color, font=label_font, anchor="la")
            rotated = label_image.rotate(90, expand=True)
            image.paste(
                rotated,
                (
                    max(2, margin_left - 58),
                    round(margin_top + plot_h / 2 - rotated.height / 2),
                ),
                rotated,
            )

    @staticmethod
    def _square_plot_layout(
        width: int,
        height: int,
        margin_left: int,
        margin_top: int,
        margin_right: int,
        margin_bottom: int,
    ) -> tuple[int, int, int, int]:
        available_w = max(width - margin_left - margin_right, 1)
        available_h = max(height - margin_top - margin_bottom, 1)
        plot_size = max(1, min(available_w, available_h))
        margin_left += max(0, (available_w - plot_size) // 2)
        margin_top += max(0, (available_h - plot_size) // 2)
        return margin_left, margin_top, plot_size, plot_size

    def _current_plane_axis_labels(self) -> tuple[str, str]:
        summary_plane = (self.current_summary_data or {}).get("initial_condition_plane", {})
        x_label = str(summary_plane.get("x_state", ""))
        y_label = str(summary_plane.get("y_state", ""))
        if not x_label and hasattr(self, "vars"):
            x_label = self.vars["x_state"].get().strip()
        if not y_label and hasattr(self, "vars"):
            y_label = self.vars["y_state"].get().strip()
        return x_label or "x", y_label or "v"

    def _current_phase_axis_labels(self) -> tuple[str, str]:
        phase_meta = (self.current_summary_data or {}).get("phase", {})
        x_label = str(phase_meta.get("x_state", ""))
        y_label = str(phase_meta.get("y_state", ""))
        if x_label and y_label:
            return x_label, y_label
        config = self._current_result_config()
        classification = config.get("classification", {}) if config else {}
        plane_x, plane_y = self._current_plane_axis_labels()
        return str(classification.get("observable_state", plane_x)), str(classification.get("zero_cross_state", plane_y))

    def _current_sweep_axis_labels(self) -> tuple[str, str]:
        sweep_param = str((self.current_summary_data or {}).get("sweep", {}).get("parameter", "parameter"))
        phase_x, _phase_y = self._current_phase_axis_labels()
        return sweep_param, f"{phase_x} extrema"

    def _draw_pil_plot_shell(
        self,
        draw,
        margin_left: int,
        margin_top: int,
        plot_w: int,
        plot_h: int,
        title: str,
    ) -> None:
        title_font = self._pil_font(14, bold=True)
        if title:
            draw.text((margin_left, 8), title, fill="#222222", font=title_font, anchor="la")
        draw.rectangle(
            [margin_left, margin_top, margin_left + plot_w, margin_top + plot_h],
            outline="#555555",
            width=1,
        )
        for tick in range(6):
            frac = tick / 5
            x = margin_left + frac * plot_w
            y = margin_top + frac * plot_h
            draw.line([(x, margin_top), (x, margin_top + plot_h)], fill="#eeeeee", width=1)
            draw.line([(margin_left, y), (margin_left + plot_w, y)], fill="#eeeeee", width=1)

    def _render_placeholder_image(self, width: int, height: int, title: str, message: str):
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((24, 18), title, fill="#222222", font=self._pil_font(16, bold=True), anchor="la")
        draw.text((width / 2, height / 2), message, fill="#555555", font=self._pil_font(14), anchor="mm")
        return image

    def _render_basin_image(
        self,
        width: int,
        height: int,
        *,
        rows: list[list[int]] | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        feature_colors: dict[int, str] | None = None,
        title: str = "Basin map",
        apply_zoom: bool = True,
        feature_mode: bool | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
    ):
        rows = self.basin_rows if rows is None else rows
        bounds = self.basin_bounds if bounds is None else bounds
        feature_colors = self.current_label_feature_colors if feature_colors is None else feature_colors
        if not rows or bounds is None:
            return self._render_placeholder_image(width, height, title, "No basin map loaded")

        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        margin_left = 66
        margin_top = 42
        margin_right = 24
        margin_bottom = 58
        margin_left, margin_top, plot_w, plot_h = self._square_plot_layout(width, height, margin_left, margin_top, margin_right, margin_bottom)
        xmin, xmax, ymin, ymax = self._apply_zoom_view("basin", bounds) if apply_zoom else bounds
        self._draw_pil_plot_shell(draw, margin_left, margin_top, plot_w, plot_h, title)

        nrows = len(rows)
        ncols = max(len(row) for row in rows)
        cell_w = plot_w / ncols
        cell_h = plot_h / nrows
        for display_row, row in enumerate(reversed(rows)):
            y0 = margin_top + round(display_row * cell_h)
            y1 = margin_top + round((display_row + 1) * cell_h)
            for col, label in enumerate(row):
                x0 = margin_left + round(col * cell_w)
                x1 = margin_left + round((col + 1) * cell_w)
                draw.rectangle([x0, y0, x1, y1], fill=self._pil_label_color(label, feature_colors, feature_mode=feature_mode))

        draw.rectangle([margin_left, margin_top, margin_left + plot_w, margin_top + plot_h], outline="#555555", width=1)
        if x_label is None or y_label is None:
            default_x, default_y = self._current_plane_axis_labels()
            x_label = default_x if x_label is None else x_label
            y_label = default_y if y_label is None else y_label
        self._draw_pil_axes(image, draw, margin_left, margin_top, plot_w, plot_h, xmin, xmax, ymin, ymax, x_label, y_label)
        return image

    def _render_phase_image(
        self,
        width: int,
        height: int,
        *,
        series: list[dict] | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        feature_colors: dict[int, str] | None = None,
        title: str = "Phase portrait",
        apply_zoom: bool = True,
        feature_mode: bool | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        min_fraction: float | None = None,
    ):
        series = self.phase_series if series is None else series
        bounds = self.phase_bounds if bounds is None else bounds
        display_series = self._phase_series_for_display(series, min_fraction=min_fraction)
        display_bounds = self._phase_bounds_for_series(display_series) or bounds
        feature_colors = self.current_label_feature_colors if feature_colors is None else feature_colors
        if not display_series or display_bounds is None:
            return self._render_placeholder_image(width, height, title, "No phase data loaded")

        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        margin_left = 66
        margin_top = 42
        margin_right = 24
        margin_bottom = 58
        margin_left, margin_top, plot_w, plot_h = self._square_plot_layout(width, height, margin_left, margin_top, margin_right, margin_bottom)
        xmin, xmax, ymin, ymax = display_bounds
        pad_x = 0.04 * (xmax - xmin)
        pad_y = 0.04 * (ymax - ymin)
        view = (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)
        xmin, xmax, ymin, ymax = self._apply_zoom_view("phase", view) if apply_zoom else view
        self._draw_pil_plot_shell(draw, margin_left, margin_top, plot_w, plot_h, title)

        def map_point(x_value: float, y_value: float) -> tuple[float, float]:
            return (
                margin_left + (x_value - xmin) / (xmax - xmin) * plot_w,
                margin_top + (ymax - y_value) / (ymax - ymin) * plot_h,
            )

        for item in display_series:
            points = item["points"]
            if len(points) < 2:
                continue
            step = max(1, len(points) // max(plot_w * 2, 1200))
            coords = [map_point(x_value, y_value) for x_value, y_value in points[::step]]
            if len(coords) >= 2:
                draw.line(coords, fill=self._pil_label_color(int(item["label"]), feature_colors, feature_mode=feature_mode), width=2)

        if x_label is None or y_label is None:
            default_x, default_y = self._current_phase_axis_labels()
            x_label = default_x if x_label is None else x_label
            y_label = default_y if y_label is None else y_label
        self._draw_pil_axes(image, draw, margin_left, margin_top, plot_w, plot_h, xmin, xmax, ymin, ymax, x_label, y_label)
        return image

    @staticmethod
    def _sweep_export_geometry(
        width: int,
        height: int,
        bounds: tuple[float, float, float, float] | None,
        *,
        apply_zoom: bool,
        zoom_views: dict[str, tuple[float, float, float, float]] | None,
    ):
        if bounds is None:
            return None
        margin_left = 72
        margin_top = 42
        margin_right = 28
        margin_bottom = 62
        margin_left, margin_top, plot_w, plot_h = BasinGuiApp._square_plot_layout(
            width,
            height,
            margin_left,
            margin_top,
            margin_right,
            margin_bottom,
        )
        xmin, xmax, ymin, ymax = bounds
        pad_x = 0.03 * (xmax - xmin)
        pad_y = 0.05 * (ymax - ymin)
        view = (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)
        if apply_zoom and zoom_views and "sweep" in zoom_views:
            xmin, xmax, ymin, ymax = zoom_views["sweep"]
        else:
            xmin, xmax, ymin, ymax = view
        return margin_left, margin_top, plot_w, plot_h, xmin, xmax, ymin, ymax

    @staticmethod
    def _sweep_points_for_display_static(
        points: list[dict],
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        filter_values: tuple[float, int, int],
    ) -> list[dict]:
        min_fraction, max_classes, max_points = filter_values
        groups: dict[float, list[dict]] = {}
        for point in points:
            parameter_value = point["parameter_value"]
            x_extreme = point["x_extreme"]
            if xmin <= parameter_value <= xmax and ymin <= x_extreme <= ymax:
                if point.get("class_fraction", 1.0) < min_fraction:
                    continue
                if point.get("class_rank", 1) > max_classes:
                    continue
                groups.setdefault(parameter_value, []).append(point)

        if not groups:
            return []

        per_value_limit = max(1, max_points // max(1, len(groups)))
        display_points: list[dict] = []
        for parameter_value in sorted(groups):
            group = sorted(groups[parameter_value], key=lambda item: (item.get("class_rank", 1), item["x_extreme"]))
            if len(group) <= per_value_limit:
                display_points.extend(group)
            elif per_value_limit == 1:
                display_points.append(group[len(group) // 2])
            else:
                last_index = len(group) - 1
                for sample_index in range(per_value_limit):
                    idx = round(sample_index * last_index / (per_value_limit - 1))
                    display_points.append(group[idx])
        return display_points

    def _sweep_export_bounds(self, width: int, height: int, *, apply_zoom: bool = True):
        return self._sweep_export_geometry(width, height, self.sweep_bounds, apply_zoom=apply_zoom, zoom_views=self.zoom_views)

    def _render_sweep_image_from_data(
        self,
        width: int,
        height: int,
        *,
        points: list[dict],
        bounds: tuple[float, float, float, float] | None,
        filter_values: tuple[float, int, int],
        point_radius: float,
        feature_mode: bool,
        selected_value: float | None,
        title: str,
        apply_zoom: bool,
        zoom_views: dict[str, tuple[float, float, float, float]] | None,
        x_label: str = "parameter",
        y_label: str = "extrema",
    ):
        geometry = self._sweep_export_geometry(width, height, bounds, apply_zoom=apply_zoom, zoom_views=zoom_views)
        if not points or geometry is None:
            return self._render_placeholder_image(width, height, title, "No sweep data loaded")

        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        margin_left, margin_top, plot_w, plot_h, xmin, xmax, ymin, ymax = geometry
        self._draw_pil_plot_shell(draw, margin_left, margin_top, plot_w, plot_h, title)

        display_points = self._sweep_points_for_display_static(points, xmin, xmax, ymin, ymax, filter_values)
        color_key = "feature_color" if feature_mode else "label_color"
        for point in display_points:
            parameter_value = point["parameter_value"]
            x_extreme = point["x_extreme"]
            px = margin_left + (parameter_value - xmin) / (xmax - xmin) * plot_w
            py = margin_top + (ymax - x_extreme) / (ymax - ymin) * plot_h
            draw.rectangle(
                [px - point_radius, py - point_radius, px + point_radius, py + point_radius],
                fill=point[color_key],
            )

        if selected_value is not None and xmin <= selected_value <= xmax:
            cursor_x = margin_left + (selected_value - xmin) / (xmax - xmin) * plot_w
            draw.line([(cursor_x, margin_top), (cursor_x, margin_top + plot_h)], fill="#111111", width=3)
            draw.text((cursor_x, margin_top + 14), f"{selected_value:.5g}", fill="#111111", font=self._pil_font(12, bold=True), anchor="mt")

        self._draw_pil_axes(image, draw, margin_left, margin_top, plot_w, plot_h, xmin, xmax, ymin, ymax, x_label, y_label)
        return image

    def _render_sweep_image(
        self,
        width: int,
        height: int,
        *,
        selected_value: float | None = None,
        title: str = "Sweep overview",
        apply_zoom: bool = True,
        x_label: str | None = None,
        y_label: str | None = None,
    ):
        selected_value = self.selected_sweep_value if selected_value is None else selected_value
        if x_label is None or y_label is None:
            default_x, default_y = self._current_sweep_axis_labels()
            x_label = default_x if x_label is None else x_label
            y_label = default_y if y_label is None else y_label
        return self._render_sweep_image_from_data(
            width,
            height,
            points=self.sweep_points,
            bounds=self.sweep_bounds,
            filter_values=self._sweep_filter_values(),
            point_radius=self._sweep_point_radius(),
            feature_mode=self._uses_feature_colors(),
            selected_value=selected_value,
            title=title,
            apply_zoom=apply_zoom,
            zoom_views=self.zoom_views,
            x_label=x_label,
            y_label=y_label,
        )

    def export_current_images(self) -> None:
        if not self._require_export_tools():
            return
        if not self.current_result_dir:
            messagebox.showinfo("No Result", "No result is loaded yet.")
            return

        initial_dir = self.current_result_dir if self.current_result_dir.exists() else BASE_DIR
        out_dir_raw = filedialog.askdirectory(title="Export Images", initialdir=str(initial_dir))
        if not out_dir_raw:
            return
        out_dir = Path(out_dir_raw)
        stem = self._default_export_stem()
        width, height = EXPORT_IMAGE_SIZE
        exported: list[Path] = []

        try:
            if self.basin_rows:
                path = out_dir / f"{stem}_basin_map.png"
                self._render_basin_image(width, height).save(path)
                exported.append(path)
            if self.phase_series:
                path = out_dir / f"{stem}_phase_portrait.png"
                self._render_phase_image(width, height).save(path)
                exported.append(path)
            if self.sweep_points:
                path = out_dir / f"{stem}_sweep_overview.png"
                self._render_sweep_image(width, height).save(path)
                exported.append(path)
        except Exception as exc:
            messagebox.showerror("Export Failed", str(exc))
            return

        if not exported:
            messagebox.showinfo("No Images", "No plot data is currently available to export.")
            return
        self._append_log("Exported images:")
        for path in exported:
            self._append_log(str(path))
        messagebox.showinfo("Export Complete", "Saved:\n" + "\n".join(str(path) for path in exported))

    @staticmethod
    def _feature_colors_for_sweep_value_map(colors_by_value: dict[float, dict[int, str]], parameter_value: float) -> dict[int, str]:
        if not colors_by_value:
            return {}
        nearest_value = min(colors_by_value, key=lambda value: abs(value - parameter_value))
        return dict(colors_by_value.get(nearest_value, {}))

    def _draw_wrapped_pil_text(self, draw, x: int, y: int, text: str, font, fill: str, max_width: int, line_spacing: int = 4) -> int:
        words = text.split()
        if not words:
            return y

        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if bbox[2] - bbox[0] <= max_width:
                line = candidate
            else:
                draw.text((x, y), line, fill=fill, font=font, anchor="la")
                line_box = draw.textbbox((0, 0), line, font=font)
                y += (line_box[3] - line_box[1]) + line_spacing
                line = word
        draw.text((x, y), line, fill=fill, font=font, anchor="la")
        line_box = draw.textbbox((0, 0), line, font=font)
        return y + (line_box[3] - line_box[1]) + line_spacing

    @staticmethod
    def _latex_identifier(name: str) -> str:
        if name in {"pi", "π"}:
            return r"\pi"
        match = re.fullmatch(r"([A-Za-z_]+)(\d+)", name)
        if match:
            base, subscript = match.groups()
            return rf"{base}_{{{subscript}}}"
        return name.replace("_", r"\_")

    @staticmethod
    def _latex_number(value) -> str:
        if isinstance(value, int):
            return str(value)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if numeric.is_integer():
            return str(int(numeric))
        fraction = Fraction(str(numeric)).limit_denominator(20)
        if fraction.denominator != 1 and abs(float(fraction) - numeric) < 1e-12:
            return rf"\frac{{{fraction.numerator}}}{{{fraction.denominator}}}"
        return f"{numeric:.8g}"

    def _latex_ast(self, node, parent: str = "") -> str:
        if isinstance(node, ast.Expression):
            return self._latex_ast(node.body, parent)
        if isinstance(node, ast.Constant):
            return self._latex_number(node.value)
        if isinstance(node, ast.Name):
            return self._latex_identifier(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = self._latex_ast(node.operand, "unary")
            return rf"-{value}"
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Add, ast.Sub)):
                left = self._latex_ast(node.left, "add")
                right = self._latex_ast(node.right, "add")
                sign = "+" if isinstance(node.op, ast.Add) else "-"
                text = f"{left} {sign} {right}"
                return rf"\left({text}\right)" if parent in {"mul", "pow"} else text
            if isinstance(node.op, ast.Mult):
                left = self._latex_ast(node.left, "mul")
                right = self._latex_ast(node.right, "mul")
                return rf"{left}\,{right}"
            if isinstance(node.op, ast.Div):
                left = self._latex_ast(node.left, "frac")
                right = self._latex_ast(node.right, "frac")
                return rf"\frac{{{left}}}{{{right}}}"
            if isinstance(node.op, ast.Pow):
                left = self._latex_ast(node.left, "pow")
                right = self._latex_ast(node.right, "pow")
                return rf"{left}^{{{right}}}"
        if isinstance(node, ast.Call):
            func = self._latex_ast(node.func, "call")
            args = [self._latex_ast(arg, "call") for arg in node.args]
            if func == "sqrt" and args:
                return rf"\sqrt{{{args[0]}}}"
            if func in {"sin", "cos", "tan", "exp", "log"} and args:
                return rf"\{func}\left({args[0]}\right)"
            if func == "abs" and args:
                return rf"\left|{args[0]}\right|"
            if func in {"min", "max"} and args:
                return rf"\{func}\left({', '.join(args)}\right)"
            return rf"\mathrm{{{func}}}\left({', '.join(args)}\right)"
        return rf"\mathrm{{{str(ast.unparse(node)) if hasattr(ast, 'unparse') else '?'}}}"

    def _expression_to_latex(self, expression: str) -> str:
        prepared = expression.replace("^", "**").replace("π", "pi")
        try:
            parsed = ast.parse(prepared, mode="eval")
            return self._latex_ast(parsed)
        except SyntaxError:
            text = expression.replace("*", r"\,").replace("^", "^")
            text = re.sub(r"\b([A-Za-z_]+)(\d+)\b", lambda m: rf"{m.group(1)}_{{{m.group(2)}}}", text)
            return text

    def _latex_equations(self, snapshot: dict) -> list[str]:
        state_names = snapshot.get("state_names", [])
        equations = snapshot.get("equations", [])
        lines: list[str] = []
        for idx, equation in enumerate(equations):
            state_name = state_names[idx] if idx < len(state_names) else f"u{idx + 1}"
            lhs = rf"\frac{{d {self._latex_identifier(str(state_name))}}}{{d t}}"
            rhs = self._expression_to_latex(str(equation))
            lines.append(rf"${lhs} = {rhs}$")
        return lines

    def _render_mathtext_image(self, math_text: str, *, fontsize: int = 19, color: str = "#222222"):
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            from matplotlib import mathtext
            from matplotlib.font_manager import FontProperties

            buffer = io.BytesIO()
            mathtext.math_to_image(
                math_text,
                buffer,
                prop=FontProperties(size=fontsize),
                dpi=150,
                format="png",
                color=color,
            )
            buffer.seek(0)
            return Image.open(buffer).convert("RGBA")
        except Exception:
            fallback = Image.new("RGBA", (1200, 32), (255, 255, 255, 0))
            draw = ImageDraw.Draw(fallback)
            draw.text((0, 0), math_text.strip("$"), fill=color, font=self._pil_font(18), anchor="la")
            bbox = fallback.getbbox()
            return fallback.crop(bbox) if bbox else fallback

    def _paste_latex_equations(self, frame, snapshot: dict, x: int, y: int, max_width: int, max_bottom: int) -> int:
        cache = snapshot.setdefault("_mathtext_cache", {})
        for equation in self._latex_equations(snapshot):
            key = (equation, 17, "#222222")
            image = cache.get(key)
            if image is None:
                image = self._render_mathtext_image(equation, fontsize=17, color="#222222")
                cache[key] = image
            if image.width > max_width:
                scale = max_width / image.width
                image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
            if y + image.height > max_bottom:
                break
            frame.paste(image, (x, y), image)
            y += image.height + 6
        return y

    def _video_header_text(self, snapshot: dict, parameter_value: float) -> tuple[str, str, str]:
        sweep_param = snapshot.get("sweep_parameter", "parameter") or "parameter"
        parameter_names = list(snapshot.get("parameter_names", []))
        parameter_values = list(snapshot.get("parameter_values", []))
        parameter_texts = []
        for idx, name in enumerate(parameter_names):
            value = parameter_values[idx] if idx < len(parameter_values) else ""
            if name == sweep_param:
                value = parameter_value
            try:
                value_text = f"{float(value):.8g}"
            except (TypeError, ValueError):
                value_text = str(value)
            parameter_texts.append(f"{name}={value_text}")
        parameter_text = ", ".join(parameter_texts) if parameter_texts else f"{sweep_param}={parameter_value:.8g}"
        title = f"Basin of attraction sweep: {sweep_param} = {parameter_value:.8g}"
        subtitle = "Basin map, phase portrait, and steady-state extrema overview"
        return title, subtitle, parameter_text

    def _draw_video_header(self, frame, snapshot: dict, parameter_value: float) -> None:
        draw = ImageDraw.Draw(frame)
        margin_x = VIDEO_MARGIN
        title_font = self._pil_font(27, bold=True)
        subtitle_font = self._pil_font(16)
        small_font = self._pil_font(15)
        label_font = self._pil_font(14, bold=True)
        title, subtitle, parameter_text = self._video_header_text(snapshot, parameter_value)
        max_width = VIDEO_FRAME_SIZE[0] - 2 * margin_x
        draw.text((margin_x, 20), title, fill="#111111", font=title_font, anchor="la")
        draw.text((margin_x, 58), subtitle, fill="#333333", font=subtitle_font, anchor="la")
        draw.text((margin_x, 92), "Model equations", fill="#333333", font=label_font, anchor="la")
        y = self._paste_latex_equations(frame, snapshot, margin_x, 114, max_width, VIDEO_HEADER_HEIGHT - 42)
        self._draw_wrapped_pil_text(
            draw,
            margin_x,
            max(y + 4, VIDEO_HEADER_HEIGHT - 36),
            f"Parameters: {parameter_text}",
            small_font,
            "#333333",
            max_width,
            4,
        )
        draw.line([(0, VIDEO_HEADER_HEIGHT - 1), (VIDEO_FRAME_SIZE[0], VIDEO_HEADER_HEIGHT - 1)], fill="#dddddd", width=1)

    @staticmethod
    def _format_video_value(value) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return f"{value:.8g}"
        if isinstance(value, int):
            return str(value)
        return str(value)

    def _video_key_value_lines(self, values: dict, keys: list[str]) -> list[str]:
        lines: list[str] = []
        for key in keys:
            if key not in values:
                continue
            value = values[key]
            if isinstance(value, list):
                value_text = "[" + ", ".join(self._format_video_value(item) for item in value) + "]"
            else:
                value_text = self._format_video_value(value)
            lines.append(f"{key}: {value_text}")
        return lines

    def _video_integration_lines(self, integration: dict) -> list[str]:
        mode = str(integration.get("solver_mode", "")).strip()
        solver = str(integration.get("solver", "")).strip()
        mode_l = mode.lower()

        if mode_l == "fixed":
            solver_setup = f"{solver} fixed-step" if solver else "fixed-step"
        elif mode_l == "adaptive":
            solver_setup = f"{solver} adaptive" if solver else "adaptive"
        elif mode_l == "streaming_extrema":
            solver_setup = f"{solver} fixed-step Stage A; streaming extrema extraction in Stage B"
        elif mode_l == "rk4_extrema":
            solver_setup = f"{solver} fixed-step Stage A; CUDA RK4 extrema kernel in Stage B"
        elif mode_l == "rk4_full_extrema_custom":
            solver_setup = "generated CUDA RK4 A+B extrema kernel"
        elif mode_l == "rk4_full_extrema":
            solver_setup = "optimized CUDA RK4 A+B extrema kernel"
        else:
            solver_setup = mode or solver or "unknown"

        lines = [
            f"device: {self._format_video_value(integration.get('device', ''))}",
            f"precision: {self._format_video_value(integration.get('precision', ''))}",
            f"solver setup: {solver_setup}",
        ]
        return lines + self._video_key_value_lines(
            integration,
            [
                "period_mode",
                "custom_time_argument",
                "period_expression",
                "transient_periods",
                "evaluation_periods",
                "samples_per_period",
                "t_transient",
                "t_evaluation",
                "dt",
                "save_dt",
                "streaming_chunk_periods",
                "abstol",
                "reltol",
            ],
        )

    def _draw_video_text_block(
        self,
        draw,
        title: str,
        lines: list[str],
        x: int,
        y: int,
        width: int,
        max_bottom: int,
    ) -> int:
        title_font = self._pil_font(18, bold=True)
        text_font = self._pil_font(15)
        draw.text((x, y), title, fill="#222222", font=title_font, anchor="la")
        y += 28
        for line in lines:
            if y > max_bottom - 20:
                break
            y = self._draw_wrapped_pil_text(draw, x, y, line, text_font, "#333333", width, 4)
        return y

    def _sweep_video_intro_frame(self, snapshot: dict):
        frame_w, frame_h = VIDEO_FRAME_SIZE
        frame = Image.new("RGB", VIDEO_FRAME_SIZE, "white")
        draw = ImageDraw.Draw(frame)
        margin = VIDEO_MARGIN
        title_font = self._pil_font(34, bold=True)
        subtitle_font = self._pil_font(18)
        small_font = self._pil_font(15)
        label_font = self._pil_font(18, bold=True)

        sweep_param = str(snapshot.get("sweep_parameter", "parameter"))
        sweep = dict(snapshot.get("sweep_config", {}))
        value_count = len(snapshot.get("details", [])) or sweep.get("num_values", "")
        sweep_min = sweep.get("min_value", "")
        sweep_max = sweep.get("max_value", "")

        draw.text((margin, 34), "Basin of attraction parameter sweep", fill="#111111", font=title_font, anchor="la")
        subtitle = f"{sweep_param}: {self._format_video_value(sweep_min)} to {self._format_video_value(sweep_max)}"
        if value_count:
            subtitle += f" | {value_count} values"
        draw.text((margin, 82), subtitle, fill="#333333", font=subtitle_font, anchor="la")

        run_dir = snapshot.get("run_dir", "")
        if run_dir:
            self._draw_wrapped_pil_text(draw, margin, 114, f"Result: {run_dir}", small_font, "#555555", frame_w - 2 * margin, 3)

        content_top = 162
        left_x = margin
        right_x = frame_w // 2 + 18
        column_w = frame_w // 2 - margin - 42
        bottom = frame_h - 42

        draw.text((left_x, content_top), "Model equations", fill="#222222", font=label_font, anchor="la")
        y_left = self._paste_latex_equations(frame, snapshot, left_x, content_top + 34, column_w, content_top + 250)

        parameter_names = list(snapshot.get("parameter_names", []))
        parameter_values = list(snapshot.get("parameter_values", []))
        parameter_lines: list[str] = []
        for idx, name in enumerate(parameter_names):
            value = parameter_values[idx] if idx < len(parameter_values) else ""
            if str(name) == sweep_param and sweep_min != "":
                parameter_lines.append(f"{name}: sweep {self._format_video_value(sweep_min)} to {self._format_video_value(sweep_max)}")
            else:
                parameter_lines.append(f"{name}: {self._format_video_value(value)}")
        self._draw_video_text_block(draw, "Model parameters", parameter_lines, left_x, max(y_left + 26, content_top + 292), column_w, bottom)

        integration = dict(snapshot.get("integration_config", {}))
        integration_lines = self._video_integration_lines(integration)

        output = dict(snapshot.get("output_config", {}))
        classification = dict(snapshot.get("classification_config", {}))
        extra_lines = self._video_key_value_lines(
            classification,
            ["observable_state", "zero_cross_state", "fingerprint_tol", "fingerprint_k", "extrema_eps", "max_extrema"],
        )
        if output:
            extra_lines.extend(self._video_key_value_lines(output, ["run_name", "write_sweep_details"]))

        y_right = self._draw_video_text_block(draw, "Integration", integration_lines, right_x, content_top, column_w, bottom)
        if extra_lines:
            self._draw_video_text_block(draw, "Classification / output", extra_lines, right_x, min(y_right + 30, bottom - 180), column_w, bottom)

        draw.line([(0, frame_h - 1), (frame_w, frame_h - 1)], fill="#dddddd", width=1)
        return frame

    def _current_result_config(self) -> dict:
        if self.current_result_dir is None:
            return {}
        config_path = self.current_result_dir / "config_snapshot.toml"
        if not config_path.is_file():
            return {}
        try:
            with config_path.open("rb") as handle:
                return dict(tomllib.load(handle))
        except Exception:
            return {}

    def _sweep_video_snapshot(self) -> dict:
        summary = self.current_summary_data or {}
        config = self._current_result_config()
        summary_model = summary.get("model", {})
        config_model = config.get("model", {}) if config else {}
        model = {**summary_model, **config_model}
        sweep_config = config.get("sweep", {}) if config else {}
        integration_config = {**(config.get("integration", {}) if config else {}), **summary.get("integration", {})}
        classification_config = config.get("classification", {}) if config else {}
        output_config = config.get("output", {}) if config else {}
        plane = {**(config.get("initial_condition_plane", {}) if config else {}), **summary.get("initial_condition_plane", {})}
        classification = config.get("classification", {}) if config else {}
        plane_x = str(plane.get("x_state", "x"))
        plane_y = str(plane.get("y_state", "v"))
        phase_x = str(classification.get("observable_state", plane_x))
        phase_y = str(classification.get("zero_cross_state", plane_y))
        sweep_parameter = summary.get("sweep", {}).get("parameter", config.get("sweep", {}).get("parameter", "parameter") if config else "parameter")
        return {
            "details": sorted([dict(item) for item in self.sweep_details], key=lambda item: item["parameter_value"]),
            "sweep_points": [dict(point) for point in self.sweep_points],
            "sweep_bounds": self.sweep_bounds,
            "sweep_feature_colors_by_value": {float(value): dict(colors) for value, colors in self.sweep_feature_colors_by_value.items()},
            "sweep_class_fractions_by_value": {float(value): dict(fractions) for value, fractions in self.sweep_class_fractions_by_value.items()},
            "result_plane_bounds": self.result_plane_bounds,
            "feature_mode": self._uses_feature_colors(),
            "point_radius": self._sweep_point_radius(),
            "filter_values": self._sweep_filter_values(),
            "phase_min_fraction": self._phase_min_fraction(),
            "zoom_views": dict(self.zoom_views),
            "run_dir": str(summary.get("run", {}).get("run_dir", "")),
            "sweep_config": sweep_config,
            "integration_config": integration_config,
            "classification_config": classification_config,
            "output_config": output_config,
            "sweep_parameter": sweep_parameter,
            "plane_x_state": plane_x,
            "plane_y_state": plane_y,
            "phase_x_state": phase_x,
            "phase_y_state": phase_y,
            "state_names": list(model.get("state_names", [])),
            "equations": list(model.get("equations", [])),
            "parameter_names": list(model.get("parameter_names", [])),
            "parameter_values": list(model.get("parameter_values", [])),
        }

    def _sweep_video_frame_from_snapshot(self, snapshot: dict, detail: dict):
        frame_w, frame_h = VIDEO_FRAME_SIZE
        header_h = VIDEO_HEADER_HEIGHT
        panel_side = min(
            (frame_w - 2 * VIDEO_MARGIN - 2 * VIDEO_PANEL_GAP) // 3,
            frame_h - header_h - VIDEO_MARGIN,
        )
        panel_total_w = 3 * panel_side + 2 * VIDEO_PANEL_GAP
        panel_x0 = (frame_w - panel_total_w) // 2
        panel_y = header_h + max(12, (frame_h - header_h - panel_side) // 2)
        value = float(detail["parameter_value"])
        feature_colors = self._feature_colors_for_sweep_value_map(snapshot["sweep_feature_colors_by_value"], value)
        rows = self._load_label_rows(detail["labels_csv"])
        series, phase_bounds = self._load_phase_series(detail["phase_samples_csv"])
        class_fractions = self._class_fractions_for_sweep_value_map(snapshot.get("sweep_class_fractions_by_value", {}), value)
        self._annotate_phase_series(series, class_fractions)

        basin_image = self._render_basin_image(
            panel_side,
            panel_side,
            rows=rows,
            bounds=snapshot["result_plane_bounds"],
            feature_colors=feature_colors,
            title="Basin map",
            apply_zoom=False,
            feature_mode=snapshot["feature_mode"],
            x_label=snapshot.get("plane_x_state", "x"),
            y_label=snapshot.get("plane_y_state", "v"),
        )
        phase_image = self._render_phase_image(
            panel_side,
            panel_side,
            series=series,
            bounds=phase_bounds,
            feature_colors=feature_colors,
            title="Phase portrait",
            apply_zoom=False,
            feature_mode=snapshot["feature_mode"],
            x_label=snapshot.get("phase_x_state", "x"),
            y_label=snapshot.get("phase_y_state", "v"),
            min_fraction=float(snapshot.get("phase_min_fraction", 0.0)),
        )
        sweep_image = self._render_sweep_image_from_data(
            panel_side,
            panel_side,
            points=snapshot["sweep_points"],
            bounds=snapshot["sweep_bounds"],
            filter_values=snapshot["filter_values"],
            point_radius=snapshot["point_radius"],
            feature_mode=snapshot["feature_mode"],
            selected_value=value,
            title="Sweep overview",
            apply_zoom=True,
            zoom_views=snapshot["zoom_views"],
            x_label=str(snapshot.get("sweep_parameter", "parameter")),
            y_label=f"{snapshot.get('phase_x_state', 'x')} extrema",
        )

        frame = Image.new("RGB", VIDEO_FRAME_SIZE, "white")
        self._draw_video_header(frame, snapshot, value)
        frame.paste(basin_image, (panel_x0, panel_y))
        frame.paste(phase_image, (panel_x0 + panel_side + VIDEO_PANEL_GAP, panel_y))
        frame.paste(sweep_image, (panel_x0 + 2 * (panel_side + VIDEO_PANEL_GAP), panel_y))
        return frame

    def _create_video_progress_dialog(self, total_frames: int, path: Path):
        window = tk.Toplevel(self)
        window.title("Export Sweep Video")
        window.transient(self)
        window.resizable(False, False)
        window.protocol("WM_DELETE_WINDOW", lambda: None)

        frame = ttk.Frame(window, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text=f"Exporting {path.name}", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        status_label = ttk.Label(
            frame,
            text=f"0 / {total_frames} frames | 24 fps | {total_frames / VIDEO_FPS:.2f} s",
            width=54,
        )
        status_label.pack(anchor="w", pady=(8, 6))
        progress = ttk.Progressbar(frame, maximum=total_frames, length=420, mode="determinate")
        progress.pack(fill="x")
        return window, progress, status_label

    def _export_sweep_video_worker(self, path: Path, snapshot: dict, progress_queue: queue.Queue) -> None:
        writer = None
        try:
            import cv2
            import numpy as np

            details = snapshot["details"]
            codec = "MJPG" if path.suffix.lower() == ".avi" else "mp4v"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), VIDEO_FPS, VIDEO_FRAME_SIZE)
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer for: {path}")

            total = VIDEO_INTRO_FRAMES + len(details)
            intro_frame = self._sweep_video_intro_frame(snapshot)
            intro_bgr = cv2.cvtColor(np.array(intro_frame), cv2.COLOR_RGB2BGR)
            for intro_idx in range(1, VIDEO_INTRO_FRAMES + 1):
                writer.write(intro_bgr)
                if intro_idx == VIDEO_INTRO_FRAMES or intro_idx % VIDEO_FPS == 0:
                    progress_queue.put(("progress", intro_idx, total))

            for idx, detail in enumerate(details, start=1):
                frame = self._sweep_video_frame_from_snapshot(snapshot, detail)
                writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
                progress_queue.put(("progress", VIDEO_INTRO_FRAMES + idx, total))
            writer.release()
            writer = None
            progress_queue.put(("done", str(path), total, total / VIDEO_FPS))
        except Exception as exc:
            if writer is not None:
                writer.release()
            progress_queue.put(("error", str(exc)))

    def _poll_video_export_progress(self, progress_queue: queue.Queue, window, progress, status_label) -> None:
        try:
            while True:
                message = progress_queue.get_nowait()
                kind = message[0]
                if kind == "progress":
                    _, current, total = message
                    progress.configure(value=current)
                    status_label.configure(text=f"{current} / {total} frames | 24 fps | {total / VIDEO_FPS:.2f} s")
                elif kind == "done":
                    _, path, total, seconds = message
                    if window.winfo_exists():
                        window.destroy()
                    self._append_log(f"Sweep video exported: {path}")
                    messagebox.showinfo("Export Complete", f"Saved video:\n{path}\n\nFrames: {total}\nDuration: {seconds:.2f} s at {VIDEO_FPS} fps")
                    return
                elif kind == "error":
                    _, error_message = message
                    if window.winfo_exists():
                        window.destroy()
                    messagebox.showerror("Video Export Failed", error_message)
                    return
        except queue.Empty:
            pass
        if window.winfo_exists():
            self.after(100, lambda: self._poll_video_export_progress(progress_queue, window, progress, status_label))

    def export_sweep_video(self) -> None:
        if not self._require_export_tools():
            return
        if not self.sweep_details or not self.sweep_points:
            messagebox.showinfo("No Sweep Details", "Load a sweep result with saved details before exporting a video.")
            return

        default_name = f"{self._default_export_stem()}_sweep_video.mp4"
        initial_dir = self.current_result_dir if self.current_result_dir and self.current_result_dir.exists() else BASE_DIR
        path_raw = filedialog.asksaveasfilename(
            title="Export Sweep Video",
            initialdir=str(initial_dir),
            initialfile=default_name,
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("AVI video", "*.avi")],
        )
        if not path_raw:
            return

        try:
            import cv2
        except ImportError as exc:
            messagebox.showerror("Video Export Failed", f"OpenCV is required for MP4 export:\n{exc}")
            return

        path = Path(path_raw)
        snapshot = self._sweep_video_snapshot()
        details = snapshot["details"]
        if not details:
            messagebox.showinfo("No Sweep Details", "No saved sweep detail frames are available.")
            return

        progress_queue: queue.Queue = queue.Queue()
        total_frames = VIDEO_INTRO_FRAMES + len(details)
        progress_window, progress_bar, status_label = self._create_video_progress_dialog(total_frames, path)
        self._append_log(
            f"Exporting sweep video in background: {path} | frames={total_frames}, intro={VIDEO_INTRO_SECONDS}s, fps={VIDEO_FPS}, duration={total_frames / VIDEO_FPS:.2f}s"
        )
        thread = threading.Thread(
            target=self._export_sweep_video_worker,
            args=(path, snapshot, progress_queue),
            daemon=True,
        )
        thread.start()
        self.after(100, lambda: self._poll_video_export_progress(progress_queue, progress_window, progress_bar, status_label))

    def _redraw_probe_time_canvas(self) -> None:
        canvas = getattr(self, "probe_time_canvas", None)
        if canvas is None:
            return
        canvas.delete("all")
        if not self.probe_selected or not self.probe_benchmark or not self.probe_state_names:
            return

        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        margin_left = 58
        margin_top = 18
        margin_right = 18
        margin_bottom = 36
        plot_w = max(width - margin_left - margin_right, 1)
        plot_h = max(height - margin_top - margin_bottom, 1)

        all_rows = self.probe_selected + self.probe_benchmark
        xmin = min(row["t"] for row in all_rows)
        xmax = max(row["t"] for row in all_rows)
        ymin = min(min(row["values"]) for row in all_rows)
        ymax = max(max(row["values"]) for row in all_rows)
        if xmin == xmax:
            xmin -= 1.0
            xmax += 1.0
        if ymin == ymax:
            ymin -= 1.0
            ymax += 1.0
        pad_y = 0.05 * (ymax - ymin)
        ymin -= pad_y
        ymax += pad_y
        xmin, xmax, ymin, ymax = self._apply_zoom_view("probe_time", (xmin, xmax, ymin, ymax))
        self._remember_plot_view("probe_time", margin_left, margin_top, plot_w, plot_h, xmin, xmax, ymin, ymax)

        canvas.create_rectangle(margin_left, margin_top, margin_left + plot_w, margin_top + plot_h, outline="#555555")
        for tick in range(6):
            frac = tick / 5
            x = margin_left + frac * plot_w
            y = margin_top + frac * plot_h
            canvas.create_line(x, margin_top, x, margin_top + plot_h, fill="#eeeeee")
            canvas.create_line(margin_left, y, margin_left + plot_w, y, fill="#eeeeee")

        colors = ["#d1495b", "#00798c", "#edae49", "#7a5195", "#2f4b7c", "#59a14f"]

        def draw_run(rows: list[dict], state_index: int, color: str, dash=None) -> None:
            visible_rows = [row for row in rows if xmin <= row["t"] <= xmax]
            if not visible_rows:
                return
            step = max(1, len(visible_rows) // max(plot_w * 2, 800))
            coords: list[float] = []
            for row in visible_rows[::step]:
                px = margin_left + (row["t"] - xmin) / (xmax - xmin) * plot_w
                py = margin_top + (ymax - row["values"][state_index]) / (ymax - ymin) * plot_h
                coords.extend([px, py])
            if len(coords) >= 4:
                canvas.create_line(*coords, fill=color, width=1.4, dash=dash)

        for state_index, state_name in enumerate(self.probe_state_names):
            color = colors[state_index % len(colors)]
            draw_run(self.probe_selected, state_index, color)
            draw_run(self.probe_benchmark, state_index, color, dash=(4, 3))
            canvas.create_text(
                margin_left + 8 + 82 * (state_index % 4),
                margin_top + 10 + 14 * (state_index // 4),
                text=state_name,
                fill=color,
                font=("Segoe UI", 8, "bold"),
                anchor="w",
            )

        canvas.create_text(margin_left + plot_w - 8, margin_top + 10, text="solid: selected | dashed: benchmark", fill="#333333", font=("Segoe UI", 8), anchor="ne")
        self._draw_axis_labels(canvas, margin_left, margin_top, plot_w, plot_h, xmin, xmax, ymin, ymax)

    def _probe_phase_rows(self, rows: list[dict]) -> list[dict]:
        if not rows:
            return []
        if self.probe_phase_start is None or self.probe_phase_end is None:
            return rows

        eps = max(1e-9, abs(self.probe_phase_end - self.probe_phase_start) * 1e-9)
        filtered = [
            row
            for row in rows
            if self.probe_phase_start - eps <= row["t"] <= self.probe_phase_end + eps
        ]
        return filtered or rows

    def _redraw_probe_phase_canvas(self) -> None:
        canvas = getattr(self, "probe_phase_canvas", None)
        if canvas is None:
            return
        canvas.delete("all")
        if not self.probe_selected or not self.probe_benchmark or not self.probe_state_names:
            return

        try:
            x_index = self.probe_state_names.index(self.probe_x_state)
            y_index = self.probe_state_names.index(self.probe_y_state)
        except ValueError:
            x_index = 0
            y_index = 1 if len(self.probe_state_names) > 1 else 0

        selected_rows = self._probe_phase_rows(self.probe_selected)
        benchmark_rows = self._probe_phase_rows(self.probe_benchmark)
        if not selected_rows or not benchmark_rows:
            return

        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        margin_left = 58
        margin_top = 18
        margin_right = 18
        margin_bottom = 36
        plot_w = max(width - margin_left - margin_right, 1)
        plot_h = max(height - margin_top - margin_bottom, 1)

        all_rows = selected_rows + benchmark_rows
        xmin = min(row["values"][x_index] for row in all_rows)
        xmax = max(row["values"][x_index] for row in all_rows)
        ymin = min(row["values"][y_index] for row in all_rows)
        ymax = max(row["values"][y_index] for row in all_rows)
        if xmin == xmax:
            xmin -= 1.0
            xmax += 1.0
        if ymin == ymax:
            ymin -= 1.0
            ymax += 1.0
        pad_x = 0.04 * (xmax - xmin)
        pad_y = 0.04 * (ymax - ymin)
        xmin -= pad_x
        xmax += pad_x
        ymin -= pad_y
        ymax += pad_y
        xmin, xmax, ymin, ymax = self._apply_zoom_view("probe_phase", (xmin, xmax, ymin, ymax))
        self._remember_plot_view("probe_phase", margin_left, margin_top, plot_w, plot_h, xmin, xmax, ymin, ymax)

        canvas.create_rectangle(margin_left, margin_top, margin_left + plot_w, margin_top + plot_h, outline="#555555")
        for tick in range(6):
            frac = tick / 5
            x = margin_left + frac * plot_w
            y = margin_top + frac * plot_h
            canvas.create_line(x, margin_top, x, margin_top + plot_h, fill="#eeeeee")
            canvas.create_line(margin_left, y, margin_left + plot_w, y, fill="#eeeeee")

        def draw_phase(rows: list[dict], color: str, dash=None) -> None:
            step = max(1, len(rows) // 1800)
            coords: list[float] = []
            for row in rows[::step]:
                x_value = row["values"][x_index]
                y_value = row["values"][y_index]
                px = margin_left + (x_value - xmin) / (xmax - xmin) * plot_w
                py = margin_top + (ymax - y_value) / (ymax - ymin) * plot_h
                coords.extend([px, py])
            if len(coords) >= 4:
                canvas.create_line(*coords, fill=color, width=1.5, dash=dash)

        draw_phase(selected_rows, "#d1495b")
        draw_phase(benchmark_rows, "#111111", dash=(4, 3))
        canvas.create_text(margin_left + plot_w - 8, margin_top + 10, text="red: selected | black dashed: benchmark", fill="#333333", font=("Segoe UI", 8), anchor="ne")
        self._draw_axis_labels(canvas, margin_left, margin_top, plot_w, plot_h, xmin, xmax, ymin, ymax)

    def _format_axis_tick(self, value: float, span: float) -> str:
        if not math.isfinite(value):
            return str(value)
        span = abs(span)
        if not math.isfinite(span) or span <= 0:
            return f"{value:.6g}"

        step = span / 2
        decimals = max(0, min(10, int(math.ceil(-math.log10(step))) + 2))
        if abs(value) >= 1.0e6 and decimals > 0:
            text = f"{value:.8g}"
        else:
            text = f"{value:.{decimals}f}"
            if "." in text:
                text = text.rstrip("0").rstrip(".")
        return "0" if text in ("", "-0") else text

    def _draw_axis_labels(
        self,
        canvas: tk.Canvas,
        margin_left: int,
        margin_top: int,
        plot_w: int,
        plot_h: int,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        x_label: str = "",
        y_label: str = "",
    ) -> None:
        text_color = "#333333"
        x_span = xmax - xmin
        y_span = ymax - ymin
        for tick in range(3):
            frac = tick / 2
            x_value = xmin + frac * (xmax - xmin)
            x = margin_left + frac * plot_w
            canvas.create_line(x, margin_top + plot_h, x, margin_top + plot_h + 4, fill=text_color)
            canvas.create_text(
                x,
                margin_top + plot_h + 18,
                text=self._format_axis_tick(x_value, x_span),
                fill=text_color,
                font=("Segoe UI", 8),
            )

            y_value = ymax - frac * (ymax - ymin)
            y = margin_top + frac * plot_h
            canvas.create_line(margin_left - 4, y, margin_left, y, fill=text_color)
            canvas.create_text(
                margin_left - 8,
                y,
                text=self._format_axis_tick(y_value, y_span),
                fill=text_color,
                font=("Segoe UI", 8),
                anchor="e",
            )
        if x_label:
            canvas.create_text(
                margin_left + plot_w,
                margin_top + plot_h + 34,
                text=x_label,
                fill=text_color,
                font=("Segoe UI", 8, "bold"),
                anchor="e",
            )
        if y_label:
            canvas.create_text(
                8,
                margin_top + 4,
                text=y_label,
                fill=text_color,
                font=("Segoe UI", 8, "bold"),
                anchor="w",
            )

    def _draw_image_fallback(self, image_path: Path) -> None:
        self.basin_canvas.delete("all")
        try:
            image = tk.PhotoImage(file=str(image_path))
        except tk.TclError as exc:
            self.basin_info.configure(text=f"Basin image could not be loaded: {exc}")
            return

        image = self._fit_image_to_canvas(image, self.basin_canvas)
        self.current_image = image
        self.basin_canvas.create_image(0, 0, image=self.current_image, anchor="nw")

    def _label_color(self, label: int) -> str:
        if label == 0:
            return "#5a5a5a"
        hue = (label * 0.61803398875) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
        return f"#{round(255 * red):02x}{round(255 * green):02x}{round(255 * blue):02x}"

    def _fit_image_to_canvas(self, image: tk.PhotoImage, canvas: tk.Canvas) -> tk.PhotoImage:
        max_width = max(canvas.winfo_width(), 1)
        max_height = max(canvas.winfo_height(), 1)
        upsample = max(1, min(max_width // image.width(), max_height // image.height()))
        if upsample > 1:
            image = image.zoom(upsample, upsample)

        downsample = max(
            1,
            (image.width() + max_width - 1) // max_width,
            (image.height() + max_height - 1) // max_height,
        )
        if downsample > 1:
            image = image.subsample(downsample, downsample)
        return image

    def open_result_dir(self) -> None:
        if not self.current_result_dir or not self.current_result_dir.exists():
            messagebox.showinfo("No Result Folder", "No result folder is available yet.")
            return
        os.startfile(self.current_result_dir)  # type: ignore[attr-defined]

    def _on_close(self) -> None:
        self._terminate_backend_server(force=True)
        self.destroy()

    def _append_log(self, message: str) -> None:
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")


if __name__ == "__main__":
    app = BasinGuiApp()
    app.mainloop()
