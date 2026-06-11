using Dates
using DelimitedFiles
using DiffEqGPU
using OrdinaryDiffEq
using StaticArrays
using TOML

function default_duffing(u, p, t)
    x = u[1]
    v = u[2]
    U = u[3]
    w = p[1]
    F0 = p[2]
    d = p[3]
    r1 = p[4]
    r2 = p[5]
    r3 = p[6]

    T = eltype(u)
    return SVector{3, T}(
        v,
        -d * v - r1 * U + T(0.5) * x - T(0.5) * x^3 + F0 * cos(w * t),
        r2 * v - r3 * U,
    )
end


function default_duffing_phase_scaled(u, p, tau)
    x = u[1]
    v = u[2]
    U = u[3]
    w = p[1]
    F0 = p[2]
    d = p[3]
    r1 = p[4]
    r2 = p[5]
    r3 = p[6]

    T = eltype(u)
    invw = inv(w)
    t = tau * invw
    return SVector{3, T}(
        v * invw,
        (-d * v - r1 * U + T(0.5) * x - T(0.5) * x^3 + F0 * cos(w * t)) * invw,
        (r2 * v - r3 * U) * invw,
    )
end


function ensure_cuda_loaded()
    if !isdefined(@__MODULE__, :CUDA)
        Core.eval(@__MODULE__, :(using CUDA))
    end
end


const RK4_EXTREMA_CUDA_LOADED = Ref(false)


function ensure_rk4_extrema_cuda_loaded()
    if !RK4_EXTREMA_CUDA_LOADED[]
        include(joinpath(@__DIR__, "rk4_extrema_cuda.jl"))
        RK4_EXTREMA_CUDA_LOADED[] = true
    end
end


function get_precision_type(name::AbstractString)
    lower = lowercase(String(name))
    lower == "float32" && return Float32
    lower == "float64" && return Float64
    error("Unknown precision: $name")
end


function get_state_index(state_names::Vector{String}, target::String)
    idx = findfirst(==(target), state_names)
    idx === nothing && error("State '$target' was not found.")
    return idx
end


function normalize_expr_string(text::String)
    return replace(strip(text), r"\s+" => "")
end


function matches_default_duffing(state_names::Vector{String}, param_names::Vector{String}, equations::Vector{String})
    default_states = ["x", "v", "U"]
    default_params = ["w", "F0", "d", "r1", "r2", "r3"]
    default_equations = [
        "v",
        "-d*v-r1*U+0.5*x-0.5*x^3+F0*cos(w*t)",
        "r2*v-r3*U",
    ]

    return state_names == default_states &&
           param_names == default_params &&
           normalize_expr_string.(equations) == default_equations
end


struct ModelValidationError <: Exception
    message::String
end

Base.showerror(io::IO, err::ModelValidationError) = print(io, err.message)
model_validation_error(message::String) = throw(ModelValidationError(message))


const GPU_SAFE_RHS_FUNCTIONS = Set{Symbol}([
    :+, :-, :*, :/, :^,
    :sin, :cos, :tan,
    :asin, :acos, :atan,
    :sinh, :cosh, :tanh,
    :exp, :log, :log10,
    :sqrt, :abs,
    :min, :max,
    :ifelse,
    :<, :<=, :>, :>=, :(==), :(!=),
])

const GPU_SAFE_RHS_CONSTANTS = Set{Symbol}([:pi, :π])


function validate_model_identifiers(state_names::Vector{String}, param_names::Vector{String})
    identifier_re = r"^[A-Za-z_][A-Za-z0-9_]*$"
    all_names = vcat(state_names, param_names)
    reserved = union(GPU_SAFE_RHS_FUNCTIONS, GPU_SAFE_RHS_CONSTANTS, Set([:t, :phase, :T, :u, :p]))

    for name in all_names
        if !occursin(identifier_re, name)
            model_validation_error("Invalid model identifier '$name'. Use ASCII names starting with a letter or underscore, followed by letters, digits, or underscores.")
        end
        sym = Symbol(name)
        if sym in reserved
            model_validation_error("Invalid model identifier '$name'. It conflicts with a reserved function, constant, or internal variable.")
        end
    end

    if length(unique(all_names)) != length(all_names)
        model_validation_error("State and parameter names must be unique. Found duplicate names in: $(join(all_names, ", ")).")
    end
end


function validate_rhs_expr!(expr, allowed_symbols::Set{Symbol}, equation_text::String)
    if expr isa Symbol
        if !(expr in allowed_symbols || expr in GPU_SAFE_RHS_CONSTANTS)
            model_validation_error("Unsupported symbol '$expr' in equation '$equation_text'. Use only state names, parameter names, t, phase, and allowed constants pi/π.")
        end
        return
    end

    if expr isa Number
        return
    end

    if expr isa Expr
        if expr.head == :call
            fn = expr.args[1]
            if !(fn isa Symbol) || !(fn in GPU_SAFE_RHS_FUNCTIONS)
                model_validation_error("Unsupported function or operator '$fn' in equation '$equation_text'. Allowed functions: sin, cos, tan, exp, log, sqrt, abs, min, max, ifelse, and basic arithmetic.")
            end
            for arg in expr.args[2:end]
                validate_rhs_expr!(arg, allowed_symbols, equation_text)
            end
            return
        end

        if expr.head == :block
            for arg in expr.args
                validate_rhs_expr!(arg, allowed_symbols, equation_text)
            end
            return
        end

        model_validation_error("Unsupported expression form '$(expr.head)' in equation '$equation_text'. Equations must be single scalar right-hand-side expressions, not assignments, arrays, indexing, or control-flow blocks.")
    end

    model_validation_error("Unsupported token in equation '$equation_text'.")
end


function rhs_expr_uses_symbol(expr, target::Symbol)
    expr == target && return true
    expr isa Expr || return false
    return any(rhs_expr_uses_symbol(arg, target) for arg in expr.args)
end


function equations_use_symbol(equations::Vector{String}, target::Symbol)
    for equation_text in equations
        rhs_expr = Meta.parse(equation_text)
        rhs_expr_uses_symbol(rhs_expr, target) && return true
    end
    return false
end


function get_custom_time_argument(integration_cfg::Dict{String, Any})
    mode = lowercase(String(get(integration_cfg, "custom_time_argument", "time")))
    mode in ("time", "phase") || error("integration.custom_time_argument must be 'time' or 'phase'.")
    return mode
end


function is_pi_symbol_expr(expr)
    return expr isa Symbol && (expr == :pi || String(expr) == "\u03c0")
end


function is_numeric_two_expr(expr)
    return expr isa Number && Float64(expr) == 2.0
end


function is_two_pi_expr(expr)
    if expr isa Expr && expr.head == :call && expr.args[1] == :*
        factors = collect(expr.args[2:end])
        length(factors) == 2 || return false
        return (is_numeric_two_expr(factors[1]) && is_pi_symbol_expr(factors[2])) ||
               (is_pi_symbol_expr(factors[1]) && is_numeric_two_expr(factors[2]))
    end
    return false
end


function phase_frequency_symbol_from_period_expression(period_expression::String, param_names::Vector{String})
    param_symbols = Set(Symbol.(param_names))
    try
        expr = Meta.parse(period_expression)
        if expr isa Expr && expr.head == :call && expr.args[1] == :/ && length(expr.args) == 3
            numerator = expr.args[2]
            denominator = expr.args[3]
            if is_two_pi_expr(numerator) && denominator isa Symbol && denominator in param_symbols
                return denominator
            end
        end
    catch
    end

    normalized = replace(replace(replace(normalize_expr_string(period_expression), "(" => ""), ")" => ""), "\u03c0" => "pi")
    for name in param_names
        if normalized == "2*pi/$name" || normalized == "2pi/$name"
            return Symbol(name)
        end
    end
    return nothing
end


function replace_phase_frequency_time_expr(expr, frequency_symbol::Union{Symbol, Nothing})
    if frequency_symbol !== nothing && expr isa Expr && expr.head == :call && expr.args[1] == :*
        factors = collect(expr.args[2:end])
        if length(factors) == 2 &&
           ((factors[1] == frequency_symbol && factors[2] == :t) ||
            (factors[1] == :t && factors[2] == frequency_symbol))
            return :phase
        end
    end

    expr isa Expr || return expr
    return Expr(expr.head, map(arg -> replace_phase_frequency_time_expr(arg, frequency_symbol), expr.args)...)
end


function runtime_equations_for_custom_time_argument(equations::Vector{String}, param_names::Vector{String}, integration_cfg::Dict{String, Any})
    get_custom_time_argument(integration_cfg) == "phase" || return equations

    frequency_symbol = phase_frequency_symbol_from_period_expression(String(integration_cfg["period_expression"]), param_names)
    rewritten = String[]
    for equation_text in equations
        rhs_expr = Meta.parse(equation_text)
        rhs_runtime = replace_phase_frequency_time_expr(rhs_expr, frequency_symbol)
        if rhs_expr_uses_symbol(rhs_runtime, :t)
            if frequency_symbol === nothing
                error("integration.custom_time_argument = 'phase' can keep expressions like cos(w*t), but the frequency parameter could not be inferred from period_expression. Use period_expression = 2*pi/<frequency parameter> or write the equation with 'phase' explicitly.")
            end
            error("integration.custom_time_argument = 'phase' only supports physical time through $(frequency_symbol)*t, which is replaced internally by 'phase'. Remaining use of 't' in equation '$equation_text' is not phase-wrapped.")
        end
        push!(rewritten, string(rhs_runtime))
    end
    return rewritten
end


function select_runtime_custom_time_argument(equations::Vector{String}, param_names::Vector{String}, integration_cfg::Dict{String, Any}, solver_mode::String)
    configured = get_custom_time_argument(integration_cfg)
    configured == "phase" && return configured, integration_cfg, false

    if lowercase(solver_mode) == "rk4_full_extrema_custom" &&
       lowercase(String(get(integration_cfg, "period_mode", ""))) == "periodic"
        phase_cfg = copy(integration_cfg)
        phase_cfg["custom_time_argument"] = "phase"

        if equations_use_symbol(equations, :phase)
            return "phase", phase_cfg, true
        end

        try
            phase_equations = runtime_equations_for_custom_time_argument(equations, param_names, phase_cfg)
            if normalize_expr_string.(phase_equations) != normalize_expr_string.(equations)
                return "phase", phase_cfg, true
            end
        catch
            # Keep the explicit time mode when the model contains physical time
            # terms that cannot be rewritten to a wrapped forcing phase.
        end
    end

    return configured, integration_cfg, false
end


function typed_numeric_literals_expr(expr)
    if expr isa Bool
        return expr
    end

    if expr isa Number
        return :(T($expr))
    end

    if expr isa Symbol
        if expr == :π
            return :(T(pi))
        end
        if expr == :pi
            return :(T(pi))
        end
        return expr
    end

    if expr isa Expr
        if expr.head == :call && length(expr.args) == 3 && expr.args[1] == :^ && expr.args[3] isa Integer
            return Expr(:call, :^, typed_numeric_literals_expr(expr.args[2]), expr.args[3])
        end
        return Expr(expr.head, map(typed_numeric_literals_expr, expr.args)...)
    end

    return expr
end


function build_ode_function(state_names::Vector{String}, param_names::Vector{String}, equations::Vector{String})
    validate_model_identifiers(state_names, param_names)
    allowed_symbols = Set(Symbol.(vcat(state_names, param_names, ["t", "phase"])))
    nstates = length(state_names)
    rhs_exprs = Any[]
    for equation_text in equations
        try
            push!(rhs_exprs, Meta.parse(equation_text))
        catch err
            model_validation_error("Could not parse equation '$equation_text': $err")
        end
    end
    for (equation_text, rhs_expr) in zip(equations, rhs_exprs)
        validate_rhs_expr!(rhs_expr, allowed_symbols, equation_text)
    end
    assignments = Expr[]

    for (i, name) in enumerate(state_names)
        push!(assignments, :(local $(Symbol(name)) = u[$i]))
    end
    for (i, name) in enumerate(param_names)
        push!(assignments, :(local $(Symbol(name)) = p[$i]))
    end
    push!(assignments, :(local phase = t))

    svector_type = Expr(:curly, :SVector, nstates, :T)
    typed_rhs_exprs = typed_numeric_literals_expr.(rhs_exprs)
    return_expr = Expr(:call, svector_type, typed_rhs_exprs...)
    fname = Symbol("_custom_ode_", time_ns())
    function_expr = quote
        function $(fname)(u, p, t)
            T = eltype(u)
            $(assignments...)
            return $return_expr
        end
    end
    Core.eval(@__MODULE__, function_expr)
    return Base.invokelatest(getfield, @__MODULE__, fname)
end


function get_model_function(state_names::Vector{String}, param_names::Vector{String}, equations::Vector{String}, device::String; force_custom::Bool=false)
    if !force_custom && matches_default_duffing(state_names, param_names, equations)
        println("Model path: built-in Duffing model")
        return default_duffing, false
    end

    if force_custom && matches_default_duffing(state_names, param_names, equations)
        println("Model path: standard Duffing equations through generated custom model")
    else
        println("Model path: custom equations (generated GPU-safe Julia function)")
    end
    return build_ode_function(state_names, param_names, equations), true
end


function evaluate_scalar_expression(expr_str::String, param_names::Vector{String}, param_values::Vector{Float64})
    expr = Meta.parse(expr_str)
    assignments = [:(local $(Symbol(param_names[i])) = values[$i]) for i in eachindex(param_names)]
    return Core.eval(
        @__MODULE__,
        quote
            let values = $(param_values)
                $(assignments...)
                $expr
            end
        end,
    )
end


function resolve_time_settings(integration_cfg::Dict{String, Any}, param_names::Vector{String}, param_values::Vector{Float64}, ::Type{T}) where {T}
    period_mode = lowercase(String(integration_cfg["period_mode"]))
    if period_mode == "periodic"
        period = T(evaluate_scalar_expression(String(integration_cfg["period_expression"]), param_names, param_values))
        transient_periods = T(integration_cfg["transient_periods"])
        evaluation_periods = T(integration_cfg["evaluation_periods"])
        samples_per_period = Int(integration_cfg["samples_per_period"])
        dt = period / T(samples_per_period)
        save_dt = dt
        t_transient = transient_periods * period
        t_evaluation = evaluation_periods * period
        return period, t_transient, t_evaluation, dt, save_dt
    end

    period = T(0)
    t_transient = T(integration_cfg["t_transient"])
    t_evaluation = T(integration_cfg["t_evaluation"])
    dt = T(integration_cfg["dt"])
    save_dt = T(integration_cfg["save_dt"])
    return period, t_transient, t_evaluation, dt, save_dt
end


function derive_grid(plane_cfg::Dict{String, Any})
    x_min = Float64(plane_cfg["x_min"])
    x_max = Float64(plane_cfg["x_max"])
    y_min = Float64(plane_cfg["y_min"])
    y_max = Float64(plane_cfg["y_max"])
    grid_mode = lowercase(String(plane_cfg["grid_mode"]))

    if grid_mode == "manual"
        nx = Int(plane_cfg["nx"])
        ny = Int(plane_cfg["ny"])
        return nx, ny, x_min, x_max, y_min, y_max
    end

    n_target = max(2, Int(plane_cfg["n_target"]))
    lx = x_max - x_min
    ly = y_max - y_min
    ratio = lx / ly
    ny = max(2, round(Int, sqrt(n_target / ratio)))
    nx = max(2, round(Int, ratio * ny))
    return nx, ny, x_min, x_max, y_min, y_max
end


function generate_initial_conditions(state_defaults::Vector{T}, x_idx::Int, y_idx::Int, xs::Vector{T}, ys::Vector{T}) where {T}
    nstates = length(state_defaults)
    base = SVector{nstates, T}(Tuple(state_defaults))
    result = Vector{SVector{nstates, T}}(undef, length(xs) * length(ys))
    counter = 1

    for y in ys
        for x in xs
            mv = MVector{nstates, T}(base)
            mv[x_idx] = x
            mv[y_idx] = y
            result[counter] = SVector{nstates, T}(mv)
            counter += 1
        end
    end
    return result
end


function choose_cpu_solver(name::String)
    lname = lowercase(name)
    lname == "tsit5" && return Tsit5()
    lname == "vern9" && return Vern9()
    lname == "rosenbrock23" && return Rosenbrock23()
    lname == "rodas5p" && return Rodas5P()
    error("Unsupported CPU solver: $name")
end


function solve_stage_a_gpu(odefun, u0_list, pvec, t_transient, dt)
    CUDA.functional() || error("CUDA is not functional, so the GPU run is not possible.")
    prob = ODEProblem{false}(odefun, u0_list[1], (zero(eltype(u0_list[1])), t_transient), pvec)
    prob_func = (pr, i, repeat) -> remake(pr; u0=u0_list[i], p=pvec)
    ensemble = EnsembleProblem(prob; prob_func=prob_func)
    return solve(
        ensemble,
        GPUTsit5(),
        EnsembleGPUKernel(CUDA.CUDABackend());
        trajectories=length(u0_list),
        adaptive=false,
        dt=dt,
        saveat=t_transient,
        save_start=false,
    )
end


function solve_stage_a_sweep_scaled_gpu(u0_list, pvec_list, n_total::Int, tau_transient, dtau)
    CUDA.functional() || error("CUDA is not functional, so the batched GPU transient run is not possible.")
    T = eltype(u0_list[1])
    prob = ODEProblem{false}(default_duffing_phase_scaled, u0_list[1], (zero(T), tau_transient), pvec_list[1])
    prob_func = (pr, i, repeat) -> begin
        point_idx = ((i - 1) % n_total) + 1
        sweep_idx = ((i - 1) ÷ n_total) + 1
        remake(pr; u0=u0_list[point_idx], p=pvec_list[sweep_idx])
    end
    ensemble = EnsembleProblem(prob; prob_func=prob_func)
    return solve(
        ensemble,
        GPUTsit5(),
        EnsembleGPUKernel(CUDA.CUDABackend());
        trajectories=length(u0_list) * length(pvec_list),
        adaptive=false,
        dt=dtau,
        saveat=tau_transient,
        save_start=false,
    )
end


function solve_stage_b_gpu(odefun, u_trans_list, pvec, t_transient, t_evaluation, dt, save_dt)
    CUDA.functional() || error("CUDA is not functional, so the GPU run is not possible.")
    t_end = t_transient + t_evaluation
    prob = ODEProblem{false}(odefun, u_trans_list[1], (t_transient, t_end), pvec)
    prob_func = (pr, i, repeat) -> remake(pr; u0=u_trans_list[i], p=pvec)
    ensemble = EnsembleProblem(prob; prob_func=prob_func)
    return solve(
        ensemble,
        GPUTsit5(),
        EnsembleGPUKernel(CUDA.CUDABackend());
        trajectories=length(u_trans_list),
        adaptive=false,
        dt=dt,
        saveat=save_dt,
        save_start=false,
    )
end


function solve_single_trajectory_gpu(odefun, u0, pvec, t_end, dt, save_dt)
    CUDA.functional() || error("CUDA is not functional, so the GPU run is not possible.")
    T = eltype(u0)
    prob = ODEProblem{false}(odefun, u0, (zero(T), T(t_end)), pvec)
    ensemble = EnsembleProblem(prob; prob_func=(pr, _i, _repeat) -> pr)
    sol = solve(
        ensemble,
        GPUTsit5(),
        EnsembleGPUKernel(CUDA.CUDABackend());
        trajectories=1,
        adaptive=false,
        dt=T(dt),
        saveat=T(save_dt),
        save_start=false,
    )
    sol_i = sol[1]
    ts = T[zero(T)]
    us = typeof(u0)[u0]
    append!(ts, sol_i.t)
    append!(us, sol_i.u)
    return (t=ts, u=us)
end


function solve_stage_a(odefun, u0_list, pvec, t_transient, dt, solver_mode, solver_name, device, abstol, reltol)
    prob = ODEProblem{false}(odefun, u0_list[1], (zero(eltype(u0_list[1])), t_transient), pvec)
    prob_func = (pr, i, repeat) -> remake(pr; u0=u0_list[i], p=pvec)
    ensemble = EnsembleProblem(prob; prob_func=prob_func)

    if device == "gpu"
        lowercase(solver_mode) == "fixed" || error("GPU runs currently support fixed-step mode only.")
        lowercase(solver_name) == "tsit5" || error("GPU runs currently support Tsit5 only.")
        Base.invokelatest(ensure_cuda_loaded)
        return Base.invokelatest(solve_stage_a_gpu, odefun, u0_list, pvec, t_transient, dt)
    end

    solver = choose_cpu_solver(solver_name)
    if lowercase(solver_mode) == "fixed"
        return solve(
            ensemble,
            solver,
            EnsembleThreads();
            trajectories=length(u0_list),
            adaptive=false,
            dt=dt,
            saveat=t_transient,
            save_start=false,
        )
    end

    return solve(
        ensemble,
        solver,
        EnsembleThreads();
        trajectories=length(u0_list),
        adaptive=true,
        abstol=abstol,
        reltol=reltol,
        saveat=t_transient,
        save_start=false,
    )
end


function solve_stage_b(odefun, u_trans_list, pvec, t_transient, t_evaluation, dt, save_dt, solver_mode, solver_name, device, abstol, reltol)
    t_end = t_transient + t_evaluation
    prob = ODEProblem{false}(odefun, u_trans_list[1], (t_transient, t_end), pvec)
    prob_func = (pr, i, repeat) -> remake(pr; u0=u_trans_list[i], p=pvec)
    ensemble = EnsembleProblem(prob; prob_func=prob_func)

    if device == "gpu"
        Base.invokelatest(ensure_cuda_loaded)
        return Base.invokelatest(solve_stage_b_gpu, odefun, u_trans_list, pvec, t_transient, t_evaluation, dt, save_dt)
    end

    solver = choose_cpu_solver(solver_name)
    if lowercase(solver_mode) == "fixed"
        return solve(
            ensemble,
            solver,
            EnsembleThreads();
            trajectories=length(u_trans_list),
            adaptive=false,
            dt=dt,
            saveat=save_dt,
            save_start=false,
        )
    end

    return solve(
        ensemble,
        solver,
        EnsembleThreads();
        trajectories=length(u_trans_list),
        adaptive=true,
        abstol=abstol,
        reltol=reltol,
        saveat=save_dt,
        save_start=false,
    )
end


function collect_extrema(sol_i, observable_idx::Int, zero_cross_idx::Int, max_extrema::Int, eps)
    us = sol_i.u
    values = Float32[]

    for k in 1:(length(us) - 1)
        z1 = Float64(us[k][zero_cross_idx])
        z2 = Float64(us[k + 1][zero_cross_idx])

        if eps > 0
            z1 = abs(z1) < eps ? 0.0 : z1
            z2 = abs(z2) < eps ? 0.0 : z2
        end

        if z1 == 0 || z1 * z2 < 0
            x1 = Float64(us[k][observable_idx])
            x2 = Float64(us[k + 1][observable_idx])
            denom = z2 - z1
            alpha = denom == 0 ? 0.0 : (-z1 / denom)
            push!(values, Float32(x1 + alpha * (x2 - x1)))
            if length(values) >= max_extrema
                break
            end
        end
    end

    return values
end


function stream_extrema_update!(values::Vector{Float32}, prev_u, curr_u, observable_idx::Int, zero_cross_idx::Int, max_extrema::Int, eps)
    length(values) >= max_extrema && return

    z1 = Float64(prev_u[zero_cross_idx])
    z2 = Float64(curr_u[zero_cross_idx])

    if eps > 0
        z1 = abs(z1) < eps ? 0.0 : z1
        z2 = abs(z2) < eps ? 0.0 : z2
    end

    if z1 == 0 || z1 * z2 < 0
        x1 = Float64(prev_u[observable_idx])
        x2 = Float64(curr_u[observable_idx])
        denom = z2 - z1
        alpha = denom == 0 ? 0.0 : (-z1 / denom)
        push!(values, Float32(x1 + alpha * (x2 - x1)))
    end
end


function fingerprint(extrema::Vector{Float32}; tol::Float32, K::Int)
    ex = filter(isfinite, extrema)
    isempty(ex) && return ()
    sort!(ex)
    kuse = min(K, length(ex))
    return ntuple(i -> round(Int, ex[i] / tol), kuse)
end


function classify_extrema_all(extrema_all::Vector{Vector{Float32}}, tol::Float32, k_fp::Int)
    sig_to_label = Dict{Tuple{Vararg{Int}}, Int32}()
    labels_vec = Vector{Int32}(undef, length(extrema_all))
    next_label = Int32(0)
    counts = Dict{Int32, Int}()

    for i in eachindex(extrema_all)
        extrema = extrema_all[i]
        sig = fingerprint(extrema; tol=tol, K=k_fp)
        if sig == ()
            labels_vec[i] = Int32(0)
            counts[Int32(0)] = get(counts, Int32(0), 0) + 1
            continue
        end

        label = get(sig_to_label, sig, Int32(0))
        if label == 0
            next_label += 1
            sig_to_label[sig] = next_label
            label = next_label
        end
        labels_vec[i] = label
        counts[label] = get(counts, label, 0) + 1
    end

    return labels_vec, counts, Int(next_label)
end


function invalid_extrema_features()
    return (amp=Float32(NaN), mean=Float32(NaN), std=Float32(NaN), k=UInt8(0), pattern=0f0)
end


function extrema_features(extrema::Vector{Float32})
    finite = Float32[]
    for x in extrema
        isfinite(x) && push!(finite, x)
    end

    n = length(finite)
    n == 0 && return invalid_extrema_features()

    minx = Float32(Inf)
    maxx = Float32(-Inf)
    sumx = 0f0
    sumsq = 0f0
    for x in finite
        minx = min(minx, x)
        maxx = max(maxx, x)
        sumx += x
        sumsq += x * x
    end

    mean = sumx / Float32(n)
    amp = maxx - minx
    variance = max(sumsq / Float32(n) - mean * mean, 0f0)
    std = sqrt(variance)

    sort!(finite)
    inv_std = 1f0 / max(std, 1f-6)
    pat_raw = 0f0
    weight = 1f0
    for x in finite
        pat_raw += weight * ((x - mean) * inv_std)
        weight *= 0.75f0
    end
    pattern = tanh(pat_raw / 3f0)

    return (
        amp=amp,
        mean=mean,
        std=std,
        k=UInt8(min(n, typemax(UInt8))),
        pattern=pattern,
    )
end


struct SweepExtremaRow
    parameter_value::Float64
    label::Int32
    class_count::Int32
    class_fraction::Float32
    class_rank::Int32
    x_extreme::Float32
    amp::Float32
    mean::Float32
    std::Float32
    k_extrema::UInt8
    pattern::Float32
end


@inline norm01(x::Float32, lo::Float32, hi::Float32) =
    (hi > lo) ? clamp((x - lo) / (hi - lo), 0f0, 1f0) : 0.5f0


@inline function lab_inverse_f(t::Float64)
    delta = 6.0 / 29.0
    return t > delta ? t^3 : 3.0 * delta^2 * (t - 4.0 / 29.0)
end


@inline function linear_rgb_to_srgb_byte(channel::Float64)
    srgb = channel <= 0.0031308 ? 12.92 * channel : 1.055 * channel^(1.0 / 2.4) - 0.055
    return round(Int, 255.0 * clamp(srgb, 0.0, 1.0))
end


function lchab_to_rgb_bytes(lightness::Float64, chroma::Float64, hue_deg::Float64)
    hue_rad = hue_deg * pi / 180.0
    lab_a = chroma * cos(hue_rad)
    lab_b = chroma * sin(hue_rad)

    fy = (lightness + 16.0) / 116.0
    fx = fy + lab_a / 500.0
    fz = fy - lab_b / 200.0

    x = 0.95047 * lab_inverse_f(fx)
    y = 1.00000 * lab_inverse_f(fy)
    z = 1.08883 * lab_inverse_f(fz)

    r_lin = 3.2406 * x - 1.5372 * y - 0.4986 * z
    g_lin = -0.9689 * x + 1.8758 * y + 0.0415 * z
    b_lin = 0.0557 * x - 0.2040 * y + 1.0570 * z

    return (
        linear_rgb_to_srgb_byte(r_lin),
        linear_rgb_to_srgb_byte(g_lin),
        linear_rgb_to_srgb_byte(b_lin),
    )
end


function feature_color_rgb(
    amp::Float32,
    k_extrema::UInt8,
    mean::Float32,
    std::Float32,
    amin::Float32,
    amax::Float32,
    _mmin::Float32,
    _mmax::Float32,
    smin::Float32,
    smax::Float32;
    kmax::Int=12,
    pattern::Float32=0f0,
)
    if !isfinite(amp) || !isfinite(mean) || !isfinite(std) || k_extrema == 0x00
        return (51, 51, 51)
    end

    a01 = norm01(log1p(max(amp, 0f0)), log1p(max(amin, 0f0)), log1p(max(amax, 0f0)))
    s01 = norm01(log1p(max(std, 0f0)), log1p(max(smin, 0f0)), log1p(max(smax, 0f0)))
    k01 = clamp(Float32(k_extrema) / Float32(kmax), 0f0, 1f0)

    # Match the original sweep coloring: sector by well, hue by pattern,
    # lightness by amplitude, chroma by extrema count and spread.
    d_left = abs(mean + 1f0)
    d_mid = abs(mean)
    d_right = abs(mean - 1f0)
    well = (d_left < d_mid) ? ((d_left < d_right) ? 1 : 3) : ((d_mid < d_right) ? 2 : 3)
    h0, h_width = well == 1 ? (225.0, 110.0) : well == 2 ? (125.0, 110.0) : (25.0, 110.0)
    pattern01 = 0.5f0 + 0.5f0 * clamp(pattern, -1f0, 1f0)

    hue = h0 - h_width / 2.0 + h_width * Float64(pattern01)
    lightness = 12.0 + 82.0 * Float64(a01)
    chroma = 35.0 + 90.0 * Float64(k01) + 55.0 * Float64(s01)
    return lchab_to_rgb_bytes(lightness, chroma, hue)
end


function label_to_rgb(label::Int)
    if label == 0
        return (90, 90, 90)
    end

    h = mod(label * 0.61803398875, 1.0)
    s = 0.72
    v = 0.95
    i = floor(Int, h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    r, g, b =
        if i % 6 == 0
            (v, t, p)
        elseif i % 6 == 1
            (q, v, p)
        elseif i % 6 == 2
            (p, v, t)
        elseif i % 6 == 3
            (p, q, v)
        elseif i % 6 == 4
            (t, p, v)
        else
            (v, p, q)
        end

    return (round(Int, 255 * r), round(Int, 255 * g), round(Int, 255 * b))
end


function write_ppm(path::String, labels::AbstractMatrix{Int32})
    ny, nx = size(labels)
    open(path, "w") do io
        write(io, "P6\n$nx $ny\n255\n")
        for row in ny:-1:1
            for col in 1:nx
                r, g, b = label_to_rgb(Int(labels[row, col]))
                write(io, UInt8(r), UInt8(g), UInt8(b))
            end
        end
    end
end


function class_feature_summary(labels_vec, extrema_all)
    feature_template = invalid_extrema_features()
    label_features = Dict{Int32, typeof(feature_template)}()
    amin = Float32(Inf)
    amax = Float32(-Inf)
    mmin = Float32(Inf)
    mmax = Float32(-Inf)
    smin = Float32(Inf)
    smax = Float32(-Inf)

    n = min(length(labels_vec), length(extrema_all))
    for i in 1:n
        feats = extrema_features(extrema_all[i])
        if feats.k > 0
            amin = min(amin, feats.amp)
            amax = max(amax, feats.amp)
            mmin = min(mmin, feats.mean)
            mmax = max(mmax, feats.mean)
            smin = min(smin, feats.std)
            smax = max(smax, feats.std)
        end

        label = labels_vec[i]
        if label > 0 && feats.k > 0 && !haskey(label_features, label)
            label_features[label] = feats
        end
    end

    bounds = finalize_sweep_color_bounds(amin, amax, mmin, mmax, smin, smax)
    return label_features, bounds
end


function write_class_stats(path::String, counts::Dict{Int32, Int}, total::Int; labels_vec=nothing, extrema_all=nothing)
    labels = sort(collect(keys(counts)))
    has_feature_colors = labels_vec !== nothing && extrema_all !== nothing
    feature_template = invalid_extrema_features()
    label_features = Dict{Int32, typeof(feature_template)}()
    bounds = finalize_sweep_color_bounds(0f0, 1f0, -1f0, 1f0, 0f0, 1f0)
    if has_feature_colors
        label_features, bounds = class_feature_summary(labels_vec, extrema_all)
    end

    open(path, "w") do io
        println(io, "label,count,fraction,color_r,color_g,color_b,amp,mean,std,k_extrema,pattern")
        for label in labels
            count = counts[label]
            fraction = count / total
            feats = get(label_features, label, feature_template)
            if has_feature_colors && feats.k > 0
                r, g, b = feature_color_rgb(
                    feats.amp,
                    feats.k,
                    feats.mean,
                    feats.std,
                    bounds.amin,
                    bounds.amax,
                    bounds.mmin,
                    bounds.mmax,
                    bounds.smin,
                    bounds.smax;
                    pattern=feats.pattern,
                )
            else
                r, g, b = label == 0 ? (51, 51, 51) : label_to_rgb(Int(label))
            end
            println(
                io,
                "$(label),$(count),$(fraction),$r,$g,$b,$(Float64(feats.amp)),$(Float64(feats.mean)),$(Float64(feats.std)),$(Int(feats.k)),$(Float64(feats.pattern))",
            )
        end
    end
end


const PHASE_BINARY_MAGIC = UInt8[0x42, 0x41, 0x53, 0x50, 0x48, 0x30, 0x31, 0x00] # "BASPH01\0"


function select_phase_sample_indices(labels_vec::Vector{Int32}; max_labels::Int=20, samples_per_label::Int=5)
    label_indices = Dict{Int32, Vector{Int}}()
    selected = Tuple{Int32, Int32, Int}[]

    for i in eachindex(labels_vec)
        label = labels_vec[i]
        if label <= 0
            continue
        end
        push!(get!(label_indices, label, Int[]), i)
    end

    ranked_labels = sort(collect(keys(label_indices)); by=label -> (-length(label_indices[label]), Int(label)))
    for label in Iterators.take(ranked_labels, max_labels)
        indices = label_indices[label]
        n_pick = min(samples_per_label, length(indices))
        n_pick <= 0 && continue

        used_positions = Set{Int}()
        for pick in 1:n_pick
            pos = n_pick == 1 ? 1 : round(Int, 1 + (pick - 1) * (length(indices) - 1) / (n_pick - 1))
            while pos in used_positions && pos < length(indices)
                pos += 1
            end
            while pos in used_positions && pos > 1
                pos -= 1
            end
            push!(used_positions, pos)
            push!(selected, (Int32(length(selected) + 1), label, indices[pos]))
        end
    end
    return selected
end


function write_phase_samples(path::String, sol_b, labels_vec::Vector{Int32}, phase_x_idx::Int, phase_y_idx::Int; max_labels::Int=20, samples_per_label::Int=5)
    selected = select_phase_sample_indices(labels_vec; max_labels=max_labels, samples_per_label=samples_per_label)

    open(path, "w") do io
        buffer = IOBuffer()
        println(buffer, "sample_id,label,t,x,y")
        for (sample_id, label, sol_idx) in selected
            sol_i = sol_b[sol_idx]

            for k in eachindex(sol_i.u)
                u = sol_i.u[k]
                t = sol_i.t[k]
                x = u[phase_x_idx]
                y = u[phase_y_idx]
                if isfinite(Float64(x)) && isfinite(Float64(y))
                    print(buffer, sample_id)
                    print(buffer, ',')
                    print(buffer, Int(label))
                    print(buffer, ',')
                    print(buffer, Float64(t))
                    print(buffer, ',')
                    print(buffer, Float64(x))
                    print(buffer, ',')
                    println(buffer, Float64(y))
                    if position(buffer) >= 1_048_576
                        write(io, take!(buffer))
                    end
                end
            end
        end
        write(io, take!(buffer))
    end

    return length(selected)
end


function write_phase_samples_binary(path::String, sol_b, labels_vec::Vector{Int32}, phase_x_idx::Int, phase_y_idx::Int; max_labels::Int=20, samples_per_label::Int=5)
    selected = select_phase_sample_indices(labels_vec; max_labels=max_labels, samples_per_label=samples_per_label)

    open(path, "w") do io
        write(io, PHASE_BINARY_MAGIC)
        write(io, UInt32(length(selected)))
        for (sample_id, label, sol_idx) in selected
            sol_i = sol_b[sol_idx]
            finite_count = UInt32(0)
            for u in sol_i.u
                x = u[phase_x_idx]
                y = u[phase_y_idx]
                if isfinite(Float64(x)) && isfinite(Float64(y))
                    finite_count += UInt32(1)
                end
            end

            write(io, Int32(sample_id))
            write(io, Int32(label))
            write(io, finite_count)
            for u in sol_i.u
                x = u[phase_x_idx]
                y = u[phase_y_idx]
                if isfinite(Float64(x)) && isfinite(Float64(y))
                    write(io, Float32(x))
                    write(io, Float32(y))
                end
            end
        end
    end

    return length(selected)
end


function write_phase_samples_selected(path::String, selected, phase_solutions, phase_x_idx::Int, phase_y_idx::Int)
    open(path, "w") do io
        buffer = IOBuffer()
        println(buffer, "sample_id,label,t,x,y")
        for (j, (sample_id, label, _sol_idx)) in enumerate(selected)
            sol_i = phase_solutions[j]
            for k in eachindex(sol_i.u)
                u = sol_i.u[k]
                t = sol_i.t[k]
                x = u[phase_x_idx]
                y = u[phase_y_idx]
                if isfinite(Float64(x)) && isfinite(Float64(y))
                    print(buffer, sample_id)
                    print(buffer, ',')
                    print(buffer, Int(label))
                    print(buffer, ',')
                    print(buffer, Float64(t))
                    print(buffer, ',')
                    print(buffer, Float64(x))
                    print(buffer, ',')
                    println(buffer, Float64(y))
                    if position(buffer) >= 1_048_576
                        write(io, take!(buffer))
                    end
                end
            end
        end
        write(io, take!(buffer))
    end
    return length(selected)
end


function write_phase_samples_binary_selected(path::String, selected, phase_solutions, phase_x_idx::Int, phase_y_idx::Int)
    open(path, "w") do io
        write(io, PHASE_BINARY_MAGIC)
        write(io, UInt32(length(selected)))
        for (j, (sample_id, label, _sol_idx)) in enumerate(selected)
            sol_i = phase_solutions[j]
            finite_count = UInt32(0)
            for u in sol_i.u
                x = u[phase_x_idx]
                y = u[phase_y_idx]
                if isfinite(Float64(x)) && isfinite(Float64(y))
                    finite_count += UInt32(1)
                end
            end

            write(io, Int32(sample_id))
            write(io, Int32(label))
            write(io, finite_count)
            for u in sol_i.u
                x = u[phase_x_idx]
                y = u[phase_y_idx]
                if isfinite(Float64(x)) && isfinite(Float64(y))
                    write(io, Float32(x))
                    write(io, Float32(y))
                end
            end
        end
    end
    return length(selected)
end


function write_result_phase_samples(path::String, result, phase_x_idx::Int, phase_y_idx::Int; binary::Bool=false)
    if hasproperty(result, :phase_sample_solutions) && result.phase_sample_solutions !== nothing
        selected = hasproperty(result, :phase_sample_selected) ? result.phase_sample_selected : Tuple{Int32, Int32, Int}[]
        if binary
            return write_phase_samples_binary_selected(path, selected, result.phase_sample_solutions, phase_x_idx, phase_y_idx)
        end
        return write_phase_samples_selected(path, selected, result.phase_sample_solutions, phase_x_idx, phase_y_idx)
    end

    if binary
        return write_phase_samples_binary(path, result.sol_b, result.labels_vec, phase_x_idx, phase_y_idx)
    end
    return write_phase_samples(path, result.sol_b, result.labels_vec, phase_x_idx, phase_y_idx)
end


function rk4_step_local(odefun, u, p, t, dt)
    half_dt = dt / 2
    k1 = odefun(u, p, t)
    k2 = odefun(u + half_dt * k1, p, t + half_dt)
    k3 = odefun(u + half_dt * k2, p, t + half_dt)
    k4 = odefun(u + dt * k3, p, t + dt)
    return u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
end


function rk4_step_local_phase_argument(odefun, u, p, phase, dt, dphase)
    half_dt = dt / 2
    half_dphase = dphase / 2
    k1 = odefun(u, p, phase)
    k2 = odefun(u + half_dt * k1, p, wrap_phase_local(phase + half_dphase))
    k3 = odefun(u + half_dt * k2, p, wrap_phase_local(phase + half_dphase))
    k4 = odefun(u + dt * k3, p, wrap_phase_local(phase + dphase))
    return u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4), wrap_phase_local(phase + dphase)
end


@inline function wrap_phase_local(phi)
    T = typeof(phi)
    two_pi = T(2 * pi)
    wrapped = phi >= two_pi ? phi - two_pi : phi
    return wrapped < zero(T) ? wrapped + two_pi : wrapped
end


@inline function default_duffing_phase_rhs(u, p, phase)
    x = u[1]
    v = u[2]
    U = u[3]
    F0 = p[2]
    d = p[3]
    r1 = p[4]
    r2 = p[5]
    r3 = p[6]
    T = eltype(u)
    return SVector{3, T}(
        v,
        -d * v - r1 * U + T(0.5) * x - T(0.5) * x^3 + F0 * cos(phase),
        r2 * v - r3 * U,
    )
end


function rk4_step_default_duffing_phase_local(u, p, phase, dt)
    half_dt = dt / 2
    dphase = p[1] * dt
    half_dphase = dphase / 2
    k1 = default_duffing_phase_rhs(u, p, phase)
    k2 = default_duffing_phase_rhs(u + half_dt * k1, p, wrap_phase_local(phase + half_dphase))
    k3 = default_duffing_phase_rhs(u + half_dt * k2, p, wrap_phase_local(phase + half_dphase))
    k4 = default_duffing_phase_rhs(u + dt * k3, p, wrap_phase_local(phase + dphase))
    return u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4), wrap_phase_local(phase + dphase)
end


function phase_from_period(t_start, period, ::Type{T}) where {T}
    return T(mod(2 * pi * Float64(t_start) / Float64(period), 2 * pi))
end


function phase_from_time(pvec, t_start)
    T = eltype(pvec)
    return T(mod(Float64(pvec[1]) * Float64(t_start), 2 * pi))
end


function solve_trajectory_rk4_phase_wrapped_default(
    u0,
    pvec,
    t_start,
    t_duration,
    dt,
    save_dt,
    ;
    save_start::Bool=true,
)
    abs(Float64(dt) - Float64(save_dt)) <= 10 * eps(Float64(dt)) || error("Wrapped RK4 display trajectories currently require dt == save_dt.")
    T = eltype(u0)
    step_dt = T(dt)
    n_steps = max(0, round(Int, Float64(t_duration) / Float64(dt)))
    ts = Float64[]
    us = typeof(u0)[]
    u = u0
    phase = phase_from_time(pvec, t_start)
    if save_start
        push!(ts, Float64(t_start))
        push!(us, u)
    end
    for step in 1:n_steps
        u, phase = rk4_step_default_duffing_phase_local(u, pvec, phase, step_dt)
        push!(ts, Float64(t_start) + step * Float64(dt))
        push!(us, u)
    end
    return (t=ts, u=us)
end


function solve_trajectory_rk4_local(
    odefun,
    u0,
    pvec,
    t_start,
    t_duration,
    dt,
    save_dt,
    ;
    save_start::Bool=true,
)
    T = eltype(u0)
    t = T(t_start)
    t_end = T(t_start + t_duration)
    step_dt = T(dt)
    sample_dt = T(save_dt)
    step_dt > zero(T) || error("RK4 trajectory requires dt > 0.")
    sample_dt > zero(T) || error("RK4 trajectory requires save_dt > 0.")

    ts = T[]
    us = typeof(u0)[]
    u = u0
    if save_start
        push!(ts, t)
        push!(us, u)
    end

    next_save = t + sample_dt
    tol = max(eps(T) * max(one(T), abs(t_end)), T(1.0e-6) * sample_dt)
    while t < t_end - tol
        h = min(step_dt, t_end - t)
        u = rk4_step_local(odefun, u, pvec, t, h)
        t += h
        if t >= next_save - tol
            push!(ts, t)
            push!(us, u)
            while next_save <= t + tol
                next_save += sample_dt
            end
        end
    end

    if isempty(ts) || abs(Float64(ts[end] - t_end)) > Float64(tol)
        push!(ts, t_end)
        push!(us, u)
    end
    return (t=ts, u=us)
end


function solve_trajectory_rk4_local_phase_argument(
    odefun,
    u0,
    pvec,
    t_start,
    t_duration,
    dt,
    save_dt,
    period,
    ;
    save_start::Bool=true,
)
    T = eltype(u0)
    period_t = T(period)
    period_t > zero(T) || error("Phase-argument RK4 trajectory requires a positive period.")
    t = T(t_start)
    t_end = T(t_start + t_duration)
    step_dt = T(dt)
    sample_dt = T(save_dt)
    step_dt > zero(T) || error("RK4 trajectory requires dt > 0.")
    sample_dt > zero(T) || error("RK4 trajectory requires save_dt > 0.")

    ts = T[]
    us = typeof(u0)[]
    u = u0
    phase = phase_from_period(t_start, period_t, T)
    if save_start
        push!(ts, t)
        push!(us, u)
    end

    next_save = t + sample_dt
    tol = max(eps(T) * max(one(T), abs(t_end)), T(1.0e-6) * sample_dt)
    while t < t_end - tol
        h = min(step_dt, t_end - t)
        dphase = T(2 * pi) * h / period_t
        u, phase = rk4_step_local_phase_argument(odefun, u, pvec, phase, h, dphase)
        t += h
        if t >= next_save - tol
            push!(ts, t)
            push!(us, u)
            while next_save <= t + tol
                next_save += sample_dt
            end
        end
    end

    if isempty(ts) || abs(Float64(ts[end] - t_end)) > Float64(tol)
        push!(ts, t_end)
        push!(us, u)
    end
    return (t=ts, u=us)
end


function stitch_trajectory_solutions(sol_a, sol_b)
    ts = Float64.(collect(sol_a.t))
    us = collect(sol_a.u)
    for i in eachindex(sol_b.t)
        if isempty(ts) || Float64(sol_b.t[i]) > Float64(ts[end])
            push!(ts, Float64(sol_b.t[i]))
            push!(us, sol_b.u[i])
        end
    end
    return (t=ts, u=us)
end


function solve_single_trajectory(
    odefun,
    u0,
    pvec,
    t_end,
    dt,
    save_dt,
    solver_mode::String,
    solver_name::String,
    device::String,
    abstol,
    reltol,
    ;
    rk4_stage_b_start=nothing,
    custom_time_argument::String="time",
    period=nothing,
)
    T = eltype(u0)
    solver_mode_l = lowercase(solver_mode)
    rk4_selected = solver_mode_l in ("rk4_extrema", "rk4_full_extrema", "rk4_full_extrema_custom")
    if rk4_selected
        lowercase(solver_name) == "tsit5" || error("$solver_mode_l currently requires solver = Tsit5.")
        length(u0) <= 8 || error("$solver_mode_l point probes currently support at most 8 states.")
        if solver_mode_l == "rk4_full_extrema" && odefun !== default_duffing
            error("rk4_full_extrema is the optimized built-in Duffing RK4 A+B path. Use rk4_full_extrema_custom for custom models.")
        end
        if solver_mode_l in ("rk4_full_extrema", "rk4_full_extrema_custom")
            if odefun === default_duffing
                return solve_trajectory_rk4_phase_wrapped_default(u0, pvec, zero(T), T(t_end), T(dt), T(save_dt); save_start=true)
            end
            if lowercase(custom_time_argument) == "phase"
                period !== nothing || error("Phase-argument RK4 point probes require the forcing period.")
                return solve_trajectory_rk4_local_phase_argument(odefun, u0, pvec, zero(T), T(t_end), T(dt), T(save_dt), T(period); save_start=true)
            end
            return solve_trajectory_rk4_local(odefun, u0, pvec, zero(T), T(t_end), T(dt), T(save_dt); save_start=true)
        end

        rk4_stage_b_start !== nothing || error("rk4_extrema point probes require the Stage A/B split time.")
        t_split = T(rk4_stage_b_start)
        if t_split <= zero(T)
            if odefun === default_duffing
                return solve_trajectory_rk4_phase_wrapped_default(u0, pvec, zero(T), T(t_end), T(dt), T(save_dt); save_start=true)
            end
            return solve_trajectory_rk4_local(odefun, u0, pvec, zero(T), T(t_end), T(dt), T(save_dt); save_start=true)
        end
        prob_a = ODEProblem{false}(odefun, u0, (zero(T), t_split), pvec)
        sol_a = solve(prob_a, choose_cpu_solver(solver_name); adaptive=false, dt=T(dt), saveat=T(save_dt), save_start=true)
        u_split = sol_a.u[end]
        sol_b = if odefun === default_duffing
            solve_trajectory_rk4_phase_wrapped_default(u_split, pvec, t_split, T(t_end) - t_split, T(dt), T(save_dt); save_start=false)
        else
            solve_trajectory_rk4_local(odefun, u_split, pvec, t_split, T(t_end) - t_split, T(dt), T(save_dt); save_start=false)
        end
        return stitch_trajectory_solutions(sol_a, sol_b)
    end

    effective_solver_mode = solver_mode_l == "streaming_extrema" ? "fixed" : solver_mode

    if device == "gpu"
        lowercase(effective_solver_mode) == "fixed" || error("GPU point probes currently support fixed-step mode only.")
        lowercase(solver_name) == "tsit5" || error("GPU point probes currently support Tsit5 only.")
        prob = ODEProblem{false}(odefun, u0, (zero(T), T(t_end)), pvec)
        return solve(
            prob,
            choose_cpu_solver(solver_name);
            adaptive=false,
            dt=T(dt),
            saveat=T(save_dt),
            save_start=true,
        )
    end

    prob = ODEProblem{false}(odefun, u0, (zero(T), T(t_end)), pvec)
    solver = choose_cpu_solver(solver_name)
    if lowercase(effective_solver_mode) == "fixed"
        return solve(
            prob,
            solver;
            adaptive=false,
            dt=T(dt),
            saveat=T(save_dt),
            save_start=true,
        )
    end

    return solve(
        prob,
        solver;
        adaptive=true,
        abstol=abstol,
        reltol=reltol,
        saveat=T(save_dt),
        save_start=true,
    )
end


function streaming_chunk_steps(integration_cfg::Dict{String, Any}, period, save_dt)
    chunk_periods = Float64(get(integration_cfg, "streaming_chunk_periods", 10.0))
    if Float64(period) > 0 && chunk_periods > 0
        return max(1, round(Int, chunk_periods * Float64(period) / Float64(save_dt)))
    end
    return max(1, Int(get(integration_cfg, "streaming_chunk_steps", 500)))
end


function streaming_total_save_steps(t_evaluation, save_dt)
    return max(1, round(Int, Float64(t_evaluation) / Float64(save_dt)))
end


function streaming_chunk_covers_window(integration_cfg::Dict{String, Any}, period, save_dt, t_evaluation)
    total_save_steps = streaming_total_save_steps(t_evaluation, save_dt)
    return streaming_chunk_steps(integration_cfg, period, save_dt) >= total_save_steps
end


function collect_stage_b_extrema_streaming_gpu(
    odefun,
    u_trans_list,
    pvec,
    t_transient,
    t_evaluation,
    dt,
    save_dt,
    period,
    integration_cfg::Dict{String, Any},
    observable_idx::Int,
    zero_cross_idx::Int,
    max_extrema::Int,
    eps,
    solver_name::String,
    abstol,
    reltol,
)
    n_total = length(u_trans_list)
    extrema_all = [Float32[] for _ in 1:n_total]
    current_states = copy(u_trans_list)
    prev_saved = Vector{eltype(u_trans_list)}(undef, n_total)
    has_prev = falses(n_total)

    total_save_steps = streaming_total_save_steps(t_evaluation, save_dt)
    chunk_save_steps = max(1, min(streaming_chunk_steps(integration_cfg, period, save_dt), total_save_steps))
    t0 = t_transient
    remaining = total_save_steps

    while remaining > 0
        steps_this_chunk = min(chunk_save_steps, remaining)
        chunk_duration = save_dt * steps_this_chunk
        sol_chunk = solve_stage_b(
            odefun,
            current_states,
            pvec,
            t0,
            chunk_duration,
            dt,
            save_dt,
            "fixed",
            solver_name,
            "gpu",
            abstol,
            reltol,
        )

        for i in 1:n_total
            sol_i = sol_chunk[i]
            for u in sol_i.u
                if has_prev[i]
                    stream_extrema_update!(extrema_all[i], prev_saved[i], u, observable_idx, zero_cross_idx, max_extrema, eps)
                end
                prev_saved[i] = u
                has_prev[i] = true
            end
            current_states[i] = sol_i.u[end]
        end

        t0 += chunk_duration
        remaining -= steps_this_chunk
    end

    return extrema_all, current_states
end


function collect_stage_b_extrema_streaming_cpu(
    odefun,
    u_trans_list,
    pvec,
    t_transient,
    t_evaluation,
    dt,
    observable_idx::Int,
    zero_cross_idx::Int,
    max_extrema::Int,
    eps,
    solver_name::String,
)
    t_end = t_transient + t_evaluation
    extrema_all = Vector{Vector{Float32}}(undef, length(u_trans_list))
    final_states = Vector{eltype(u_trans_list)}(undef, length(u_trans_list))

    Threads.@threads for i in eachindex(u_trans_list)
        solver = choose_cpu_solver(solver_name)
        prob = ODEProblem{false}(odefun, u_trans_list[i], (t_transient, t_end), pvec)
        integrator = init(prob, solver; adaptive=false, dt=dt, save_everystep=false, save_start=false)
        values = Float32[]
        if integrator.t < t_end
            step!(integrator)
            prev_u = integrator.u
            while integrator.t < t_end && length(values) < max_extrema
                step!(integrator)
                stream_extrema_update!(values, prev_u, integrator.u, observable_idx, zero_cross_idx, max_extrema, eps)
                prev_u = integrator.u
            end
            while integrator.t < t_end
                step!(integrator)
            end
        end
        extrema_all[i] = values
        final_states[i] = integrator.u
    end

    return extrema_all, final_states
end


function collect_stage_b_extrema_streaming(
    odefun,
    u_trans_list,
    pvec,
    t_transient,
    t_evaluation,
    dt,
    save_dt,
    period,
    integration_cfg::Dict{String, Any},
    observable_idx::Int,
    zero_cross_idx::Int,
    max_extrema::Int,
    eps,
    device::String,
    solver_name::String,
    abstol,
    reltol,
)
    if device == "gpu"
        return collect_stage_b_extrema_streaming_gpu(
            odefun,
            u_trans_list,
            pvec,
            t_transient,
            t_evaluation,
            dt,
            save_dt,
            period,
            integration_cfg,
            observable_idx,
            zero_cross_idx,
            max_extrema,
            eps,
            solver_name,
            abstol,
            reltol,
        )
    end

    return collect_stage_b_extrema_streaming_cpu(
        odefun,
        u_trans_list,
        pvec,
        t_transient,
        t_evaluation,
        dt,
        observable_idx,
        zero_cross_idx,
        max_extrema,
        eps,
        solver_name,
    )
end


function solve_phase_sample_solutions(
    odefun,
    u_trans_list,
    selected,
    pvec,
    t_transient,
    t_evaluation,
    dt,
    save_dt,
    solver_name::String,
    device::String,
    abstol,
    reltol,
)
    isempty(selected) && return Any[]
    selected_u = [u_trans_list[sol_idx] for (_sample_id, _label, sol_idx) in selected]
    sol = solve_stage_b(
        odefun,
        selected_u,
        pvec,
        t_transient,
        t_evaluation,
        dt,
        save_dt,
        "fixed",
        solver_name,
        device,
        abstol,
        reltol,
    )
    return [sol[i] for i in eachindex(selected)]
end


function solve_phase_sample_solutions_rk4(
    odefun,
    u_trans_list,
    selected,
    pvec,
    t_transient,
    t_evaluation,
    dt,
    save_dt,
    ;
    custom_time_argument::String="time",
    period=nothing,
)
    isempty(selected) && return Any[]
    if odefun === default_duffing
        return [
            solve_trajectory_rk4_phase_wrapped_default(
                u_trans_list[sol_idx],
                pvec,
                t_transient,
                t_evaluation,
                dt,
                save_dt;
                save_start=false,
            ) for (_sample_id, _label, sol_idx) in selected
        ]
    end

    if lowercase(custom_time_argument) == "phase"
        period !== nothing || error("Phase-argument RK4 phase samples require the forcing period.")
        return [
            solve_trajectory_rk4_local_phase_argument(
                odefun,
                u_trans_list[sol_idx],
                pvec,
                t_transient,
                t_evaluation,
                dt,
                save_dt,
                period;
                save_start=false,
            ) for (_sample_id, _label, sol_idx) in selected
        ]
    end

    return [
        solve_trajectory_rk4_local(
            odefun,
            u_trans_list[sol_idx],
            pvec,
            t_transient,
            t_evaluation,
            dt,
            save_dt;
            save_start=false,
        ) for (_sample_id, _label, sol_idx) in selected
    ]
end


function write_trajectory_csv(path::String, sol, state_names::Vector{String})
    open(path, "w") do io
        println(io, join(vcat(["t"], state_names), ","))
        for i in eachindex(sol.t)
            values = [string(Float64(sol.t[i]))]
            u = sol.u[i]
            for value in u
                push!(values, string(Float64(value)))
            end
            println(io, join(values, ","))
        end
    end
end


function supports_batched_sweep_transient(
    state_names::Vector{String},
    param_names::Vector{String},
    equations::Vector{String},
    sweep_parameter::String,
    integration_cfg::Dict{String, Any},
    device::String,
    solver_mode::String,
    solver_name::String,
)
    period_mode = lowercase(String(integration_cfg["period_mode"]))
    period_expr = normalize_expr_string(String(integration_cfg["period_expression"]))
    solver_mode_l = lowercase(solver_mode)
    return device == "gpu" &&
           solver_mode_l in ("fixed", "streaming_extrema", "rk4_extrema") &&
           lowercase(solver_name) == "tsit5" &&
           period_mode == "periodic" &&
           sweep_parameter == "w" &&
           period_expr == "2*pi/w" &&
           matches_default_duffing(state_names, param_names, equations)
end


function build_sweep_pvecs(
    sweep_values,
    sweep_idx::Int,
    param_values_f64::Vector{Float64},
    ::Type{precision},
) where {precision}
    pvecs = Vector{SVector{length(param_values_f64), precision}}(undef, length(sweep_values))
    for (k, sweep_value) in enumerate(sweep_values)
        params_i = copy(param_values_f64)
        params_i[sweep_idx] = Float64(sweep_value)
        pvecs[k] = SVector{length(params_i), precision}(Tuple(precision.(params_i)))
    end
    return pvecs
end


function extract_batched_transient_states(sol_a_batch, n_total::Int, sweep_num::Int)
    first_state = sol_a_batch[1][end]
    transients = Vector{Vector{typeof(first_state)}}(undef, sweep_num)
    for k in 1:sweep_num
        states_k = Vector{typeof(first_state)}(undef, n_total)
        base = (k - 1) * n_total
        for i in 1:n_total
            states_k[i] = sol_a_batch[base + i][end]
        end
        transients[k] = states_k
    end
    return transients
end


function compute_operating_point(
    odefun,
    u0_list,
    pvec,
    nx::Int,
    ny::Int,
    n_total::Int,
    observable_idx::Int,
    zero_cross_idx::Int,
    max_extrema::Int,
    tol::Float32,
    k_fp::Int,
    extrema_eps::Float64,
    integration_cfg::Dict{String, Any},
    param_names::Vector{String},
    param_values_f64::Vector{Float64},
    ::Type{precision},
    device::String,
    solver_mode::String,
    solver_name::String,
    abstol,
    reltol,
    ;
    verbose::Bool=true,
    precomputed_u_trans_list=nothing,
    collect_phase_samples::Bool=true,
    state_names::Vector{String}=String[],
    model_equations::Vector{String}=String[],
) where {precision}
    period, t_transient, t_evaluation, dt, save_dt = resolve_time_settings(integration_cfg, param_names, param_values_f64, precision)
    custom_time_argument = get_custom_time_argument(integration_cfg)
    solver_mode_l = lowercase(solver_mode)
    streaming_extrema = solver_mode_l == "streaming_extrema"
    rk4_extrema = solver_mode_l == "rk4_extrema"
    rk4_full_extrema = solver_mode_l == "rk4_full_extrema"
    rk4_full_extrema_custom = solver_mode_l == "rk4_full_extrema_custom"
    rk4_full_extrema_any = rk4_full_extrema || rk4_full_extrema_custom
    rk4_streamed = streaming_extrema || rk4_extrema || rk4_full_extrema_any
    stage_solver_mode = rk4_streamed ? "fixed" : solver_mode
    runtime_model_equations = (custom_time_argument == "phase" && !isempty(model_equations)) ?
                              runtime_equations_for_custom_time_argument(model_equations, param_names, integration_cfg) :
                              model_equations
    streaming_gpu_full_window = streaming_extrema &&
                                device == "gpu" &&
                                streaming_chunk_covers_window(integration_cfg, period, save_dt, t_evaluation)

    if rk4_extrema || rk4_full_extrema_any
        device == "gpu" || error("$solver_mode_l currently supports GPU runs only.")
        if rk4_extrema || rk4_full_extrema
            precision == Float32 || error("$solver_mode_l currently supports Float32 only.")
        elseif rk4_full_extrema_custom
            precision in (Float32, Float64) || error("$solver_mode_l currently supports Float32 and Float64 only.")
        end
        lowercase(solver_name) == "tsit5" || error("$solver_mode_l currently requires solver = Tsit5.")
        nstates_rk4 = length(u0_list[1])
        2 <= nstates_rk4 <= 8 || error("$solver_mode_l currently supports 2 to 8 states.")
        if rk4_full_extrema && odefun !== default_duffing
            error("rk4_full_extrema is the optimized built-in Duffing RK4 A+B path. Use rk4_full_extrema_custom for custom models.")
        end
        if rk4_full_extrema_custom
            !isempty(state_names) || error("rk4_full_extrema_custom requires state_names.")
            !isempty(runtime_model_equations) || error("rk4_full_extrema_custom requires model equations.")
            if custom_time_argument == "phase"
                lowercase(String(integration_cfg["period_mode"])) == "periodic" || error("rk4_full_extrema_custom with phase time argument requires period_mode = periodic.")
                period > zero(period) || error("rk4_full_extrema_custom with phase time argument requires a positive period.")
            end
        end
        if rk4_full_extrema_any && precomputed_u_trans_list !== nothing
            error("$solver_mode_l integrates Stage A inside the fused RK4 kernel and does not accept precomputed transient states.")
        end
    end

    if verbose
        if lowercase(String(integration_cfg["period_mode"])) == "periodic"
            println("Period: $period | t_transient=$t_transient | t_evaluation=$t_evaluation | dt=$dt")
        else
            println("t_transient=$t_transient | t_evaluation=$t_evaluation | dt=$dt | save_dt=$save_dt")
        end
        println(rk4_full_extrema_any ? "Stage A: fused RK4 kernel will include transient integration" : "Stage A: transient integration")
    end

    if rk4_full_extrema_any
        stage_a_seconds = 0.0
        transient_state_seconds = 0.0
        u_trans_list = nothing
    elseif precomputed_u_trans_list === nothing
        t_stage_a_start = time()
        sol_a = Base.invokelatest(solve_stage_a, odefun, u0_list, pvec, t_transient, dt, stage_solver_mode, solver_name, device, abstol, reltol)
        stage_a_seconds = time() - t_stage_a_start

        t_trans_state_start = time()
        u_trans_list = [sol_a[i][end] for i in 1:n_total]
        transient_state_seconds = time() - t_trans_state_start
    else
        verbose && println("Stage A: using precomputed transient states")
        stage_a_seconds = 0.0
        t_trans_state_start = time()
        u_trans_list = precomputed_u_trans_list
        transient_state_seconds = time() - t_trans_state_start
    end

    stage_b_label = rk4_full_extrema ?
                    "Stage A+B: optimized built-in Duffing GPU RK4 transient and extrema evaluation" :
                    (rk4_full_extrema_custom ?
                     "Stage A+B: specialized custom GPU RK4 transient and extrema evaluation" :
                    (rk4_extrema ?
                     "Stage B: fused GPU RK4 extrema evaluation window" :
                    (streaming_extrema && !streaming_gpu_full_window ?
                     "Stage B: streaming extrema evaluation window" :
                     "Stage B: evaluation window")))
    verbose && println(stage_b_label)
    t_stage_b_start = time()
    sol_b = nothing
    phase_sample_selected = nothing
    phase_sample_solutions = nothing
    if rk4_full_extrema
        Base.invokelatest(ensure_rk4_extrema_cuda_loaded)
        rk4_collect_default = Base.invokelatest(
            getfield,
            @__MODULE__,
            :collect_full_extrema_rk4_gpu_default_duffing,
        )
        extrema_all, u_trans_list = Base.invokelatest(
            rk4_collect_default,
            u0_list,
            pvec,
            t_transient,
            t_evaluation,
            dt,
            save_dt,
            observable_idx,
            zero_cross_idx,
            max_extrema,
            extrema_eps,
        )
    elseif rk4_full_extrema_custom
        Base.invokelatest(ensure_rk4_extrema_cuda_loaded)
        rk4_builder = Base.invokelatest(
            getfield,
            @__MODULE__,
            :build_specialized_rk4_full_extrema_collector,
        )
        rk4_collect_custom = Base.invokelatest(
            rk4_builder,
            state_names,
            param_names,
            runtime_model_equations,
            custom_time_argument,
        )
        extrema_all, u_trans_list = Base.invokelatest(
            rk4_collect_custom,
            u0_list,
            pvec,
            t_transient,
            t_evaluation,
            dt,
            save_dt,
            period,
            observable_idx,
            zero_cross_idx,
            max_extrema,
            extrema_eps,
        )
    elseif rk4_extrema
        Base.invokelatest(ensure_rk4_extrema_cuda_loaded)
        rk4_collect = Base.invokelatest(
            getfield,
            @__MODULE__,
            odefun === default_duffing ? :collect_stage_b_extrema_rk4_gpu_default_duffing : :collect_stage_b_extrema_rk4_gpu_generic,
        )
        extrema_all = Base.invokelatest(
            rk4_collect,
            (odefun === default_duffing ? () : (odefun,))...,
            u_trans_list,
            pvec,
            t_transient,
            t_evaluation,
            dt,
            save_dt,
            observable_idx,
            zero_cross_idx,
            max_extrema,
            extrema_eps,
        )
    elseif streaming_extrema && !streaming_gpu_full_window
        extrema_all, _final_states = collect_stage_b_extrema_streaming(
            odefun,
            u_trans_list,
            pvec,
            t_transient,
            t_evaluation,
            dt,
            save_dt,
            period,
            integration_cfg,
            observable_idx,
            zero_cross_idx,
            max_extrema,
            extrema_eps,
            device,
            solver_name,
            abstol,
            reltol,
        )
    else
        sol_b = Base.invokelatest(solve_stage_b, odefun, u_trans_list, pvec, t_transient, t_evaluation, dt, save_dt, stage_solver_mode, solver_name, device, abstol, reltol)
        extrema_all = Vector{Vector{Float32}}(undef, n_total)
        for i in 1:n_total
            extrema_all[i] = collect_extrema(sol_b[i], observable_idx, zero_cross_idx, max_extrema, extrema_eps)
        end
    end

    stage_b_extrema_seconds = time() - t_stage_b_start

    verbose && println(rk4_streamed ? "Classification using streamed extrema" : "Classification using extrema")
    t_class_start = time()
    labels_vec, counts, num_classes = classify_extrema_all(extrema_all, tol, k_fp)
    classification_seconds = time() - t_class_start

    phase_sample_seconds = 0.0
    if rk4_streamed && collect_phase_samples && sol_b === nothing
        t_phase_sample_start = time()
        phase_sample_selected = select_phase_sample_indices(labels_vec)
        phase_sample_solutions = if rk4_extrema || rk4_full_extrema_any
            Base.invokelatest(
                solve_phase_sample_solutions_rk4,
                odefun,
                u_trans_list,
                phase_sample_selected,
                pvec,
                t_transient,
                t_evaluation,
                dt,
                save_dt;
                custom_time_argument=custom_time_argument,
                period=period,
            )
        else
            solve_phase_sample_solutions(
                odefun,
                u_trans_list,
                phase_sample_selected,
                pvec,
                t_transient,
                t_evaluation,
                dt,
                save_dt,
                solver_name,
                device,
                abstol,
                reltol,
            )
        end
        phase_sample_seconds = time() - t_phase_sample_start
    end
    stage_b_seconds = stage_b_extrema_seconds + phase_sample_seconds

    labels_mat = reshape(labels_vec, nx, ny)'
    return (
        labels_vec=labels_vec,
        labels_mat=labels_mat,
        counts=counts,
        sol_b=sol_b,
        extrema_all=extrema_all,
        num_classes=num_classes,
        phase_sample_selected=phase_sample_selected,
        phase_sample_solutions=phase_sample_solutions,
        period=period,
        t_transient=t_transient,
        t_evaluation=t_evaluation,
        dt=dt,
        save_dt=save_dt,
        timing=(
            stage_a_seconds=stage_a_seconds,
            transient_state_seconds=transient_state_seconds,
            stage_b_extrema_seconds=stage_b_extrema_seconds,
            stage_b_seconds=stage_b_seconds,
            classification_seconds=classification_seconds,
            phase_sample_seconds=phase_sample_seconds,
            total_seconds=stage_a_seconds + transient_state_seconds + stage_b_seconds + classification_seconds,
        ),
    )
end


function write_sweep_class_stats_header(path::String)
    open(path, "w") do io
        println(io, "parameter_value,label,count,fraction")
    end
end


function append_sweep_class_stats(path::String, parameter_value::Float64, counts::Dict{Int32, Int}, total::Int)
    labels = sort(collect(keys(counts)))
    open(path, "a") do io
        for label in labels
            count = counts[label]
            fraction = count / total
            println(io, "$(parameter_value),$(label),$(count),$(fraction)")
        end
    end
end


function write_sweep_detail_manifest_header(path::String)
    open(path, "w") do io
        println(io, "index,parameter_value,labels_csv,phase_samples_csv,phase_samples_format,num_classes,n_phase_samples")
    end
end


function append_sweep_detail_manifest(
    path::String,
    index::Int,
    parameter_value::Float64,
    labels_path::String,
    phase_samples_path::String,
    phase_samples_format::String,
    num_classes::Int,
    n_phase_samples::Int,
)
    open(path, "a") do io
        println(io, "$(index),$(parameter_value),$(labels_path),$(phase_samples_path),$(phase_samples_format),$(num_classes),$(n_phase_samples)")
    end
end


function write_sweep_timing_header(path::String)
    open(path, "w") do io
        println(
            io,
            "index,parameter_value,stage_a_s,transient_state_s,stage_b_s,stage_b_extrema_s,phase_sample_s,classification_s,compute_total_s,collect_rows_s,write_stats_s,write_details_s,iteration_total_s,num_classes,num_extrema,n_phase_samples",
        )
    end
end


function append_sweep_timing(
    path::String,
    index::Int,
    parameter_value::Float64,
    result,
    collect_rows_seconds::Float64,
    write_stats_seconds::Float64,
    write_details_seconds::Float64,
    iteration_total_seconds::Float64,
    num_extrema::Int,
    n_phase_samples::Int,
)
    timing = result.timing
    open(path, "a") do io
        println(
            io,
            join(
                [
                    index,
                    parameter_value,
                    timing.stage_a_seconds,
                    timing.transient_state_seconds,
                    timing.stage_b_seconds,
                    timing.stage_b_extrema_seconds,
                    timing.phase_sample_seconds,
                    timing.classification_seconds,
                    timing.total_seconds,
                    collect_rows_seconds,
                    write_stats_seconds,
                    write_details_seconds,
                    iteration_total_seconds,
                    result.num_classes,
                    num_extrema,
                    n_phase_samples,
                ],
                ",",
            ),
        )
    end
end


function sweep_label_ranks(labels_vec::Vector{Int32})
    counts = Dict{Int32, Int}()
    for label in labels_vec
        label > 0 || continue
        counts[label] = get(counts, label, 0) + 1
    end
    ordered = sort(collect(keys(counts)); by=label -> (-counts[label], Int(label)))
    ranks = Dict{Int32, Int32}()
    for (idx, label) in enumerate(ordered)
        ranks[label] = Int32(idx)
    end
    return counts, ranks
end


function collect_sweep_extrema_rows!(
    rows::Vector{SweepExtremaRow},
    parameter_value::Float64,
    labels_vec::Vector{Int32},
    extrema_all::Vector{Vector{Float32}},
    extrema_mode::String,
    min_class_fraction::Float64,
    max_classes_per_value::Int,
)
    feature_template = invalid_extrema_features()
    features = Vector{typeof(feature_template)}(undef, length(extrema_all))
    label_features = Dict{Int32, typeof(feature_template)}()
    label_extrema = Dict{Int32, Vector{Float32}}()
    label_counts, label_ranks = sweep_label_ranks(labels_vec)
    n_total = max(1, length(labels_vec))

    local_amin = Float32(Inf)
    local_amax = Float32(-Inf)
    local_mmin = Float32(Inf)
    local_mmax = Float32(-Inf)
    local_smin = Float32(Inf)
    local_smax = Float32(-Inf)

    for i in eachindex(extrema_all)
        feats = extrema_features(extrema_all[i])
        features[i] = feats

        if feats.k > 0
            local_amin = min(local_amin, feats.amp)
            local_amax = max(local_amax, feats.amp)
            local_mmin = min(local_mmin, feats.mean)
            local_mmax = max(local_mmax, feats.mean)
            local_smin = min(local_smin, feats.std)
            local_smax = max(local_smax, feats.std)
        end

        label = labels_vec[i]
        if label > 0 && feats.k > 0 && !haskey(label_features, label)
            label_features[label] = feats
            label_extrema[label] = extrema_all[i]
        end
    end

    rows_before = length(rows)
    if extrema_mode == "trajectories"
        for i in eachindex(labels_vec)
            label = labels_vec[i]
            class_count = get(label_counts, label, 0)
            class_fraction = Float32(class_count / n_total)
            class_rank = get(label_ranks, label, Int32(typemax(Int32)))
            if label > 0 && (class_fraction < min_class_fraction || Int(class_rank) > max_classes_per_value)
                continue
            end
            feats = label > 0 ? get(label_features, label, features[i]) : features[i]
            for x_extreme in extrema_all[i]
                if isfinite(x_extreme)
                    push!(
                        rows,
                        SweepExtremaRow(
                            parameter_value,
                            label,
                            Int32(class_count),
                            class_fraction,
                            class_rank,
                            x_extreme,
                            feats.amp,
                            feats.mean,
                            feats.std,
                            feats.k,
                            feats.pattern,
                        ),
                    )
                end
            end
        end
    else
        for label in sort(collect(keys(label_extrema)); by=label -> (get(label_ranks, label, Int32(typemax(Int32))), Int(label)))
            class_count = get(label_counts, label, 0)
            class_fraction = Float32(class_count / n_total)
            class_rank = get(label_ranks, label, Int32(typemax(Int32)))
            if class_fraction < min_class_fraction || Int(class_rank) > max_classes_per_value
                continue
            end
            feats = label_features[label]
            for x_extreme in label_extrema[label]
                if isfinite(x_extreme)
                    push!(
                        rows,
                        SweepExtremaRow(
                            parameter_value,
                            label,
                            Int32(class_count),
                            class_fraction,
                            class_rank,
                            x_extreme,
                            feats.amp,
                            feats.mean,
                            feats.std,
                            feats.k,
                            feats.pattern,
                        ),
                    )
                end
            end
        end
    end
    rows_added = length(rows) - rows_before

    return (
        amin=local_amin,
        amax=local_amax,
        mmin=local_mmin,
        mmax=local_mmax,
        smin=local_smin,
        smax=local_smax,
        rows_added=rows_added,
    )
end


function finalize_sweep_color_bounds(
    amin::Float32,
    amax::Float32,
    mmin::Float32,
    mmax::Float32,
    smin::Float32,
    smax::Float32,
)
    if !isfinite(amin) || !isfinite(amax) || amin == amax
        amin = 0f0
        amax = 1f0
    end
    if !isfinite(mmin) || !isfinite(mmax) || mmin == mmax
        mmin = -1f0
        mmax = 1f0
    end
    if !isfinite(smin) || !isfinite(smax) || smin == smax
        smin = 0f0
        smax = 1f0
    end
    return (amin=amin, amax=amax, mmin=mmin, mmax=mmax, smin=smin, smax=smax)
end


function write_sweep_extrema(path::String, rows::Vector{SweepExtremaRow}, color_bounds)
    bounds = finalize_sweep_color_bounds(
        color_bounds.amin,
        color_bounds.amax,
        color_bounds.mmin,
        color_bounds.mmax,
        color_bounds.smin,
        color_bounds.smax,
    )

    open(path, "w") do io
        println(io, "parameter_value,label,class_count,class_fraction,class_rank,x_extreme,color_r,color_g,color_b,amp,mean,std,k_extrema,pattern")
        for row in rows
            r, g, b = feature_color_rgb(
                row.amp,
                row.k_extrema,
                row.mean,
                row.std,
                bounds.amin,
                bounds.amax,
                bounds.mmin,
                bounds.mmax,
                bounds.smin,
                bounds.smax;
                pattern=row.pattern,
            )
            println(
                io,
                "$(row.parameter_value),$(Int(row.label)),$(Int(row.class_count)),$(Float64(row.class_fraction)),$(Int(row.class_rank)),$(Float64(row.x_extreme)),$r,$g,$b,$(Float64(row.amp)),$(Float64(row.mean)),$(Float64(row.std)),$(Int(row.k_extrema)),$(Float64(row.pattern))",
            )
        end
    end
end


function run_config(config_path::String)
    config_path = abspath(config_path)
    cfg = TOML.parsefile(config_path)

    analysis_cfg = get(cfg, "analysis", Dict{String, Any}("mode" => "single_basin"))
    sweep_cfg = get(cfg, "sweep", Dict{String, Any}())
    probe_cfg = get(cfg, "probe", Dict{String, Any}())
    optimization_cfg = get(cfg, "optimization", Dict{String, Any}())
    model_cfg = cfg["model"]
    plane_cfg = cfg["initial_condition_plane"]
    integration_cfg = cfg["integration"]
    class_cfg = cfg["classification"]
    output_cfg = cfg["output"]
    analysis_mode = lowercase(String(get(analysis_cfg, "mode", "single_basin")))

    state_names = String.(model_cfg["state_names"])
    equations = String.(model_cfg["equations"])
    param_names = String.(model_cfg["parameter_names"])
    param_values_f64 = Float64.(model_cfg["parameter_values"])
    precision = get_precision_type(String(integration_cfg["precision"]))
    param_values = precision.(param_values_f64)
    state_defaults = precision.(model_cfg["state_defaults"])

    x_idx = get_state_index(state_names, String(plane_cfg["x_state"]))
    y_idx = get_state_index(state_names, String(plane_cfg["y_state"]))
    observable_idx = get_state_index(state_names, String(class_cfg["observable_state"]))
    zero_cross_idx = get_state_index(state_names, String(class_cfg["zero_cross_state"]))

    nx, ny, x_min, x_max, y_min, y_max = derive_grid(plane_cfg)
    xs = collect(range(precision(x_min), precision(x_max); length=nx))
    ys = collect(range(precision(y_min), precision(y_max); length=ny))
    n_total = nx * ny
    plane_summary = Dict(
        "x_state" => String(plane_cfg["x_state"]),
        "y_state" => String(plane_cfg["y_state"]),
        "x_min" => x_min,
        "x_max" => x_max,
        "y_min" => y_min,
        "y_max" => y_max,
        "grid_mode" => String(plane_cfg["grid_mode"]),
        "n_target" => Int(plane_cfg["n_target"]),
        "nx" => nx,
        "ny" => ny,
    )

    device = lowercase(String(integration_cfg["device"]))
    solver_mode = String(integration_cfg["solver_mode"])
    solver_name = String(integration_cfg["solver"])
    configured_custom_time_argument = get_custom_time_argument(integration_cfg)
    custom_time_argument, integration_cfg, custom_time_argument_auto = select_runtime_custom_time_argument(
        equations,
        param_names,
        integration_cfg,
        solver_mode,
    )
    abstol = precision(integration_cfg["abstol"])
    reltol = precision(integration_cfg["reltol"])
    max_extrema = Int(class_cfg["max_extrema"])
    tol = Float32(class_cfg["fingerprint_tol"])
    k_fp = Int(class_cfg["fingerprint_k"])
    extrema_eps = Float64(class_cfg["extrema_eps"])

    println("Configuration loaded: $config_path")
    println("States: ", join(state_names, ", "))
    println("Parameters: ", join(["$(param_names[i])=$(param_values_f64[i])" for i in eachindex(param_names)], ", "))
    println("Grid: nx=$nx, ny=$ny, N=$n_total")
    println("Solver: device=$device, mode=$solver_mode, solver=$solver_name, precision=$(integration_cfg["precision"])")
    println("Analysis mode: $analysis_mode")

    if equations_use_symbol(equations, :phase) && custom_time_argument != "phase"
        error("Equations that use the symbol 'phase' require integration.custom_time_argument = 'phase'.")
    end
    if custom_time_argument == "phase"
        lowercase(solver_mode) == "rk4_full_extrema_custom" || error("integration.custom_time_argument = 'phase' is currently supported only by solver_mode = 'rk4_full_extrema_custom'.")
        lowercase(String(integration_cfg["period_mode"])) == "periodic" || error("integration.custom_time_argument = 'phase' requires integration.period_mode = 'periodic'.")
    end
    if custom_time_argument_auto
        println("Custom time argument: using wrapped phase automatically for periodic custom RK4 A+B.")
    end

    runtime_equations = runtime_equations_for_custom_time_argument(equations, param_names, integration_cfg)
    if custom_time_argument == "phase" && normalize_expr_string.(runtime_equations) != normalize_expr_string.(equations)
        frequency_symbol = phase_frequency_symbol_from_period_expression(String(integration_cfg["period_expression"]), param_names)
        println("Custom time argument: replacing $(frequency_symbol)*t by wrapped phase internally.")
    end

    force_custom_model = lowercase(solver_mode) == "rk4_full_extrema_custom"
    odefun, uses_dynamic_model = get_model_function(state_names, param_names, runtime_equations, device; force_custom=force_custom_model)
    u0_list = generate_initial_conditions(state_defaults, x_idx, y_idx, xs, ys)

    started = time()
    run_name = replace(String(output_cfg["run_name"]), r"[^A-Za-z0-9_\-]" => "_")
    stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    run_dir = joinpath(@__DIR__, "runs", "$(run_name)_$(stamp)")
    mkpath(run_dir)
    summary_path = joinpath(run_dir, "summary.toml")
    config_snapshot_path = joinpath(run_dir, "config_snapshot.toml")
    write(config_snapshot_path, read(config_path, String))

    if analysis_mode == "point_probe"
        initial_values_f64 = Float64.(get(probe_cfg, "initial_state", model_cfg["state_defaults"]))
        length(initial_values_f64) == length(state_names) || error("Probe initial_state must contain one value per state.")

        period, t_transient, t_evaluation, dt_probe, save_dt_probe = resolve_time_settings(
            integration_cfg,
            param_names,
            param_values_f64,
            precision,
        )
        t_end = t_transient + t_evaluation
        Float64(t_end) > 0 || error("Point probe requires a positive total integration time.")
        if Float64(save_dt_probe) <= 0
            save_dt_probe = precision(Float64(t_end) / 2000.0)
        end
        if Float64(dt_probe) <= 0
            dt_probe = save_dt_probe
        end

        pvec_probe = SVector{length(param_values), precision}(Tuple(param_values))
        u0_probe = SVector{length(initial_values_f64), precision}(Tuple(precision.(initial_values_f64)))
        selected_device = device
        if selected_device == "gpu"
            if lowercase(solver_mode) in ("rk4_extrema", "rk4_full_extrema", "rk4_full_extrema_custom")
                println("Point probe: RK4 extrema mode selected; generating the displayed trajectory with the same fixed-step RK4 update used for extrema detection.")
            else
                println("Point probe: GPU selected; generating the displayed single trajectory locally with the same fixed-step settings as the GPU run.")
            end
        end

        println("Point probe: selected method")
        selected_sol = Base.invokelatest(
            solve_single_trajectory,
            odefun,
            u0_probe,
            pvec_probe,
            t_end,
            dt_probe,
            save_dt_probe,
            solver_mode,
            solver_name,
            selected_device,
            abstol,
            reltol,
            ;
            rk4_stage_b_start=t_transient,
            custom_time_argument=custom_time_argument,
            period=period,
        )

        _, t_transient_bench, t_evaluation_bench, dt_bench, save_dt_bench = resolve_time_settings(
            integration_cfg,
            param_names,
            param_values_f64,
            Float64,
        )
        t_end_bench = t_transient_bench + t_evaluation_bench
        if save_dt_bench <= 0
            save_dt_bench = t_end_bench / 2000.0
        end
        if dt_bench <= 0
            dt_bench = save_dt_bench
        end

        pvec_bench = SVector{length(param_values_f64), Float64}(Tuple(param_values_f64))
        u0_bench = SVector{length(initial_values_f64), Float64}(Tuple(initial_values_f64))

        println("Point probe: benchmark CPU Float64 adaptive Vern9")
        benchmark_sol = Base.invokelatest(
            solve_single_trajectory,
            odefun,
            u0_bench,
            pvec_bench,
            t_end_bench,
            dt_bench,
            save_dt_bench,
            "adaptive",
            "Vern9",
            "cpu",
            1.0e-10,
            1.0e-10,
        )

        selected_path = joinpath(run_dir, "trajectory_selected.csv")
        benchmark_path = joinpath(run_dir, "trajectory_benchmark.csv")
        write_trajectory_csv(selected_path, selected_sol, state_names)
        write_trajectory_csv(benchmark_path, benchmark_sol, state_names)

        elapsed = round(time() - started; digits=3)
        summary = Dict(
            "analysis" => Dict("mode" => "point_probe"),
            "run" => Dict(
                "run_dir" => run_dir,
                "config_snapshot" => config_snapshot_path,
                "elapsed_seconds" => elapsed,
            ),
            "initial_condition_plane" => plane_summary,
            "integration" => Dict(
                "device" => device,
                "effective_device" => selected_device == "gpu" ? "cpu" : selected_device,
                "solver_mode" => solver_mode,
                "solver" => solver_name,
                "precision" => String(integration_cfg["precision"]),
                "custom_time_argument" => custom_time_argument,
                "configured_custom_time_argument" => configured_custom_time_argument,
                "custom_time_argument_auto" => custom_time_argument_auto,
                "dt" => Float64(dt_probe),
                "save_dt" => Float64(save_dt_probe),
                "period" => Float64(period),
                "t_transient" => Float64(t_transient),
                "t_evaluation" => Float64(t_evaluation),
                "t_end" => Float64(t_end),
                "benchmark_device" => "cpu",
                "benchmark_solver_mode" => "adaptive",
                "benchmark_solver" => "Vern9",
                "benchmark_precision" => "Float64",
                "benchmark_abstol" => 1.0e-10,
                "benchmark_reltol" => 1.0e-10,
            ),
            "probe" => Dict(
                "initial_state" => initial_values_f64,
                "x_state" => state_names[x_idx],
                "y_state" => state_names[y_idx],
                "x_value" => initial_values_f64[x_idx],
                "y_value" => initial_values_f64[y_idx],
            ),
            "paths" => Dict(
                "trajectory_selected_csv" => selected_path,
                "trajectory_benchmark_csv" => benchmark_path,
            ),
            "model" => Dict(
                "state_names" => state_names,
                "parameter_names" => param_names,
                "parameter_values" => param_values_f64,
                "uses_dynamic_model" => uses_dynamic_model,
            ),
        )

        open(summary_path, "w") do io
            TOML.print(io, summary)
        end

        println("Point probe completed.")
        println("Result folder: $run_dir")
        println("RESULT_DIR=$(run_dir)")
        println("RESULT_SUMMARY=$(summary_path)")
        return
    end

    if analysis_mode == "parameter_sweep"
        sweep_parameter = String(get(sweep_cfg, "parameter", param_names[1]))
        sweep_idx = findfirst(==(sweep_parameter), param_names)
        sweep_idx === nothing && error("Sweep parameter '$sweep_parameter' was not found in parameter_names.")

        sweep_min = Float64(get(sweep_cfg, "min_value", param_values_f64[sweep_idx]))
        sweep_max = Float64(get(sweep_cfg, "max_value", param_values_f64[sweep_idx]))
        sweep_num = Int(get(sweep_cfg, "num_values", 25))
        sweep_num >= 2 || error("Sweep num_values must be at least 2.")
        sweep_values = collect(range(sweep_min, sweep_max; length=sweep_num))

        sweep_extrema_path = joinpath(run_dir, "sweep_extrema.csv")
        sweep_stats_path = joinpath(run_dir, "sweep_class_stats.csv")
        sweep_details_path = joinpath(run_dir, "sweep_details.csv")
        sweep_timing_path = joinpath(run_dir, "sweep_timing.csv")
        sweep_details_dir = joinpath(run_dir, "sweep_details")
        write_sweep_details = Bool(get(output_cfg, "write_sweep_details", true))
        sweep_extrema_mode = lowercase(String(get(output_cfg, "sweep_extrema_mode", "classes")))
        sweep_extrema_mode in ("classes", "trajectories") || error("output.sweep_extrema_mode must be 'classes' or 'trajectories'.")
        sweep_extrema_min_fraction = Float64(get(output_cfg, "sweep_extrema_min_fraction", 2.0e-4))
        sweep_extrema_max_classes = Int(get(output_cfg, "sweep_extrema_max_classes_per_value", 80))
        println("Sweep extrema output mode: $sweep_extrema_mode")
        println("Sweep extrema class filter: min_fraction=$sweep_extrema_min_fraction, max_classes_per_value=$sweep_extrema_max_classes")
        write_sweep_class_stats_header(sweep_stats_path)
        write_sweep_detail_manifest_header(sweep_details_path)
        write_sweep_timing_header(sweep_timing_path)
        write_sweep_details && mkpath(sweep_details_dir)

        batch_transient_requested = Bool(get(optimization_cfg, "batch_sweep_transient", false))
        batch_transient_used = false
        batch_transient_solve_seconds = 0.0
        batch_transient_extract_seconds = 0.0
        batched_transients = nothing
        if batch_transient_requested
            if supports_batched_sweep_transient(
                state_names,
                param_names,
                equations,
                sweep_parameter,
                integration_cfg,
                device,
                solver_mode,
                solver_name,
            )
                println("Optimization: batching sweep transient stage on GPU in phase-scaled time.")
                sweep_pvecs = build_sweep_pvecs(sweep_values, sweep_idx, param_values_f64, precision)
                tau_period = precision(2 * pi)
                tau_transient = precision(integration_cfg["transient_periods"]) * tau_period
                dtau = tau_period / precision(Int(integration_cfg["samples_per_period"]))
                Base.invokelatest(ensure_cuda_loaded)
                t_batch_solve_start = time()
                sol_a_batch = Base.invokelatest(
                    solve_stage_a_sweep_scaled_gpu,
                    u0_list,
                    sweep_pvecs,
                    n_total,
                    tau_transient,
                    dtau,
                )
                batch_transient_solve_seconds = time() - t_batch_solve_start
                t_batch_extract_start = time()
                batched_transients = extract_batched_transient_states(sol_a_batch, n_total, sweep_num)
                batch_transient_extract_seconds = time() - t_batch_extract_start
                batch_transient_used = true
                println(
                    "Optimization: batched transient solve=$(round(batch_transient_solve_seconds; digits=3))s, " *
                    "extract=$(round(batch_transient_extract_seconds; digits=3))s",
                )
            else
                println("Optimization: requested batched transient stage, but this run is not eligible; using the standard sweep loop.")
            end
        end

        sweep_rows = SweepExtremaRow[]
        g_amin = Float32(Inf)
        g_amax = Float32(-Inf)
        g_mmin = Float32(Inf)
        g_mmax = Float32(-Inf)
        g_smin = Float32(Inf)
        g_smax = Float32(-Inf)
        max_classes = 0
        total_extrema = 0
        for (k, sweep_value) in enumerate(sweep_values)
            t_iteration_start = time()
            println("Sweep $k/$sweep_num: $sweep_parameter=$sweep_value")
            params_i_f64 = copy(param_values_f64)
            params_i_f64[sweep_idx] = Float64(sweep_value)
            params_i = precision.(params_i_f64)
            pvec_i = SVector{length(params_i), precision}(Tuple(params_i))

            result_i = compute_operating_point(
                odefun,
                u0_list,
                pvec_i,
                nx,
                ny,
                n_total,
                observable_idx,
                zero_cross_idx,
                max_extrema,
                tol,
                k_fp,
                extrema_eps,
                integration_cfg,
                param_names,
                params_i_f64,
                precision,
                device,
                solver_mode,
                solver_name,
                abstol,
                reltol;
                verbose=false,
                precomputed_u_trans_list=batch_transient_used ? batched_transients[k] : nothing,
                collect_phase_samples=write_sweep_details,
                state_names=state_names,
                model_equations=runtime_equations,
            )

            t_collect_rows_start = time()
            local_bounds = collect_sweep_extrema_rows!(
                sweep_rows,
                Float64(sweep_value),
                result_i.labels_vec,
                result_i.extrema_all,
                sweep_extrema_mode,
                sweep_extrema_min_fraction,
                sweep_extrema_max_classes,
            )
            collect_rows_seconds = time() - t_collect_rows_start
            g_amin = min(g_amin, local_bounds.amin)
            g_amax = max(g_amax, local_bounds.amax)
            g_mmin = min(g_mmin, local_bounds.mmin)
            g_mmax = max(g_mmax, local_bounds.mmax)
            g_smin = min(g_smin, local_bounds.smin)
            g_smax = max(g_smax, local_bounds.smax)

            t_write_stats_start = time()
            append_sweep_class_stats(sweep_stats_path, Float64(sweep_value), result_i.counts, n_total)
            write_stats_seconds = time() - t_write_stats_start
            n_phase_samples = 0
            write_details_seconds = 0.0
            if write_sweep_details
                t_write_details_start = time()
                detail_prefix = "value_$(lpad(k, 4, '0'))"
                labels_i_path = joinpath(sweep_details_dir, "$(detail_prefix)_labels.csv")
                phase_i_path = joinpath(sweep_details_dir, "$(detail_prefix)_phase_samples.bin")
                writedlm(labels_i_path, result_i.labels_mat, ',')
                n_phase_samples = write_result_phase_samples(
                    phase_i_path,
                    result_i,
                    observable_idx,
                    zero_cross_idx;
                    binary=true,
                )
                append_sweep_detail_manifest(
                    sweep_details_path,
                    k,
                    Float64(sweep_value),
                    labels_i_path,
                    phase_i_path,
                    "binary_v1",
                    result_i.num_classes,
                    n_phase_samples,
                )
                write_details_seconds = time() - t_write_details_start
            end
            max_classes = max(max_classes, result_i.num_classes)
            n_extrema_i = local_bounds.rows_added
            total_extrema += n_extrema_i
            iteration_total_seconds = time() - t_iteration_start
            append_sweep_timing(
                sweep_timing_path,
                k,
                Float64(sweep_value),
                result_i,
                collect_rows_seconds,
                write_stats_seconds,
                write_details_seconds,
                iteration_total_seconds,
                n_extrema_i,
                n_phase_samples,
            )
            println("  classes=$(result_i.num_classes), extrema=$total_extrema")
        end

        color_bounds = finalize_sweep_color_bounds(g_amin, g_amax, g_mmin, g_mmax, g_smin, g_smax)
        println("Writing sweep extrema plot data ($(length(sweep_rows)) rows)")
        t_write_extrema_start = time()
        write_sweep_extrema(sweep_extrema_path, sweep_rows, color_bounds)
        write_extrema_seconds = time() - t_write_extrema_start

        elapsed = round(time() - started; digits=3)
        summary = Dict(
            "analysis" => Dict("mode" => "parameter_sweep"),
            "run" => Dict(
                "run_dir" => run_dir,
                "config_snapshot" => config_snapshot_path,
                "elapsed_seconds" => elapsed,
            ),
            "initial_condition_plane" => plane_summary,
            "integration" => Dict(
                "device" => device,
                "solver_mode" => solver_mode,
                "solver" => solver_name,
                "precision" => String(integration_cfg["precision"]),
                "custom_time_argument" => custom_time_argument,
                "configured_custom_time_argument" => configured_custom_time_argument,
                "custom_time_argument_auto" => custom_time_argument_auto,
            ),
            "optimization" => Dict(
                "batch_sweep_transient_requested" => batch_transient_requested,
                "batch_sweep_transient_used" => batch_transient_used,
                "batch_transient_solve_seconds" => round(batch_transient_solve_seconds; digits=6),
                "batch_transient_extract_seconds" => round(batch_transient_extract_seconds; digits=6),
            ),
            "sweep" => Dict(
                "parameter" => sweep_parameter,
                "min_value" => sweep_min,
                "max_value" => sweep_max,
                "num_values" => sweep_num,
            ),
            "sweep_color" => Dict(
                "method" => "global_feature_color",
                "amp_min" => Float64(color_bounds.amin),
                "amp_max" => Float64(color_bounds.amax),
                "mean_min" => Float64(color_bounds.mmin),
                "mean_max" => Float64(color_bounds.mmax),
                "std_min" => Float64(color_bounds.smin),
                "std_max" => Float64(color_bounds.smax),
            ),
            "results" => Dict(
                "nx" => nx,
                "ny" => ny,
                "n_trajectories_per_value" => n_total,
                "n_trajectories_total" => n_total * sweep_num,
                "num_values" => sweep_num,
                "max_classes" => max_classes,
                "num_extrema" => total_extrema,
                "sweep_extrema_mode" => sweep_extrema_mode,
                "sweep_extrema_min_fraction" => sweep_extrema_min_fraction,
                "sweep_extrema_max_classes_per_value" => sweep_extrema_max_classes,
                "sweep_details_saved" => write_sweep_details,
                "write_extrema_seconds" => round(write_extrema_seconds; digits=6),
            ),
            "paths" => Dict(
                "sweep_extrema_csv" => sweep_extrema_path,
                "class_stats_csv" => sweep_stats_path,
                "sweep_details_csv" => sweep_details_path,
                "sweep_timing_csv" => sweep_timing_path,
                "labels_csv" => "",
                "basin_image" => "",
                "phase_samples_csv" => "",
            ),
            "model" => Dict(
                "state_names" => state_names,
                "parameter_names" => param_names,
                "parameter_values" => param_values_f64,
                "uses_dynamic_model" => uses_dynamic_model,
            ),
        )

        open(summary_path, "w") do io
            TOML.print(io, summary)
        end

        println("Sweep extrema points: $total_extrema")
        println("Result folder: $run_dir")
        println("RESULT_DIR=$(run_dir)")
        println("RESULT_SUMMARY=$(summary_path)")
        return
    end

    if analysis_mode != "single_basin"
        error("Unknown analysis mode: $analysis_mode")
    end

    pvec = SVector{length(param_values), precision}(Tuple(param_values))
    result = compute_operating_point(
        odefun,
        u0_list,
        pvec,
        nx,
        ny,
        n_total,
        observable_idx,
        zero_cross_idx,
        max_extrema,
        tol,
        k_fp,
        extrema_eps,
        integration_cfg,
        param_names,
        param_values_f64,
        precision,
        device,
        solver_mode,
        solver_name,
        abstol,
        reltol;
        verbose=true,
        state_names=state_names,
        model_equations=runtime_equations,
    )

    elapsed = round(time() - started; digits=3)
    labels_path = joinpath(run_dir, "labels.csv")
    class_stats_path = joinpath(run_dir, "class_stats.csv")
    basin_image_path = joinpath(run_dir, "basin_map.ppm")
    phase_samples_path = joinpath(run_dir, "phase_samples.csv")

    write_class_stats(class_stats_path, result.counts, n_total; labels_vec=result.labels_vec, extrema_all=result.extrema_all)
    phase_sample_count = write_result_phase_samples(phase_samples_path, result, observable_idx, zero_cross_idx)

    if Bool(output_cfg["write_labels_csv"])
        writedlm(labels_path, result.labels_mat, ',')
    else
        labels_path = ""
    end

    if Bool(output_cfg["write_basin_image"])
        write_ppm(basin_image_path, result.labels_mat)
    else
        basin_image_path = ""
    end

    summary = Dict(
        "analysis" => Dict("mode" => "single_basin"),
        "run" => Dict(
            "run_dir" => run_dir,
            "config_snapshot" => config_snapshot_path,
            "elapsed_seconds" => elapsed,
        ),
        "initial_condition_plane" => plane_summary,
        "integration" => Dict(
            "device" => device,
            "solver_mode" => solver_mode,
            "solver" => solver_name,
            "precision" => String(integration_cfg["precision"]),
            "custom_time_argument" => custom_time_argument,
            "configured_custom_time_argument" => configured_custom_time_argument,
            "custom_time_argument_auto" => custom_time_argument_auto,
            "dt" => Float64(result.dt),
            "save_dt" => Float64(result.save_dt),
            "t_transient" => Float64(result.t_transient),
            "t_evaluation" => Float64(result.t_evaluation),
            "period" => Float64(result.period),
        ),
        "results" => Dict(
            "num_classes" => result.num_classes,
            "nx" => nx,
            "ny" => ny,
            "n_trajectories" => n_total,
        ),
        "paths" => Dict(
            "labels_csv" => labels_path,
            "class_stats_csv" => class_stats_path,
            "basin_image" => basin_image_path,
            "phase_samples_csv" => phase_samples_path,
        ),
        "phase" => Dict(
            "x_state" => state_names[observable_idx],
            "y_state" => state_names[zero_cross_idx],
            "max_labels" => 20,
            "samples_per_label" => 5,
            "n_samples" => phase_sample_count,
        ),
        "model" => Dict(
            "state_names" => state_names,
            "parameter_names" => param_names,
            "parameter_values" => param_values_f64,
            "uses_dynamic_model" => uses_dynamic_model,
        ),
    )

    open(summary_path, "w") do io
        TOML.print(io, summary)
    end

    println("Number of detected classes: $(result.num_classes)")
    println("Result folder: $run_dir")
    println("RESULT_DIR=$(run_dir)")
    println("RESULT_SUMMARY=$(summary_path)")
end


function run_server()
    println("__BACKEND_READY__")
    flush(stdout)
    for raw_line in eachline(stdin)
        config_path = String(strip(raw_line))
        isempty(config_path) && continue
        if config_path == "__quit__"
            println("__BACKEND_EXITING__")
            flush(stdout)
            return 0
        end

        return_code = 0
        try
            println("__JOB_STARTED__=$(config_path)")
            flush(stdout)
            run_config(config_path)
        catch err
            return_code = 1
            showerror(stdout, err)
            if !(err isa ModelValidationError)
                Base.show_backtrace(stdout, catch_backtrace())
            end
            println()
        end
        println("__JOB_DONE__=$(return_code)")
        flush(stdout)
    end
    return 0
end


function main()
    if length(ARGS) >= 1 && ARGS[1] == "--server"
        return run_server()
    end

    length(ARGS) >= 1 || error("Usage: julia basin_backend.jl <config.toml> | --server")
    run_config(ARGS[1])
    return 0
end


if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(main())
    catch err
        showerror(stdout, err)
        if !(err isa ModelValidationError)
            Base.show_backtrace(stdout, catch_backtrace())
        end
        println()
        exit(1)
    end
end
