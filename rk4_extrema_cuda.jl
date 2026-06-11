using CUDA
using StaticArrays


@inline function _duffing_rhs_scalars(x, v, U, w, F0, d, r1, r2, r3, t)
    T = typeof(x)
    return (
        v,
        -d * v - r1 * U + T(0.5) * x - T(0.5) * x * x * x + F0 * cos(w * t),
        r2 * v - r3 * U,
    )
end


@inline function _wrap_phase_near(phi)
    T = typeof(phi)
    two_pi = T(2 * pi)
    wrapped = phi >= two_pi ? phi - two_pi : phi
    return wrapped < zero(T) ? wrapped + two_pi : wrapped
end


@inline function _duffing_rhs_phase_arg_scalars(x, v, U, w, F0, d, r1, r2, r3, phase)
    T = typeof(x)
    return (
        v,
        -d * v - r1 * U + T(0.5) * x - T(0.5) * x * x * x + F0 * cos(phase),
        r2 * v - r3 * U,
    )
end


@inline function _duffing_rhs_phase_scalars(x, v, U, w, F0, d, r1, r2, r3, tau)
    T = typeof(x)
    invw = inv(w)
    return (
        v * invw,
        (-d * v - r1 * U + T(0.5) * x - T(0.5) * x * x * x + F0 * cos(tau)) * invw,
        (r2 * v - r3 * U) * invw,
    )
end


@inline function _rk4_duffing_step_phasewrapped_scalars(x, v, U, w, F0, d, r1, r2, r3, phase, dt, dphase)
    T = typeof(x)
    half = T(0.5)
    two = T(2)
    sixth = inv(T(6))
    half_dt = half * dt
    half_dphase = half * dphase

    k1x, k1v, k1U = _duffing_rhs_phase_arg_scalars(x, v, U, w, F0, d, r1, r2, r3, phase)
    k2x, k2v, k2U = _duffing_rhs_phase_arg_scalars(
        x + half_dt * k1x,
        v + half_dt * k1v,
        U + half_dt * k1U,
        w,
        F0,
        d,
        r1,
        r2,
        r3,
        _wrap_phase_near(phase + half_dphase),
    )
    k3x, k3v, k3U = _duffing_rhs_phase_arg_scalars(
        x + half_dt * k2x,
        v + half_dt * k2v,
        U + half_dt * k2U,
        w,
        F0,
        d,
        r1,
        r2,
        r3,
        _wrap_phase_near(phase + half_dphase),
    )
    k4x, k4v, k4U = _duffing_rhs_phase_arg_scalars(
        x + dt * k3x,
        v + dt * k3v,
        U + dt * k3U,
        w,
        F0,
        d,
        r1,
        r2,
        r3,
        _wrap_phase_near(phase + dphase),
    )

    return (
        x + dt * sixth * (k1x + two * k2x + two * k3x + k4x),
        v + dt * sixth * (k1v + two * k2v + two * k3v + k4v),
        U + dt * sixth * (k1U + two * k2U + two * k3U + k4U),
        _wrap_phase_near(phase + dphase),
    )
end


@inline function _rk4_duffing_step_scalars(x, v, U, w, F0, d, r1, r2, r3, t, dt)
    T = typeof(x)
    half = T(0.5)
    two = T(2)
    sixth = inv(T(6))

    k1x, k1v, k1U = _duffing_rhs_scalars(x, v, U, w, F0, d, r1, r2, r3, t)
    k2x, k2v, k2U = _duffing_rhs_scalars(
        x + half * dt * k1x,
        v + half * dt * k1v,
        U + half * dt * k1U,
        w,
        F0,
        d,
        r1,
        r2,
        r3,
        t + half * dt,
    )
    k3x, k3v, k3U = _duffing_rhs_scalars(
        x + half * dt * k2x,
        v + half * dt * k2v,
        U + half * dt * k2U,
        w,
        F0,
        d,
        r1,
        r2,
        r3,
        t + half * dt,
    )
    k4x, k4v, k4U = _duffing_rhs_scalars(
        x + dt * k3x,
        v + dt * k3v,
        U + dt * k3U,
        w,
        F0,
        d,
        r1,
        r2,
        r3,
        t + dt,
    )

    return (
        x + dt * sixth * (k1x + two * k2x + two * k3x + k4x),
        v + dt * sixth * (k1v + two * k2v + two * k3v + k4v),
        U + dt * sixth * (k1U + two * k2U + two * k3U + k4U),
    )
end


@inline function _rk4_duffing_phase_step_scalars(x, v, U, w, F0, d, r1, r2, r3, tau, dtau)
    T = typeof(x)
    half = T(0.5)
    two = T(2)
    sixth = inv(T(6))

    k1x, k1v, k1U = _duffing_rhs_phase_scalars(x, v, U, w, F0, d, r1, r2, r3, tau)
    k2x, k2v, k2U = _duffing_rhs_phase_scalars(
        x + half * dtau * k1x,
        v + half * dtau * k1v,
        U + half * dtau * k1U,
        w,
        F0,
        d,
        r1,
        r2,
        r3,
        tau + half * dtau,
    )
    k3x, k3v, k3U = _duffing_rhs_phase_scalars(
        x + half * dtau * k2x,
        v + half * dtau * k2v,
        U + half * dtau * k2U,
        w,
        F0,
        d,
        r1,
        r2,
        r3,
        tau + half * dtau,
    )
    k4x, k4v, k4U = _duffing_rhs_phase_scalars(
        x + dtau * k3x,
        v + dtau * k3v,
        U + dtau * k3U,
        w,
        F0,
        d,
        r1,
        r2,
        r3,
        tau + dtau,
    )

    return (
        x + dtau * sixth * (k1x + two * k2x + two * k3x + k4x),
        v + dtau * sixth * (k1v + two * k2v + two * k3v + k4v),
        U + dtau * sixth * (k1U + two * k2U + two * k3U + k4U),
    )
end


@inline function _state_component_3(x, v, U, idx::Int32)
    return idx == Int32(1) ? x : (idx == Int32(2) ? v : U)
end


function rk4_duffing_advance_kernel!(
    states,
    pvec,
    n_total::Int32,
    t_start,
    dt,
    n_steps::Int32,
)
    idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    if idx > n_total
        return nothing
    end

    x = states[1, idx]
    v = states[2, idx]
    U = states[3, idx]
    w = pvec[1]
    F0 = pvec[2]
    d = pvec[3]
    r1 = pvec[4]
    r2 = pvec[5]
    r3 = pvec[6]
    t = t_start

    for _step in Int32(1):n_steps
        x, v, U = _rk4_duffing_step_scalars(x, v, U, w, F0, d, r1, r2, r3, t, dt)
        t += dt
    end

    states[1, idx] = x
    states[2, idx] = v
    states[3, idx] = U
    return nothing
end


function rk4_duffing_advance_phasewrapped_kernel!(
    states,
    pvec,
    n_total::Int32,
    phase_start,
    dt,
    dphase,
    n_steps::Int32,
)
    idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    if idx > n_total
        return nothing
    end

    x = states[1, idx]
    v = states[2, idx]
    U = states[3, idx]
    w = pvec[1]
    F0 = pvec[2]
    d = pvec[3]
    r1 = pvec[4]
    r2 = pvec[5]
    r3 = pvec[6]
    phase = phase_start

    for _step in Int32(1):n_steps
        x, v, U, phase = _rk4_duffing_step_phasewrapped_scalars(x, v, U, w, F0, d, r1, r2, r3, phase, dt, dphase)
    end

    states[1, idx] = x
    states[2, idx] = v
    states[3, idx] = U
    return nothing
end


function rk4_duffing_phase_advance_kernel!(
    states,
    pvec,
    n_total::Int32,
    tau_start,
    dtau,
    n_steps::Int32,
)
    idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    if idx > n_total
        return nothing
    end

    x = states[1, idx]
    v = states[2, idx]
    U = states[3, idx]
    w = pvec[1]
    F0 = pvec[2]
    d = pvec[3]
    r1 = pvec[4]
    r2 = pvec[5]
    r3 = pvec[6]
    tau = tau_start

    for _step in Int32(1):n_steps
        x, v, U = _rk4_duffing_phase_step_scalars(x, v, U, w, F0, d, r1, r2, r3, tau, dtau)
        tau += dtau
    end

    states[1, idx] = x
    states[2, idx] = v
    states[3, idx] = U
    return nothing
end


function rk4_duffing_extrema_kernel!(
    u_trans,
    pvec,
    extrema_values,
    extrema_counts,
    n_total::Int32,
    t_transient,
    dt,
    total_steps::Int32,
    observable_idx::Int32,
    zero_cross_idx::Int32,
    max_extrema::Int32,
    eps,
)
    idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    if idx > n_total
        return nothing
    end

    x = u_trans[1, idx]
    v = u_trans[2, idx]
    U = u_trans[3, idx]
    w = pvec[1]
    F0 = pvec[2]
    d = pvec[3]
    r1 = pvec[4]
    r2 = pvec[5]
    r3 = pvec[6]
    t = t_transient

    count = Int32(0)
    if total_steps > Int32(1) && max_extrema > Int32(0)
        prev_x, prev_v, prev_U = _rk4_duffing_step_scalars(x, v, U, w, F0, d, r1, r2, r3, t, dt)
        t += dt

        for _step in Int32(2):total_steps
            curr_x, curr_v, curr_U = _rk4_duffing_step_scalars(prev_x, prev_v, prev_U, w, F0, d, r1, r2, r3, t, dt)
            t += dt

            z1 = _state_component_3(prev_x, prev_v, prev_U, zero_cross_idx)
            z2 = _state_component_3(curr_x, curr_v, curr_U, zero_cross_idx)
            if eps > zero(eps)
                z1 = abs(z1) < eps ? zero(z1) : z1
                z2 = abs(z2) < eps ? zero(z2) : z2
            end

            if z1 == zero(z1) || z1 * z2 < zero(z1)
                x1 = _state_component_3(prev_x, prev_v, prev_U, observable_idx)
                x2 = _state_component_3(curr_x, curr_v, curr_U, observable_idx)
                denom = z2 - z1
                alpha = denom == zero(denom) ? zero(denom) : (-z1 / denom)
                count += Int32(1)
                extrema_values[count, idx] = Float32(x1 + alpha * (x2 - x1))
                if count >= max_extrema
                    break
                end
            end

            prev_x = curr_x
            prev_v = curr_v
            prev_U = curr_U
        end
    end

    extrema_counts[idx] = count
    return nothing
end


function rk4_duffing_extrema_phasewrapped_kernel!(
    u_trans,
    pvec,
    extrema_values,
    extrema_counts,
    n_total::Int32,
    phase_start,
    dt,
    dphase,
    total_steps::Int32,
    observable_idx::Int32,
    zero_cross_idx::Int32,
    max_extrema::Int32,
    eps,
)
    idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    if idx > n_total
        return nothing
    end

    x = u_trans[1, idx]
    v = u_trans[2, idx]
    U = u_trans[3, idx]
    w = pvec[1]
    F0 = pvec[2]
    d = pvec[3]
    r1 = pvec[4]
    r2 = pvec[5]
    r3 = pvec[6]
    phase = phase_start

    count = Int32(0)
    if total_steps > Int32(1) && max_extrema > Int32(0)
        prev_x, prev_v, prev_U, phase = _rk4_duffing_step_phasewrapped_scalars(x, v, U, w, F0, d, r1, r2, r3, phase, dt, dphase)

        for _step in Int32(2):total_steps
            curr_x, curr_v, curr_U, phase = _rk4_duffing_step_phasewrapped_scalars(prev_x, prev_v, prev_U, w, F0, d, r1, r2, r3, phase, dt, dphase)

            z1 = _state_component_3(prev_x, prev_v, prev_U, zero_cross_idx)
            z2 = _state_component_3(curr_x, curr_v, curr_U, zero_cross_idx)
            if eps > zero(eps)
                z1 = abs(z1) < eps ? zero(z1) : z1
                z2 = abs(z2) < eps ? zero(z2) : z2
            end

            if z1 == zero(z1) || z1 * z2 < zero(z1)
                x1 = _state_component_3(prev_x, prev_v, prev_U, observable_idx)
                x2 = _state_component_3(curr_x, curr_v, curr_U, observable_idx)
                denom = z2 - z1
                alpha = denom == zero(denom) ? zero(denom) : (-z1 / denom)
                count += Int32(1)
                extrema_values[count, idx] = Float32(x1 + alpha * (x2 - x1))
                if count >= max_extrema
                    break
                end
            end

            prev_x = curr_x
            prev_v = curr_v
            prev_U = curr_U
        end
    end

    extrema_counts[idx] = count
    return nothing
end


function rk4_duffing_full_extrema_from_transient_kernel!(
    transient_states,
    pvec,
    extrema_values,
    extrema_counts,
    n_total::Int32,
    t_transient,
    dt,
    evaluation_steps::Int32,
    observable_idx::Int32,
    zero_cross_idx::Int32,
    max_extrema::Int32,
    eps,
)
    idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    if idx > n_total
        return nothing
    end

    x = transient_states[1, idx]
    v = transient_states[2, idx]
    U = transient_states[3, idx]
    w = pvec[1]
    F0 = pvec[2]
    d = pvec[3]
    r1 = pvec[4]
    r2 = pvec[5]
    r3 = pvec[6]
    t = t_transient

    count = Int32(0)
    if evaluation_steps > Int32(1) && max_extrema > Int32(0)
        prev_x, prev_v, prev_U = _rk4_duffing_step_scalars(x, v, U, w, F0, d, r1, r2, r3, t, dt)
        t += dt

        for _step in Int32(2):evaluation_steps
            curr_x, curr_v, curr_U = _rk4_duffing_step_scalars(prev_x, prev_v, prev_U, w, F0, d, r1, r2, r3, t, dt)
            t += dt

            z1 = _state_component_3(prev_x, prev_v, prev_U, zero_cross_idx)
            z2 = _state_component_3(curr_x, curr_v, curr_U, zero_cross_idx)
            if eps > zero(eps)
                z1 = abs(z1) < eps ? zero(z1) : z1
                z2 = abs(z2) < eps ? zero(z2) : z2
            end

            if z1 == zero(z1) || z1 * z2 < zero(z1)
                x1 = _state_component_3(prev_x, prev_v, prev_U, observable_idx)
                x2 = _state_component_3(curr_x, curr_v, curr_U, observable_idx)
                denom = z2 - z1
                alpha = denom == zero(denom) ? zero(denom) : (-z1 / denom)
                count += Int32(1)
                extrema_values[count, idx] = Float32(x1 + alpha * (x2 - x1))
                if count >= max_extrema
                    break
                end
            end

            prev_x = curr_x
            prev_v = curr_v
            prev_U = curr_U
        end
    end

    extrema_counts[idx] = count
    return nothing
end


function rk4_duffing_full_extrema_from_transient_phasewrapped_kernel!(
    transient_states,
    pvec,
    extrema_values,
    extrema_counts,
    n_total::Int32,
    phase_start,
    dt,
    dphase,
    evaluation_steps::Int32,
    observable_idx::Int32,
    zero_cross_idx::Int32,
    max_extrema::Int32,
    eps,
)
    idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    if idx > n_total
        return nothing
    end

    x = transient_states[1, idx]
    v = transient_states[2, idx]
    U = transient_states[3, idx]
    w = pvec[1]
    F0 = pvec[2]
    d = pvec[3]
    r1 = pvec[4]
    r2 = pvec[5]
    r3 = pvec[6]
    phase = phase_start

    count = Int32(0)
    if evaluation_steps > Int32(1) && max_extrema > Int32(0)
        prev_x, prev_v, prev_U, phase = _rk4_duffing_step_phasewrapped_scalars(x, v, U, w, F0, d, r1, r2, r3, phase, dt, dphase)

        for _step in Int32(2):evaluation_steps
            curr_x, curr_v, curr_U, phase = _rk4_duffing_step_phasewrapped_scalars(prev_x, prev_v, prev_U, w, F0, d, r1, r2, r3, phase, dt, dphase)

            z1 = _state_component_3(prev_x, prev_v, prev_U, zero_cross_idx)
            z2 = _state_component_3(curr_x, curr_v, curr_U, zero_cross_idx)
            if eps > zero(eps)
                z1 = abs(z1) < eps ? zero(z1) : z1
                z2 = abs(z2) < eps ? zero(z2) : z2
            end

            if z1 == zero(z1) || z1 * z2 < zero(z1)
                x1 = _state_component_3(prev_x, prev_v, prev_U, observable_idx)
                x2 = _state_component_3(curr_x, curr_v, curr_U, observable_idx)
                denom = z2 - z1
                alpha = denom == zero(denom) ? zero(denom) : (-z1 / denom)
                count += Int32(1)
                extrema_values[count, idx] = Float32(x1 + alpha * (x2 - x1))
                if count >= max_extrema
                    break
                end
            end

            prev_x = curr_x
            prev_v = curr_v
            prev_U = curr_U
        end
    end

    extrema_counts[idx] = count
    return nothing
end


function rk4_duffing_phase_extrema_from_transient_kernel!(
    transient_states,
    pvec,
    extrema_values,
    extrema_counts,
    n_total::Int32,
    tau_transient,
    dtau,
    evaluation_steps::Int32,
    observable_idx::Int32,
    zero_cross_idx::Int32,
    max_extrema::Int32,
    eps,
)
    idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    if idx > n_total
        return nothing
    end

    x = transient_states[1, idx]
    v = transient_states[2, idx]
    U = transient_states[3, idx]
    w = pvec[1]
    F0 = pvec[2]
    d = pvec[3]
    r1 = pvec[4]
    r2 = pvec[5]
    r3 = pvec[6]
    tau = tau_transient

    count = Int32(0)
    if evaluation_steps > Int32(1) && max_extrema > Int32(0)
        prev_x, prev_v, prev_U = _rk4_duffing_phase_step_scalars(x, v, U, w, F0, d, r1, r2, r3, tau, dtau)
        tau += dtau

        for _step in Int32(2):evaluation_steps
            curr_x, curr_v, curr_U = _rk4_duffing_phase_step_scalars(prev_x, prev_v, prev_U, w, F0, d, r1, r2, r3, tau, dtau)
            tau += dtau

            z1 = _state_component_3(prev_x, prev_v, prev_U, zero_cross_idx)
            z2 = _state_component_3(curr_x, curr_v, curr_U, zero_cross_idx)
            if eps > zero(eps)
                z1 = abs(z1) < eps ? zero(z1) : z1
                z2 = abs(z2) < eps ? zero(z2) : z2
            end

            if z1 == zero(z1) || z1 * z2 < zero(z1)
                x1 = _state_component_3(prev_x, prev_v, prev_U, observable_idx)
                x2 = _state_component_3(curr_x, curr_v, curr_U, observable_idx)
                denom = z2 - z1
                alpha = denom == zero(denom) ? zero(denom) : (-z1 / denom)
                count += Int32(1)
                extrema_values[count, idx] = Float32(x1 + alpha * (x2 - x1))
                if count >= max_extrema
                    break
                end
            end

            prev_x = curr_x
            prev_v = curr_v
            prev_U = curr_U
        end
    end

    extrema_counts[idx] = count
    return nothing
end


function stage_b_total_steps(t_evaluation, dt)
    return max(1, round(Int, Float64(t_evaluation) / Float64(dt)))
end


function rk4_step_count(duration, dt)
    return max(0, round(Int, Float64(duration) / Float64(dt)))
end


function rk4_kernel_max_steps()
    return 2_000
end


function rk4_phase_from_time(pvec, t_start, ::Type{T}) where {T}
    return T(mod(Float64(pvec[1]) * Float64(t_start), 2 * pi))
end


function rk4_advance_phase_host(phase, dphase, steps::Int, ::Type{T}) where {T}
    return T(mod(Float64(phase) + Float64(dphase) * steps, 2 * pi))
end


function rk4_matrix_to_svector_list(u_mat::Matrix{T}) where {T}
    n_total = size(u_mat, 2)
    states = Vector{SVector{3, T}}(undef, n_total)
    @inbounds for i in 1:n_total
        states[i] = SVector{3, T}(u_mat[1, i], u_mat[2, i], u_mat[3, i])
    end
    return states
end


function rk4_extrema_matrix_to_vectors(extrema_mat::Matrix{Float32}, counts::Vector{Int32})
    n_total = length(counts)
    extrema_all = Vector{Vector{Float32}}(undef, n_total)
    @inbounds for i in 1:n_total
        count = Int(counts[i])
        values = Vector{Float32}(undef, count)
        for j in 1:count
            values[j] = extrema_mat[j, i]
        end
        extrema_all[i] = values
    end
    return extrema_all
end


@inline function _rk4_generic_step(odefun, u::SVector{N, T}, pvec, t, dt) where {N, T}
    half = T(0.5)
    two = T(2)
    sixth = inv(T(6))
    half_dt = half * dt

    k1 = odefun(u, pvec, t)
    k2 = odefun(u + half_dt * k1, pvec, t + half_dt)
    k3 = odefun(u + half_dt * k2, pvec, t + half_dt)
    k4 = odefun(u + dt * k3, pvec, t + dt)
    return u + dt * sixth * (k1 + two * k2 + two * k3 + k4)
end


@inline function _load_state_generic(states, idx::Int32, ::Val{N}) where {N}
    return SVector{N}(ntuple(i -> states[i, idx], Val(N)))
end


@inline function _store_state_generic!(states, u::SVector{N, T}, idx::Int32, ::Val{N}) where {N, T}
    @inbounds for i in 1:N
        states[i, idx] = u[i]
    end
    return nothing
end


@inline function _state_component_generic(u::SVector{N, T}, idx::Int32) where {N, T}
    value = u[1]
    @inbounds for i in 2:N
        value = ifelse(idx == Int32(i), u[i], value)
    end
    return value
end


function rk4_generic_advance_kernel!(
    odefun,
    states,
    pvec,
    n_total::Int32,
    t_start,
    dt,
    n_steps::Int32,
    ::Val{N},
) where {N}
    idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    if idx > n_total
        return nothing
    end

    u = _load_state_generic(states, idx, Val(N))
    t = t_start
    for _step in Int32(1):n_steps
        u = _rk4_generic_step(odefun, u, pvec, t, dt)
        t += dt
    end
    _store_state_generic!(states, u, idx, Val(N))
    return nothing
end


function rk4_generic_extrema_kernel!(
    odefun,
    u_trans,
    pvec,
    extrema_values,
    extrema_counts,
    n_total::Int32,
    t_transient,
    dt,
    total_steps::Int32,
    observable_idx::Int32,
    zero_cross_idx::Int32,
    max_extrema::Int32,
    eps,
    ::Val{N},
) where {N}
    idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    if idx > n_total
        return nothing
    end

    u = _load_state_generic(u_trans, idx, Val(N))
    t = t_transient
    count = Int32(0)

    if total_steps > Int32(1) && max_extrema > Int32(0)
        prev_u = _rk4_generic_step(odefun, u, pvec, t, dt)
        t += dt

        for _step in Int32(2):total_steps
            curr_u = _rk4_generic_step(odefun, prev_u, pvec, t, dt)
            t += dt

            z1 = _state_component_generic(prev_u, zero_cross_idx)
            z2 = _state_component_generic(curr_u, zero_cross_idx)
            if eps > zero(eps)
                z1 = abs(z1) < eps ? zero(z1) : z1
                z2 = abs(z2) < eps ? zero(z2) : z2
            end

            if z1 == zero(z1) || z1 * z2 < zero(z1)
                x1 = _state_component_generic(prev_u, observable_idx)
                x2 = _state_component_generic(curr_u, observable_idx)
                denom = z2 - z1
                alpha = denom == zero(denom) ? zero(denom) : (-z1 / denom)
                count += Int32(1)
                extrema_values[count, idx] = Float32(x1 + alpha * (x2 - x1))
                if count >= max_extrema
                    break
                end
            end

            prev_u = curr_u
        end
    end

    extrema_counts[idx] = count
    return nothing
end


function rk4_generic_full_extrema_from_transient_kernel!(
    odefun,
    transient_states,
    pvec,
    extrema_values,
    extrema_counts,
    n_total::Int32,
    t_transient,
    dt,
    evaluation_steps::Int32,
    observable_idx::Int32,
    zero_cross_idx::Int32,
    max_extrema::Int32,
    eps,
    ::Val{N},
) where {N}
    idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    if idx > n_total
        return nothing
    end

    u = _load_state_generic(transient_states, idx, Val(N))
    t = t_transient
    count = Int32(0)

    if evaluation_steps > Int32(1) && max_extrema > Int32(0)
        prev_u = _rk4_generic_step(odefun, u, pvec, t, dt)
        t += dt

        for _step in Int32(2):evaluation_steps
            curr_u = _rk4_generic_step(odefun, prev_u, pvec, t, dt)
            t += dt

            z1 = _state_component_generic(prev_u, zero_cross_idx)
            z2 = _state_component_generic(curr_u, zero_cross_idx)
            if eps > zero(eps)
                z1 = abs(z1) < eps ? zero(z1) : z1
                z2 = abs(z2) < eps ? zero(z2) : z2
            end

            if z1 == zero(z1) || z1 * z2 < zero(z1)
                x1 = _state_component_generic(prev_u, observable_idx)
                x2 = _state_component_generic(curr_u, observable_idx)
                denom = z2 - z1
                alpha = denom == zero(denom) ? zero(denom) : (-z1 / denom)
                count += Int32(1)
                extrema_values[count, idx] = Float32(x1 + alpha * (x2 - x1))
                if count >= max_extrema
                    break
                end
            end

            prev_u = curr_u
        end
    end

    extrema_counts[idx] = count
    return nothing
end


function rk4_matrix_to_svector_list(u_mat::Matrix{T}, ::Val{N}) where {T, N}
    n_total = size(u_mat, 2)
    states = Vector{SVector{N, T}}(undef, n_total)
    @inbounds for i in 1:n_total
        states[i] = SVector{N, T}(ntuple(j -> u_mat[j, i], Val(N)))
    end
    return states
end


function rk4_state_matrix(u_list, ::Val{N}) where {N}
    T = eltype(u_list[1])
    n_total = length(u_list)
    u_host = Matrix{T}(undef, N, n_total)
    @inbounds for i in 1:n_total
        u = u_list[i]
        for j in 1:N
            u_host[j, i] = u[j]
        end
    end
    return u_host
end


function check_generic_rk4_inputs(u_list, pvec, dt, save_dt, max_extrema::Int, label::String)
    CUDA.functional() || error("CUDA is not functional, so the $label run is not possible.")
    isempty(u_list) && return 0
    T = eltype(u_list[1])
    T in (Float32, Float64) || error("$label currently supports Float32 and Float64 only.")
    nstates = length(u_list[1])
    2 <= nstates <= 8 || error("$label currently supports 2 to 8 states.")
    max_extrema >= 1 || error("classification.max_extrema must be at least 1 for $label.")
    abs(Float64(dt) - Float64(save_dt)) <= 10 * Base.eps(Float64(dt)) || error("$label currently requires dt == save_dt.")
    return nstates
end


const CUSTOM_RK4_FULL_EXTREMA_COLLECTORS = Dict{String, Function}()


function custom_rk4_full_extrema_key(state_names::Vector{String}, param_names::Vector{String}, equations::Vector{String}, time_argument::String)
    return join(vcat([time_argument], state_names, ["|"], param_names, ["|"], normalize_expr_string.(equations)), "\x1f")
end


function inline_function_expr(function_expr)
    return Expr(
        :macrocall,
        Symbol("@inline"),
        LineNumberNode(0, Symbol("rk4_extrema_cuda.jl")),
        function_expr,
    )
end


function tuple_assignment_expr(lhs, rhs)
    return Expr(:(=), Expr(:tuple, lhs...), Expr(:tuple, rhs...))
end


function tuple_call_assignment_expr(lhs, fname::Symbol, args...)
    return Expr(:(=), Expr(:tuple, lhs...), Expr(:call, fname, args...))
end


function build_specialized_rk4_full_extrema_collector(
    state_names::Vector{String},
    param_names::Vector{String},
    equations::Vector{String},
    time_argument::String="time",
)
    time_argument_l = lowercase(time_argument)
    time_argument_l in ("time", "phase") || error("custom RK4 time argument must be 'time' or 'phase'.")
    use_phase_argument = time_argument_l == "phase"
    key = custom_rk4_full_extrema_key(state_names, param_names, equations, time_argument_l)
    cached = get(CUSTOM_RK4_FULL_EXTREMA_COLLECTORS, key, nothing)
    cached !== nothing && return cached

    validate_model_identifiers(state_names, param_names)
    allowed_symbols = Set(Symbol.(vcat(state_names, param_names, ["t", "phase"])))
    rhs_exprs = Any[]
    for equation_text in equations
        rhs_expr = Meta.parse(equation_text)
        validate_rhs_expr!(rhs_expr, allowed_symbols, equation_text)
        push!(rhs_exprs, typed_numeric_literals_expr(rhs_expr))
    end
    nstates = length(state_names)
    2 <= nstates <= 8 || error("specialized custom full GPU RK4 extrema currently supports 2 to 8 states.")

    stamp = string(time_ns())
    rhs_name = Symbol("_rk4_custom_rhs_", stamp)
    step_name = Symbol("_rk4_custom_step_", stamp)
    component_name = Symbol("_rk4_custom_component_", stamp)
    advance_kernel_name = Symbol("_rk4_custom_advance_kernel_", stamp, "!")
    extrema_kernel_name = Symbol("_rk4_custom_full_extrema_kernel_", stamp, "!")
    collector_name = Symbol("_collect_full_extrema_rk4_gpu_custom_", stamp)

    rhs_state_args = [gensym(:s) for _ in 1:nstates]
    rhs_param_args = [gensym(:par) for _ in eachindex(param_names)]
    rhs_t_arg = gensym(:t)
    rhs_phase_arg = gensym(:phase)
    rhs_assignments = Expr[
        Expr(:(=), Symbol(state_names[i]), rhs_state_args[i]) for i in 1:nstates
    ]
    push!(rhs_assignments, Expr(:(=), :t, rhs_t_arg))
    push!(rhs_assignments, Expr(:(=), :phase, rhs_phase_arg))
    append!(
        rhs_assignments,
        Expr[
            Expr(:(=), Symbol(param_names[i]), rhs_param_args[i]) for i in eachindex(param_names)
        ],
    )
    rhs_returns = [:(T($(rhs_exprs[i]))) for i in 1:nstates]
    rhs_body = Expr(
        :block,
        :(T = typeof($(rhs_state_args[1]))),
        rhs_assignments...,
        :(return $(Expr(:tuple, rhs_returns...))),
    )
    rhs_def = inline_function_expr(Expr(:function, Expr(:call, rhs_name, rhs_state_args..., rhs_param_args..., rhs_t_arg, rhs_phase_arg), rhs_body))

    step_state_args = [gensym(:s) for _ in 1:nstates]
    step_param_args = [gensym(:par) for _ in eachindex(param_names)]
    step_t_arg = gensym(:t)
    step_phase_arg = gensym(:phase)
    step_dt_arg = gensym(:dt)
    step_dphase_arg = gensym(:dphase)
    k1 = [gensym(:k1) for _ in 1:nstates]
    k2 = [gensym(:k2) for _ in 1:nstates]
    k3 = [gensym(:k3) for _ in 1:nstates]
    k4 = [gensym(:k4) for _ in 1:nstates]
    k2_states = [:($(step_state_args[i]) + half_dt * $(k1[i])) for i in 1:nstates]
    k3_states = [:($(step_state_args[i]) + half_dt * $(k2[i])) for i in 1:nstates]
    k4_states = [:($(step_state_args[i]) + $(step_dt_arg) * $(k3[i])) for i in 1:nstates]
    step_returns = [
        :($(step_state_args[i]) + $(step_dt_arg) * sixth * ($(k1[i]) + two * $(k2[i]) + two * $(k3[i]) + $(k4[i])))
        for i in 1:nstates
    ]
    half_phase_expr = use_phase_argument ? :(_wrap_phase_near($(step_phase_arg) + half_dphase)) : :($(step_t_arg) + half_dt)
    full_phase_expr = use_phase_argument ? :(_wrap_phase_near($(step_phase_arg) + $(step_dphase_arg))) : :($(step_t_arg) + $(step_dt_arg))
    step_body_parts = Any[
        :(T = typeof($(step_state_args[1]))),
        :(half = T(0.5)),
        :(two = T(2)),
        :(sixth = inv(T(6))),
        :(half_dt = half * $(step_dt_arg)),
    ]
    if use_phase_argument
        push!(step_body_parts, :(half_dphase = half * $(step_dphase_arg)))
    end
    append!(
        step_body_parts,
        Any[
            tuple_call_assignment_expr(k1, rhs_name, step_state_args..., step_param_args..., step_t_arg, use_phase_argument ? step_phase_arg : step_t_arg),
            tuple_call_assignment_expr(k2, rhs_name, k2_states..., step_param_args..., :($(step_t_arg) + half_dt), half_phase_expr),
            tuple_call_assignment_expr(k3, rhs_name, k3_states..., step_param_args..., :($(step_t_arg) + half_dt), half_phase_expr),
            tuple_call_assignment_expr(k4, rhs_name, k4_states..., step_param_args..., :($(step_t_arg) + $(step_dt_arg)), full_phase_expr),
        ],
    )
    step_return_values = use_phase_argument ? vcat(step_returns, [full_phase_expr]) : step_returns
    push!(step_body_parts, :(return $(Expr(:tuple, step_return_values...))))
    step_body = Expr(
        :block,
        step_body_parts...,
    )
    step_signature = use_phase_argument ?
                     Expr(:call, step_name, step_state_args..., step_param_args..., step_t_arg, step_phase_arg, step_dt_arg, step_dphase_arg) :
                     Expr(:call, step_name, step_state_args..., step_param_args..., step_t_arg, step_dt_arg)
    step_def = inline_function_expr(Expr(:function, step_signature, step_body))

    component_state_args = [gensym(:s) for _ in 1:nstates]
    component_idx_arg = gensym(:idx)
    component_updates = [
        :(value = ifelse($(component_idx_arg) == Int32($i), $(component_state_args[i]), value))
        for i in 2:nstates
    ]
    component_body = Expr(
        :block,
        :(value = $(component_state_args[1])),
        component_updates...,
        :(return value),
    )
    component_def = inline_function_expr(Expr(:function, Expr(:call, component_name, component_idx_arg, component_state_args...), component_body))

    state_vars = [gensym(:state) for _ in 1:nstates]
    param_vars = [gensym(:param) for _ in eachindex(param_names)]
    load_state = [Expr(:(=), state_vars[i], :(states[$i, idx])) for i in 1:nstates]
    load_params = [Expr(:(=), param_vars[i], :(pvec[$i])) for i in eachindex(param_names)]
    store_state = [:(states[$i, idx] = $(state_vars[i])) for i in 1:nstates]
    advance_phase_lhs = vcat(state_vars, [:phase])
    advance_step_expr = use_phase_argument ?
                        tuple_call_assignment_expr(advance_phase_lhs, step_name, state_vars..., param_vars..., :t, :phase, :dt, :dphase) :
                        tuple_call_assignment_expr(state_vars, step_name, state_vars..., param_vars..., :t, :dt)
    advance_phase_init = use_phase_argument ? [:(phase = phase_start)] : Expr[]
    advance_prefix = quote
        idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
        if idx > n_total
            return nothing
        end
    end
    advance_loop = quote
        for _step in Int32(1):n_steps
            $advance_step_expr
            t += dt
        end
    end
    advance_body = Expr(
        :block,
        advance_prefix.args...,
        load_state...,
        load_params...,
        :(t = t_start),
        advance_phase_init...,
        advance_loop.args...,
        store_state...,
        :(return nothing),
    )
    advance_signature = use_phase_argument ?
                        Expr(:call, advance_kernel_name, :states, :pvec, :(n_total::Int32), :t_start, :phase_start, :dt, :dphase, :(n_steps::Int32)) :
                        Expr(:call, advance_kernel_name, :states, :pvec, :(n_total::Int32), :t_start, :dt, :(n_steps::Int32))
    advance_kernel_def = Expr(:function, advance_signature, advance_body)

    eval_state_vars = [gensym(:state) for _ in 1:nstates]
    eval_param_vars = [gensym(:param) for _ in eachindex(param_names)]
    prev_vars = [gensym(:prev) for _ in 1:nstates]
    curr_vars = [gensym(:curr) for _ in 1:nstates]
    load_eval_state = [Expr(:(=), eval_state_vars[i], :(transient_states[$i, idx])) for i in 1:nstates]
    load_eval_params = [Expr(:(=), eval_param_vars[i], :(pvec[$i])) for i in eachindex(param_names)]
    eval_prev_lhs = use_phase_argument ? vcat(prev_vars, [:phase]) : prev_vars
    eval_curr_lhs = use_phase_argument ? vcat(curr_vars, [:phase]) : curr_vars
    eval_prev_step_expr = use_phase_argument ?
                          tuple_call_assignment_expr(eval_prev_lhs, step_name, eval_state_vars..., eval_param_vars..., :t, :phase, :dt, :dphase) :
                          tuple_call_assignment_expr(prev_vars, step_name, eval_state_vars..., eval_param_vars..., :t, :dt)
    eval_curr_step_expr = use_phase_argument ?
                          tuple_call_assignment_expr(eval_curr_lhs, step_name, prev_vars..., eval_param_vars..., :t, :phase, :dt, :dphase) :
                          tuple_call_assignment_expr(curr_vars, step_name, prev_vars..., eval_param_vars..., :t, :dt)
    eval_phase_init = use_phase_argument ? [:(phase = phase_start)] : Expr[]
    extrema_prefix = quote
        idx = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
        if idx > n_total
            return nothing
        end
    end

    extrema_loop = quote
        if evaluation_steps > Int32(1) && max_extrema > Int32(0)
            $eval_prev_step_expr
            t += dt

            for _step in Int32(2):evaluation_steps
                $eval_curr_step_expr
                t += dt

                z1 = $(Expr(:call, component_name, :zero_cross_idx, prev_vars...))
                z2 = $(Expr(:call, component_name, :zero_cross_idx, curr_vars...))
                if eps > zero(eps)
                    z1 = abs(z1) < eps ? zero(z1) : z1
                    z2 = abs(z2) < eps ? zero(z2) : z2
                end

                if z1 == zero(z1) || z1 * z2 < zero(z1)
                    x1 = $(Expr(:call, component_name, :observable_idx, prev_vars...))
                    x2 = $(Expr(:call, component_name, :observable_idx, curr_vars...))
                    denom = z2 - z1
                    alpha = denom == zero(denom) ? zero(denom) : (-z1 / denom)
                    count += Int32(1)
                    extrema_values[count, idx] = Float32(x1 + alpha * (x2 - x1))
                    if count >= max_extrema
                        break
                    end
                end

                $(tuple_assignment_expr(prev_vars, curr_vars))
            end
        end

        extrema_counts[idx] = count
        return nothing
    end
    extrema_body = Expr(
        :block,
        extrema_prefix.args...,
        load_eval_state...,
        load_eval_params...,
        :(t = t_transient),
        eval_phase_init...,
        :(count = Int32(0)),
        extrema_loop.args...,
    )
    extrema_signature = use_phase_argument ?
                        Expr(:call, extrema_kernel_name, :transient_states, :pvec, :extrema_values, :extrema_counts, :(n_total::Int32), :t_transient, :phase_start, :dt, :dphase, :(evaluation_steps::Int32), :(observable_idx::Int32), :(zero_cross_idx::Int32), :(max_extrema::Int32), :eps) :
                        Expr(:call, extrema_kernel_name, :transient_states, :pvec, :extrema_values, :extrema_counts, :(n_total::Int32), :t_transient, :dt, :(evaluation_steps::Int32), :(observable_idx::Int32), :(zero_cross_idx::Int32), :(max_extrema::Int32), :eps)
    extrema_kernel_def = Expr(:function, extrema_signature, extrema_body)

    collector_def = quote
        function $(collector_name)(
            u0_list,
            pvec,
            t_transient,
            t_evaluation,
            dt,
            save_dt,
            period,
            observable_idx::Int,
            zero_cross_idx::Int,
            max_extrema::Int,
            eps,
        )
            nstates = check_generic_rk4_inputs(u0_list, pvec, dt, save_dt, max_extrema, "specialized custom full GPU RK4 extrema")
            nstates == 0 && return Vector{Vector{Float32}}(), typeof(u0_list)()
            nstates == $nstates || error("specialized custom full GPU RK4 extrema was generated for " * string($nstates) * " states but received " * string(nstates) * ".")
            val_n = Val($nstates)
            T = eltype(u0_list[1])
            n_total = length(u0_list)
            states_dev = CuArray(rk4_state_matrix(u0_list, val_n))
            p_dev = CuArray(collect(T.(pvec)))
            extrema_dev = CUDA.fill(Float32(NaN), max_extrema, n_total)
            counts_dev = CUDA.zeros(Int32, n_total)
            transient_steps = rk4_step_count(t_transient, dt)
            evaluation_steps = stage_b_total_steps(t_evaluation, dt)
            threads = 128
            blocks = cld(n_total, threads)
            use_phase_argument = $use_phase_argument
            if use_phase_argument && !(T(period) > zero(T))
                error("specialized custom full GPU RK4 extrema with phase argument requires a positive period.")
            end
            dphase = use_phase_argument ? (T(2 * pi) * T(dt) / T(period)) : zero(T)

            t_current = zero(T)
            phase_current = zero(T)
            remaining = transient_steps
            max_chunk_steps = rk4_kernel_max_steps()
            while remaining > 0
                steps_this_chunk = min(max_chunk_steps, remaining)
                if use_phase_argument
                    @cuda threads=threads blocks=blocks $(advance_kernel_name)(
                        states_dev,
                        p_dev,
                        Int32(n_total),
                        t_current,
                        phase_current,
                        T(dt),
                        dphase,
                        Int32(steps_this_chunk),
                    )
                else
                    @cuda threads=threads blocks=blocks $(advance_kernel_name)(
                        states_dev,
                        p_dev,
                        Int32(n_total),
                        t_current,
                        T(dt),
                        Int32(steps_this_chunk),
                    )
                end
                CUDA.synchronize()
                t_current += T(dt) * T(steps_this_chunk)
                if use_phase_argument
                    phase_current = rk4_advance_phase_host(phase_current, dphase, steps_this_chunk, T)
                end
                remaining -= steps_this_chunk
            end

            if use_phase_argument
                @cuda threads=threads blocks=blocks $(extrema_kernel_name)(
                    states_dev,
                    p_dev,
                    extrema_dev,
                    counts_dev,
                    Int32(n_total),
                    T(t_transient),
                    phase_current,
                    T(dt),
                    dphase,
                    Int32(evaluation_steps),
                    Int32(observable_idx),
                    Int32(zero_cross_idx),
                    Int32(max_extrema),
                    T(eps),
                )
            else
                @cuda threads=threads blocks=blocks $(extrema_kernel_name)(
                    states_dev,
                    p_dev,
                    extrema_dev,
                    counts_dev,
                    Int32(n_total),
                    T(t_transient),
                    T(dt),
                    Int32(evaluation_steps),
                    Int32(observable_idx),
                    Int32(zero_cross_idx),
                    Int32(max_extrema),
                    T(eps),
                )
            end
            CUDA.synchronize()

            transient_states = rk4_matrix_to_svector_list(Array(states_dev), val_n)
            extrema_all = rk4_extrema_matrix_to_vectors(Array(extrema_dev), Array(counts_dev))
            return extrema_all, transient_states
        end
    end

    Core.eval(
        @__MODULE__,
        Expr(
            :block,
            rhs_def,
            step_def,
            component_def,
            advance_kernel_def,
            extrema_kernel_def,
            collector_def,
        ),
    )
    collector = Base.invokelatest(getfield, @__MODULE__, collector_name)
    CUSTOM_RK4_FULL_EXTREMA_COLLECTORS[key] = collector
    return collector
end


function collect_stage_b_extrema_rk4_gpu_generic(
    odefun,
    u_trans_list,
    pvec,
    t_transient,
    t_evaluation,
    dt,
    save_dt,
    observable_idx::Int,
    zero_cross_idx::Int,
    max_extrema::Int,
    eps,
)
    nstates = check_generic_rk4_inputs(u_trans_list, pvec, dt, save_dt, max_extrema, "generic GPU RK4 extrema")
    nstates == 0 && return Vector{Vector{Float32}}()
    val_n = Val(nstates)
    T = eltype(u_trans_list[1])
    n_total = length(u_trans_list)
    u_dev = CuArray(rk4_state_matrix(u_trans_list, val_n))
    p_dev = CuArray(collect(T.(pvec)))
    extrema_dev = CUDA.fill(Float32(NaN), max_extrema, n_total)
    counts_dev = CUDA.zeros(Int32, n_total)
    total_steps = stage_b_total_steps(t_evaluation, dt)
    threads = 128
    blocks = cld(n_total, threads)

    @cuda threads=threads blocks=blocks rk4_generic_extrema_kernel!(
        odefun,
        u_dev,
        p_dev,
        extrema_dev,
        counts_dev,
        Int32(n_total),
        T(t_transient),
        T(dt),
        Int32(total_steps),
        Int32(observable_idx),
        Int32(zero_cross_idx),
        Int32(max_extrema),
        T(eps),
        val_n,
    )
    CUDA.synchronize()

    return rk4_extrema_matrix_to_vectors(Array(extrema_dev), Array(counts_dev))
end


function collect_full_extrema_rk4_gpu_generic(
    odefun,
    u0_list,
    pvec,
    t_transient,
    t_evaluation,
    dt,
    save_dt,
    observable_idx::Int,
    zero_cross_idx::Int,
    max_extrema::Int,
    eps,
)
    nstates = check_generic_rk4_inputs(u0_list, pvec, dt, save_dt, max_extrema, "generic full GPU RK4 extrema")
    nstates == 0 && return Vector{Vector{Float32}}(), typeof(u0_list)()
    val_n = Val(nstates)
    T = eltype(u0_list[1])
    n_total = length(u0_list)
    states_dev = CuArray(rk4_state_matrix(u0_list, val_n))
    p_dev = CuArray(collect(T.(pvec)))
    extrema_dev = CUDA.fill(Float32(NaN), max_extrema, n_total)
    counts_dev = CUDA.zeros(Int32, n_total)
    transient_steps = rk4_step_count(t_transient, dt)
    evaluation_steps = stage_b_total_steps(t_evaluation, dt)
    threads = 128
    blocks = cld(n_total, threads)

    t_current = zero(T)
    remaining = transient_steps
    max_chunk_steps = rk4_kernel_max_steps()
    while remaining > 0
        steps_this_chunk = min(max_chunk_steps, remaining)
        @cuda threads=threads blocks=blocks rk4_generic_advance_kernel!(
            odefun,
            states_dev,
            p_dev,
            Int32(n_total),
            t_current,
            T(dt),
            Int32(steps_this_chunk),
            val_n,
        )
        CUDA.synchronize()
        t_current += T(dt) * T(steps_this_chunk)
        remaining -= steps_this_chunk
    end

    @cuda threads=threads blocks=blocks rk4_generic_full_extrema_from_transient_kernel!(
        odefun,
        states_dev,
        p_dev,
        extrema_dev,
        counts_dev,
        Int32(n_total),
        T(t_transient),
        T(dt),
        Int32(evaluation_steps),
        Int32(observable_idx),
        Int32(zero_cross_idx),
        Int32(max_extrema),
        T(eps),
        val_n,
    )
    CUDA.synchronize()

    transient_states = rk4_matrix_to_svector_list(Array(states_dev), val_n)
    extrema_all = rk4_extrema_matrix_to_vectors(Array(extrema_dev), Array(counts_dev))
    return extrema_all, transient_states
end


function collect_stage_b_extrema_rk4_gpu_default_duffing(
    u_trans_list,
    pvec,
    t_transient,
    t_evaluation,
    dt,
    save_dt,
    observable_idx::Int,
    zero_cross_idx::Int,
    max_extrema::Int,
    eps,
)
    CUDA.functional() || error("CUDA is not functional, so the RK4 extrema GPU run is not possible.")
    isempty(u_trans_list) && return Vector{Vector{Float32}}()
    T = eltype(u_trans_list[1])
    T == Float32 || error("GPU RK4 extrema currently supports Float32 only.")
    length(u_trans_list[1]) == 3 || error("GPU RK4 extrema currently supports three-state Duffing trajectories only.")
    length(pvec) == 6 || error("GPU RK4 extrema currently expects six Duffing parameters.")
    max_extrema >= 1 || error("classification.max_extrema must be at least 1 for GPU RK4 extrema.")
    abs(Float64(dt) - Float64(save_dt)) <= 10 * Base.eps(Float64(dt)) || error("GPU RK4 extrema currently requires dt == save_dt.")

    n_total = length(u_trans_list)
    u_host = Matrix{T}(undef, 3, n_total)
    @inbounds for i in 1:n_total
        u = u_trans_list[i]
        u_host[1, i] = u[1]
        u_host[2, i] = u[2]
        u_host[3, i] = u[3]
    end

    u_dev = CuArray(u_host)
    p_dev = CuArray(collect(T.(pvec)))
    extrema_dev = CUDA.fill(Float32(NaN), max_extrema, n_total)
    counts_dev = CUDA.zeros(Int32, n_total)
    total_steps = stage_b_total_steps(t_evaluation, dt)
    phase_start = rk4_phase_from_time(pvec, t_transient, T)
    dphase = T(pvec[1]) * T(dt)
    threads = 128
    blocks = cld(n_total, threads)

    @cuda threads=threads blocks=blocks rk4_duffing_extrema_phasewrapped_kernel!(
        u_dev,
        p_dev,
        extrema_dev,
        counts_dev,
        Int32(n_total),
        phase_start,
        T(dt),
        dphase,
        Int32(total_steps),
        Int32(observable_idx),
        Int32(zero_cross_idx),
        Int32(max_extrema),
        T(eps),
    )
    CUDA.synchronize()

    return rk4_extrema_matrix_to_vectors(Array(extrema_dev), Array(counts_dev))
end


function collect_full_extrema_rk4_gpu_default_duffing(
    u0_list,
    pvec,
    t_transient,
    t_evaluation,
    dt,
    save_dt,
    observable_idx::Int,
    zero_cross_idx::Int,
    max_extrema::Int,
    eps,
)
    CUDA.functional() || error("CUDA is not functional, so the full RK4 extrema GPU run is not possible.")
    isempty(u0_list) && return Vector{Vector{Float32}}(), typeof(u0_list)()
    T = eltype(u0_list[1])
    T == Float32 || error("GPU full RK4 extrema currently supports Float32 only.")
    length(u0_list[1]) == 3 || error("GPU full RK4 extrema currently supports three-state Duffing trajectories only.")
    length(pvec) == 6 || error("GPU full RK4 extrema currently expects six Duffing parameters.")
    max_extrema >= 1 || error("classification.max_extrema must be at least 1 for GPU full RK4 extrema.")
    abs(Float64(dt) - Float64(save_dt)) <= 10 * Base.eps(Float64(dt)) || error("GPU full RK4 extrema currently requires dt == save_dt.")

    n_total = length(u0_list)
    u_host = Matrix{T}(undef, 3, n_total)
    @inbounds for i in 1:n_total
        u = u0_list[i]
        u_host[1, i] = u[1]
        u_host[2, i] = u[2]
        u_host[3, i] = u[3]
    end

    p_dev = CuArray(collect(T.(pvec)))
    states_dev = CuArray(u_host)
    extrema_dev = CUDA.fill(Float32(NaN), max_extrema, n_total)
    counts_dev = CUDA.zeros(Int32, n_total)
    transient_steps = rk4_step_count(t_transient, dt)
    evaluation_steps = stage_b_total_steps(t_evaluation, dt)
    threads = 128
    blocks = cld(n_total, threads)

    phase = zero(T)
    dphase = T(pvec[1]) * T(dt)
    remaining = transient_steps
    max_chunk_steps = rk4_kernel_max_steps()
    while remaining > 0
        steps_this_chunk = min(max_chunk_steps, remaining)
        @cuda threads=threads blocks=blocks rk4_duffing_advance_phasewrapped_kernel!(
            states_dev,
            p_dev,
            Int32(n_total),
            phase,
            T(dt),
            dphase,
            Int32(steps_this_chunk),
        )
        CUDA.synchronize()
        phase = rk4_advance_phase_host(phase, dphase, steps_this_chunk, T)
        remaining -= steps_this_chunk
    end

    phase_transient = phase
    @cuda threads=threads blocks=blocks rk4_duffing_full_extrema_from_transient_phasewrapped_kernel!(
        states_dev,
        p_dev,
        extrema_dev,
        counts_dev,
        Int32(n_total),
        phase_transient,
        T(dt),
        dphase,
        Int32(evaluation_steps),
        Int32(observable_idx),
        Int32(zero_cross_idx),
        Int32(max_extrema),
        T(eps),
    )
    CUDA.synchronize()

    transient_states = rk4_matrix_to_svector_list(Array(states_dev))
    extrema_all = rk4_extrema_matrix_to_vectors(Array(extrema_dev), Array(counts_dev))
    return extrema_all, transient_states
end


function collect_full_extrema_rk4_phase_gpu_default_duffing(
    u0_list,
    pvec,
    transient_periods,
    evaluation_periods,
    samples_per_period::Int,
    observable_idx::Int,
    zero_cross_idx::Int,
    max_extrema::Int,
    eps,
)
    CUDA.functional() || error("CUDA is not functional, so the phase RK4 extrema GPU run is not possible.")
    isempty(u0_list) && return Vector{Vector{Float32}}(), typeof(u0_list)()
    T = eltype(u0_list[1])
    T == Float32 || error("GPU phase RK4 extrema currently supports Float32 only.")
    length(u0_list[1]) == 3 || error("GPU phase RK4 extrema currently supports three-state Duffing trajectories only.")
    length(pvec) == 6 || error("GPU phase RK4 extrema currently expects six Duffing parameters.")
    max_extrema >= 1 || error("classification.max_extrema must be at least 1 for GPU phase RK4 extrema.")
    samples_per_period >= 1 || error("samples_per_period must be at least 1 for GPU phase RK4 extrema.")

    n_total = length(u0_list)
    u_host = Matrix{T}(undef, 3, n_total)
    @inbounds for i in 1:n_total
        u = u0_list[i]
        u_host[1, i] = u[1]
        u_host[2, i] = u[2]
        u_host[3, i] = u[3]
    end

    p_dev = CuArray(collect(T.(pvec)))
    states_dev = CuArray(u_host)
    extrema_dev = CUDA.fill(Float32(NaN), max_extrema, n_total)
    counts_dev = CUDA.zeros(Int32, n_total)
    dtau = T(2 * pi) / T(samples_per_period)
    transient_steps = max(0, round(Int, Float64(transient_periods) * samples_per_period))
    evaluation_steps = max(1, round(Int, Float64(evaluation_periods) * samples_per_period))
    threads = 128
    blocks = cld(n_total, threads)

    tau = zero(T)
    remaining = transient_steps
    max_chunk_steps = rk4_kernel_max_steps()
    while remaining > 0
        steps_this_chunk = min(max_chunk_steps, remaining)
        @cuda threads=threads blocks=blocks rk4_duffing_phase_advance_kernel!(
            states_dev,
            p_dev,
            Int32(n_total),
            tau,
            dtau,
            Int32(steps_this_chunk),
        )
        CUDA.synchronize()
        tau += dtau * T(steps_this_chunk)
        remaining -= steps_this_chunk
    end

    @cuda threads=threads blocks=blocks rk4_duffing_phase_extrema_from_transient_kernel!(
        states_dev,
        p_dev,
        extrema_dev,
        counts_dev,
        Int32(n_total),
        T(2 * pi) * T(transient_periods),
        dtau,
        Int32(evaluation_steps),
        Int32(observable_idx),
        Int32(zero_cross_idx),
        Int32(max_extrema),
        T(eps),
    )
    CUDA.synchronize()

    transient_states = rk4_matrix_to_svector_list(Array(states_dev))
    extrema_all = rk4_extrema_matrix_to_vectors(Array(extrema_dev), Array(counts_dev))
    return extrema_all, transient_states
end
