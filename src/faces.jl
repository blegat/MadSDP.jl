"""
    FaceOptions(; history=3, discard=1e-12, trigger_mu=1e-4, ...)

Experimental single-block Float64 face restarts. Average near-kernel projectors,
retaining every eigenvalue above `discard` (including rotating directions).
Finite history is a heuristic, not a certificate. Rejected restarts roll back to
an untouched full-space iterate; later attempts use fresh full-space history.
"""
Base.@kwdef struct FaceOptions
    history::Int = 3
    discard::Float64 = 1e-12
    trigger_mu::Float64 = 1e-4
    feasibility_gate::Float64 = 1e-3
    max_restarts::Int = 3
    rank_tol::Float64 = 1e-12
    projector_svd::Bool = true
end

function face_data(model)
    if LRO.num_matrices(model) != 1 || LRO.num_scalars(model) != 0
        throw(ArgumentError("Face restarts require one PSD block and no scalars."))
    end
    idx = LRO.MatrixIndex(1)
    return Matrix(LRO.grad(model, idx)),
    [Matrix(LRO.jac(model, j, idx)) for j = 1:model.meta.ncon],
    copy(LRO.cons_constant(model))
end

function face_model(C, A, b)
    n, m = size(C, 1), length(b)
    return LRO.Model(
        [SparseArrays.sparse(C)],
        reshape(SparseArrays.sparse.(A), 1, m),
        b,
        SparseArrays.spzeros(eltype(C), 0),
        SparseArrays.spzeros(eltype(C), m, 0),
        [n],
    )
end

face_lift(V, Y) = Matrix(Symmetric(V * Y * V'))
face_project(V, X) = Matrix(Symmetric(V' * X * V))

"""Compress equations by SVD, preserving the original right hand side.
Returns `nothing` if the discarded equations are inconsistent. `U` maps reduced
multipliers back to original multipliers. This rank decision is approximate.
"""
function restrict_face(C, A, b, V; rank_tol = 1e-12)
    if size(V, 2) == 0 || norm(V' * V - I) > 1e-10
        throw(ArgumentError("Expected a nonempty orthonormal face basis."))
    end
    Ar = [face_project(V, a) for a in A]
    M = reduce(hcat, vec.(Ar))'
    F = svd(M)
    r = count(>(rank_tol * maximum(F.S; init = 0.0)), F.S)
    U = F.U[:, 1:r]
    inconsistency = norm(b - U * (U' * b)) / max(1, norm(b))
    if r == 0 || inconsistency > rank_tol
        return nothing
    end
    Ac = [sum(U[i, j] * Ar[i] for i in eachindex(A)) for j = 1:r]
    return (;
        model = face_model(face_project(V, C), Ac, U' * b),
        U,
        inconsistency,
        condition = F.S[1] / F.S[r],
    )
end

# Jacobi diagonalization also works for BigFloat, without a new dependency.
# Used only by the independent, small dense problem verifier.
function face_eigmin(A)
    B = Matrix(Symmetric(A))
    n = size(B, 1)
    for _ = 1:100
        off = zero(eltype(B))
        for q = 2:n, p = 1:(q-1)
            a = B[p, q]
            off = max(off, abs(a))
            if abs(a) <= eps(eltype(B)) * max(one(a), norm(B))
                continue
            end
            τ = (B[q, q] - B[p, p]) / (2a)
            t = copysign(one(τ), τ) / (abs(τ) + sqrt(1 + τ^2))
            c = inv(sqrt(1 + t^2))
            s = t * c
            for k = 1:n
                if k != p && k != q
                    u, v = B[k, p], B[k, q]
                    B[k, p] = B[p, k] = c * u - s * v
                    B[k, q] = B[q, k] = s * u + c * v
                end
            end
            B[p, p] -= t * a
            B[q, q] += t * a
            B[p, q] = B[q, p] = 0
        end
        if off <= eps(eltype(B)) * max(one(off), norm(B))
            return minimum(diag(B))
        end
    end
    error("Jacobi verification did not converge")
end

"""
    face_residuals(C, A, b, X, y, S)

Independent original-data residuals. Norms are Euclidean/Frobenius. PSD errors
use negative minimum eigenvalues. Works on BigFloat-converted stored data.
"""
function face_residuals(C, A, b, X, y, S)
    p, d = dot(C, X), dot(b, y)
    return (
        primal = norm([dot(a, X) for a in A] - b) / max(1, norm(b)),
        dual = norm(C - sum(y[i] * A[i] for i in eachindex(A)) - S) / max(1, norm(C)),
        primal_psd = max(0, -face_eigmin(X)) / max(1, norm(X)),
        dual_psd = max(0, -face_eigmin(S)) / max(1, norm(S)),
        gap = abs(p - d) / max(1, abs(p) + abs(d)),
        complementarity = norm(X * S) / max(1, norm(X) * norm(S)),
    )
end

# At a machine-precision target, cancellation in Float64 is too optimistic.
function verified_face_residuals(C, A, b, X, y, S)
    return setprecision(256) do
        r = face_residuals(
            BigFloat.(C),
            [BigFloat.(a) for a in A],
            BigFloat.(b),
            BigFloat.(X),
            BigFloat.(y),
            BigFloat.(S),
        )
        return map(Float64, r)
    end
end

"""Exact original-data dual bound for small Float64 problems.

Check PSD by rational LDL elimination on the exact binary input data and y.
Returns `nothing` when PSD fails; never infer a bound from a reduced slack.
This intentionally expensive diagnostic is optional in `adaptive_face`.
"""
function face_bound(C, A, b, y)
    Q = Rational{BigInt}
    S = Q.(C) - sum(Q(y[i]) * Q.(A[i]) for i in eachindex(A))
    for j in axes(S, 1)
        if S[j, j] < 0
            return nothing
        elseif iszero(S[j, j])
            if any(!iszero, S[(j+1):end, j])
                return nothing
            end
        else
            for k = (j+1):size(S, 1), l = k:size(S, 1)
                S[l, k] -= S[l, j] * S[k, j] / S[j, j]
                S[k, l] = S[l, k]
            end
        end
    end
    return dot(Q.(b), Q.(y))
end

face_merit(r) = maximum(values(r))

# Recover multipliers from full-space complementarity, including cross blocks.
# This is a least-squares candidate, accepted only after full-space verification.
function face_dual(C, A, X; rank_tol = 1e-8)
    E = eigen(Symmetric(X))
    W = E.vectors[:, E.values .> rank_tol*max(1, maximum(E.values))]
    M = reduce(hcat, [vec(a * W) for a in A])
    y = pinv(M) * vec(C * W)
    S = Matrix(Symmetric(C - sum(y[i] * A[i] for i in eachindex(A))))
    return y, S
end

# Orthogonal least-squares correction in a fixed face. PSD is not preserved;
# the caller must verify both cones afterwards. No right hand side is changed.
function face_round(A, b, V, X)
    Ar = [face_project(V, a) for a in A]
    M = reduce(hcat, vec.(Ar))'
    inverse = pinv(M; rtol = 1e-12)
    Y = face_project(V, X)
    for _ = 1:3
        residual = b - [dot(a, Y) for a in Ar]
        Y += reshape(inverse * residual, size(Y))
        Y = Matrix(Symmetric(Y))
    end
    return face_lift(V, Y)
end

"""
    refine_rank_one(C, A, b, X, y; steps=5)

Local Newton rounding of the rank-one KKT equations A(vv')=b and
(C-A*(y))v=0. The vector may rotate; this is not a convex face restriction.
Use only near a rank-one solution and verify the full KKT residuals afterwards.
Also supports BigFloat for reference calculations.
"""
function refine_rank_one(C, A, b, X, y; steps = 5)
    # Float64 eigenvectors only seed Newton, including for BigFloat references.
    E = eigen(Symmetric(Float64.(X)))
    v = eltype(C).(E.vectors[:, end] * sqrt(max(0, E.values[end])))
    y = copy(y)
    for _ = 1:steps
        S = C - sum(y[i] * A[i] for i in eachindex(A))
        Av = reduce(hcat, [a * v for a in A])
        residual = vcat([dot(v, a * v) for a in A] - b, S * v)
        J = [2Av' zeros(eltype(C), length(b), length(b)); S -Av]
        delta = J \ residual
        v -= delta[1:length(v)]
        y -= delta[(length(v)+1):end]
    end
    S = Matrix(Symmetric(C - sum(y[i] * A[i] for i in eachindex(A))))
    return (; X = v * v', y, S)
end

function face_candidate(
    C,
    A,
    b,
    V,
    X,
    y;
    schur = QRSchur(),
    tol = 1e-15,
    max_iter = 100,
    rank_tol = 1e-12,
    refine = true,
    rotate = false,
)
    reduced = restrict_face(C, A, b, V; rank_tol)
    if isnothing(reduced)
        return (; accepted = false, reason = :inconsistent, iterations = 0, V)
    end
    # Positive definite relative-interior restart. Equality infeasibility is
    # handled by the ordinary infeasible-start IPM, never by changing b.
    Y = face_project(V, X)
    δ = max(1e-6, -eigmin(Symmetric(Y)) + 1e-6)
    Y += δ * I
    z = reduced.U' * y
    Cr, Ar, br = face_data(reduced.model)
    Sr = Cr - sum(z[i] * Ar[i] for i in eachindex(Ar))
    # ScaledIdentity is a safe fallback if the mapped dual is not interior.
    initial = eigmin(Symmetric(Sr)) > 0 ? InitialPoint([Y], z) : ScaledIdentity()
    solver = MadSDPSolver(
        reduced.model;
        step = CholeskyBacktrack(),
        schur,
        initial_point = initial,
        tol,
        mu_min = 0.0,
        regularize_schur = 0.0,
        max_iter,
    )
    solve!(solver; verbose = false)
    Xr = face_lift(V, only(solver.blocks).X)
    raw_X = copy(Xr)
    if refine && count(>(1e-8 * max(1, norm(Xr))), eigvals(Symmetric(Xr))) == 1
        rounded = refine_rank_one(Cr, Ar, br, only(solver.blocks).X, solver.y)
        Xr = face_lift(V, rounded.X)
    else
        Xr = face_round(A, b, V, Xr)
    end
    yr = y + reduced.U * (solver.y - reduced.U' * y)
    S = Matrix(Symmetric(C - sum(yr[i] * A[i] for i in eachindex(A))))
    raw_residuals = verified_face_residuals(C, A, b, raw_X, yr, S)
    residuals = verified_face_residuals(C, A, b, Xr, yr, S)
    yp, Sp = face_dual(C, A, Xr)
    polished = verified_face_residuals(C, A, b, Xr, yp, Sp)
    if face_merit(polished) < face_merit(residuals)
        yr, S, residuals = yp, Sp, polished
    end
    rotated = false
    if rotate && face_merit(residuals) > tol
        rounded = refine_rank_one(C, A, b, Xr, yr)
        rr = verified_face_residuals(C, A, b, rounded.X, rounded.y, rounded.S)
        if face_merit(rr) < face_merit(residuals)
            Xr, yr, S, residuals = rounded.X, rounded.y, rounded.S, rr
            rotated = true
        end
    end
    accepted = face_merit(residuals) <= tol
    return (;
        accepted,
        reason = accepted ? :verified : :full_space_failure,
        X = Xr,
        y = yr,
        S,
        residuals,
        raw_residuals,
        rotated,
        iterations = solver.iter,
        V,
        condition = reduced.condition,
        status = solver.status,
    )
end

function try_face_candidate(C, A, b, V, X, y; kwargs...)
    try
        return face_candidate(C, A, b, V, X, y; kwargs...)
    catch err
        if err isa LinearAlgebra.SingularException ||
           err isa LinearAlgebra.PosDefException ||
           err isa LinearAlgebra.LAPACKException
            return (; accepted = false, reason = :numerical_failure, iterations = 0, V)
        end
        rethrow()
    end
end

"""
    adaptive_face(model; face=FaceOptions(), schur=QRSchur(), tol=1e-14, ...)

Optional experimental restart driver, leaving `solve!` unchanged. Returns a
full-space solution and all attempted candidates (including failures). Only a
candidate passing every original-space residual at `tol` is accepted. Otherwise
resume the saved full solver, permitting rotation/expansion all the way back to
the original cone. The original model is never mutated. `refine=true` applies
five local rank-one Newton rounding steps within the face when its solution has
numerical rank one; `rotate=true` additionally permits full-space Newton
rounding. These are local nonlinear refinements, verified against both full
cones. `bounds=true` saves an exactly checked original-space rational dual
bound independently of the returned (approximately feasible) solution.
"""
function adaptive_face(
    model;
    face = FaceOptions(),
    schur = QRSchur(),
    tol = 1e-14,
    max_iter = 100,
    refine = true,
    rotate = false,
    bounds = false,
)
    if face.history < 1 || !(0 <= face.discard < 1)
        throw(ArgumentError("Invalid projector history or discard threshold."))
    end
    C, A, b = face_data(model)
    solver = MadSDPSolver(
        model;
        step = CholeskyBacktrack(),
        schur,
        tol,
        max_iter,
        mu_min = 0.0,
        regularize_schur = 0.0,
    )
    initialize!(solver)
    history = Matrix{Float64}[]
    events = NamedTuple[]
    trigger = face.trigger_mu
    best = nothing
    best_merit = Inf
    iterations = 0
    dual_bound = nothing
    bound_y = nothing
    while true
        update_termination!(solver)
        block = only(solver.blocks)
        r = face_residuals(C, A, b, block.X, solver.y, block.S)
        if bounds
            bound = face_bound(C, A, b, solver.y)
            if !isnothing(bound) && (isnothing(dual_bound) || bound > dual_bound)
                dual_bound, bound_y = bound, copy(solver.y)
            end
        end
        if face_merit(r) < best_merit
            best =
                (; X = copy(block.X), y = copy(solver.y), S = copy(block.S), residuals = r)
            best_merit = face_merit(r)
        end
        if solver.status != MadNLP.REGULAR
            break
        end
        μ = total_complementarity(solver) / _barrier_dim(solver)
        if max(solver.inf_pr, solver.inf_du) <= face.feasibility_gate
            E = eigen(Symmetric(block.S))
            scale = max(1, norm(block.S))
            threshold = min(0.2 * scale, 5sqrt(max(0, μ) * scale / max(1, norm(block.X))))
            W = E.vectors[:, E.values .<= threshold]
            if 0 < size(W, 2) < size(C, 1)
                push!(history, W * W')
                if length(history) > face.history
                    popfirst!(history)
                end
            end
            if length(history) == face.history &&
               μ <= trigger &&
               length(events) < face.max_restarts
                if face.projector_svd
                    # [P₁ ... Pₕ][P₁ ... Pₕ]' = h P̄. Avoid squaring the
                    # small singular values that encode rotation.
                    F = svd(reduce(hcat, history))
                    spectrum = reverse(F.S .^ 2 / length(history))
                    V = F.U[:, F.S .> sqrt(length(history)*face.discard)]
                else
                    H = eigen(Symmetric(sum(history) / length(history)))
                    spectrum = H.values
                    V = H.vectors[:, H.values .> face.discard]
                end
                if 0 < size(V, 2) < size(C, 1)
                    candidate = try_face_candidate(
                        C,
                        A,
                        b,
                        V,
                        block.X,
                        solver.y;
                        schur,
                        tol,
                        max_iter,
                        rank_tol = face.rank_tol,
                        refine,
                        rotate,
                    )
                    push!(events, (; iteration = solver.iter, mu = μ, spectrum, candidate))
                    iterations += candidate.iterations
                    if candidate.accepted
                        if bounds
                            bound = face_bound(C, A, b, candidate.y)
                            if !isnothing(bound) &&
                               (isnothing(dual_bound) || bound > dual_bound)
                                dual_bound, bound_y = bound, copy(candidate.y)
                            end
                        end
                        return (;
                            candidate.X,
                            candidate.y,
                            candidate.S,
                            candidate.residuals,
                            events,
                            iterations = iterations + solver.iter,
                            restricted = !candidate.rotated,
                            rotated = candidate.rotated,
                            verified = true,
                            dual_bound,
                            bound_y,
                        )
                    end
                    # Rollback: solver was never modified by the trial.
                    empty!(history)
                    trigger = μ / 100
                end
            end
        end
        iterate!(solver)
    end
    return (;
        best.X,
        best.y,
        best.S,
        best.residuals,
        events,
        iterations = iterations + solver.iter,
        restricted = false,
        rotated = false,
        verified = face_merit(verified_face_residuals(C, A, b, best.X, best.y, best.S)) <=
                   tol,
        dual_bound,
        bound_y,
    )
end
