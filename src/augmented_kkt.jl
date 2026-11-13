# These hooks keep equality multipliers separate from cone complementarity.
additional_objective(_) = 0
additional_infeasibility(_) = 0
additional_residuals!(_, solver) = nothing
additional_step!(_, alpha, beta) = nothing

mutable struct AugmentedKKTWorkspace{T,M}
    model::M
    equalities::Vector{Int}
    E::SparseArrays.SparseMatrixCSC{T,Int}
    target::Vector{T}
    eta::Vector{T}
    delta_eta::Vector{T}
    residual::Vector{T}
    entries::Vector{Tuple{Vector{Int},Matrix{T},Matrix{T},Matrix{T}}}
    matrix::SparseArrays.SparseMatrixCSC{T,Int32}
    linear_solver::Any
    positions::Dict{Tuple{Int,Int},Int}
    rhs::Vector{T}
    scale::Vector{T}
    saved_rhs::Vector{T}
    refinement::Vector{T}
    original_E::SparseArrays.SparseMatrixCSC{T,Int}
    original_target::Vector{T}
    original_residual::Vector{T}
end

needs_schur_buffer(::AugmentedKKT) = false
function check_schur(::AugmentedKKT, ::AbstractAlgorithm)
    throw(ArgumentError("AugmentedKKT requires PrimalDual."))
end
check_schur(::AugmentedKKT, ::PrimalDual) = nothing

function schur_workspace(::AugmentedKKT, ::Type{T}, n, src) where {T}
    model = LRO.primal_model(src)
    eq = filter(i -> model.meta.lcon[i] == model.meta.ucon[i], 1:LRO.num_scalars(model))
    E = model.jacobian[eq, :]
    target = model.meta.lcon[eq] - model.constant[eq]
    residual=copy(target)
    N = n + length(eq)
    return AugmentedKKTWorkspace(
        model,
        eq,
        E,
        target,
        zeros(T, length(eq)),
        zeros(T, length(eq)),
        residual,
        Tuple{Vector{Int},Matrix{T},Matrix{T},Matrix{T}}[],
        SparseArrays.spzeros(T, Int32, N, N),
        nothing,
        Dict{Tuple{Int,Int},Int}(),
        zeros(T, N),
        ones(T, N),
        zeros(T, N),
        zeros(T, N),
        E,
        target,
        residual,
    )
end

function _initialize_kkt!(ws::AugmentedKKTWorkspace{T}, solver) where {T}
    n = solver.m
    pairs = Set((i, i) for i = 1:length(ws.rhs))
    for idx in LRO.matrix_indices(solver.model)
        offset = ws.model.dim.offsets[idx.value]
        d = LRO.side_dimension(solver.model, idx)
        J = ws.model.jacobian[(offset+1):(offset+d*d), :]
        cols = unique(SparseArrays.findnz(J)[2])
        coefficients = -Matrix(J[:, cols])
        push!(
            ws.entries,
            (
                cols,
                coefficients,
                similar(coefficients),
                zeros(T, length(cols), length(cols)),
            ),
        )
        for j in cols, i in cols
            push!(pairs, (max(i, j), min(i, j)))
        end
    end
    C = _inner(solver.model).C_lin
    for k in axes(C, 2)
        ids = SparseArrays.rowvals(C)[SparseArrays.nzrange(C, k)]
        for j in ids, i in ids
            push!(pairs, (max(i, j), min(i, j)))
        end
    end
    I, J, V = SparseArrays.findnz(ws.E)
    for k in eachindex(V)
        push!(pairs, (n+I[k], J[k]))
    end
    pairs = collect(pairs)
    ws.matrix = SparseArrays.sparse(
        Int32[first(p) for p in pairs],
        Int32[last(p) for p in pairs],
        ones(T, length(pairs)),
        length(ws.rhs),
        length(ws.rhs),
    )
    for j in axes(ws.matrix, 2), p in SparseArrays.nzrange(ws.matrix, j)
        ws.positions[(SparseArrays.rowvals(ws.matrix)[p], j)] = p
    end
    ws.linear_solver = MadNLP.UmfpackSolver(ws.matrix)
    return
end

function factorize_schur!(ws::AugmentedKKTWorkspace{T}, solver::MadSDPSolver) where {T}
    if isnothing(ws.linear_solver)
        _initialize_kkt!(ws, solver)
    end
    K = ws.matrix
    fill!(SparseArrays.nonzeros(K), zero(T))
    for (b, (cols, A, EA, H)) in zip(solver.blocks, ws.entries)
        for k in eachindex(cols)
            apply_E!(
                reshape(view(EA, :, k), b.n, b.n),
                b.work.dir,
                b,
                reshape(view(A, :, k), b.n, b.n),
            )
        end
        LinearAlgebra.mul!(H, A', EA)
        for j in eachindex(cols), i = j:length(cols)
            K.nzval[ws.positions[(cols[i], cols[j])]] += H[i, j]
        end
    end
    C = _inner(solver.model).C_lin
    for k in axes(C, 2)
        weight = solver.x[k] * solver.zinv[k]
        for q in SparseArrays.nzrange(C, k), p = q:last(SparseArrays.nzrange(C, k))
            i, j = C.rowval[p], C.rowval[q]
            K.nzval[ws.positions[(i, j)]] += weight*C.nzval[p]*C.nzval[q]
        end
    end
    I, J, V = SparseArrays.findnz(ws.E)
    for k in eachindex(V)
        K.nzval[ws.positions[(solver.m+I[k], J[k])]] = V[k]
    end
    # Equilibrate scalar variables and equality multipliers symmetrically.
    for i = 1:solver.m
        diagonal = K.nzval[ws.positions[(i, i)]]
        ws.scale[i] = iszero(diagonal) ? one(T) : inv(sqrt(abs(diagonal)))
    end
    ws.scale[(solver.m+1):end] .= zero(T)
    for k in eachindex(V)
        ws.scale[solver.m+I[k]] += abs2(V[k]*ws.scale[J[k]])
    end
    for i = (solver.m+1):length(ws.rhs)
        ws.scale[i] = iszero(ws.scale[i]) ? one(T) : inv(sqrt(ws.scale[i]))
    end
    for j in axes(K, 2), p in SparseArrays.nzrange(K, j)
        K.nzval[p] *= ws.scale[K.rowval[p]] * ws.scale[j]
    end
    for i = 1:length(ws.rhs)
        K.nzval[ws.positions[(i, i)]] +=
            (i <= solver.m ? 1 : -1) * solver.opt.regularize_schur
    end
    MadNLP.factorize!(ws.linear_solver)
    return SparseArrays.UMFPACK.issuccess(ws.linear_solver.inner)
end

function solve_directions!(
    ws::AugmentedKKTWorkspace,
    solver::MadSDPSolver{T},
    mu::T,
    corrected::Bool,
) where {T}
    rhs!(solver.dely, solver, mu, corrected)
    ws.rhs[1:solver.m] .= solver.dely
    ws.rhs[(solver.m+1):end] .= ws.residual
    ws.rhs .*= ws.scale
    copyto!(ws.saved_rhs, ws.rhs)
    MadNLP.solve_linear_system!(ws.linear_solver, ws.rhs)
    for iteration = 1:5
        copyto!(ws.refinement, ws.saved_rhs)
        LinearAlgebra.mul!(
            ws.refinement,
            LinearAlgebra.Symmetric(ws.matrix, :L),
            ws.rhs,
            -1,
            true,
        )
        for i in eachindex(ws.rhs)
            ws.refinement[i] +=
                (i <= solver.m ? 1 : -1) * solver.opt.regularize_schur * ws.rhs[i]
        end
        if LinearAlgebra.norm(ws.refinement, Inf) <=
           1e-12*max(1, LinearAlgebra.norm(ws.saved_rhs, Inf))
            break
        end
        MadNLP.solve_linear_system!(ws.linear_solver, ws.refinement)
        ws.rhs .+= ws.refinement
    end
    ws.rhs .*= ws.scale
    solver.dely .= view(ws.rhs, 1:solver.m)
    ws.delta_eta .= view(ws.rhs, (solver.m+1):length(ws.rhs))
    directions!(solver, solver.dely, mu, corrected)
    return
end

function additional_residuals!(ws::AugmentedKKTWorkspace, solver)
    LinearAlgebra.mul!(solver.Rp, ws.E', ws.eta, -1, true)
    ws.residual .= cone_data_scale(solver.state) .* ws.target
    LinearAlgebra.mul!(ws.residual, ws.E, solver.y, -1, true)
    if ws.original_residual !== ws.residual
        ws.original_residual .= cone_data_scale(solver.state) .* ws.original_target
        LinearAlgebra.mul!(ws.original_residual, ws.original_E, solver.y, -1, true)
    end
    return
end
function additional_step!(ws::AugmentedKKTWorkspace, alpha, beta)
    ws.eta .+= alpha .* ws.delta_eta
    return
end
additional_objective(ws::AugmentedKKTWorkspace) = LinearAlgebra.dot(ws.target, ws.eta)
additional_infeasibility(ws::AugmentedKKTWorkspace) =
    LinearAlgebra.norm(ws.original_residual, Inf)

initialize_constraints!(_, solver) = nothing
function initialize_constraints!(ws::AugmentedKKTWorkspace{T}, solver) where {T}
    rows=LRO.equality_basis(ws.model; tolerance = solver.opt.tol/100)
    if rows != ws.equalities
        ws.equalities=rows
        ws.E=ws.model.jacobian[rows, :]
        ws.target=ws.model.meta.lcon[rows]-ws.model.constant[rows]
        ws.residual=copy(ws.target)
        ws.eta=zeros(T, length(rows))
        ws.delta_eta=zeros(T, length(rows))
        N=solver.m+length(rows)
        ws.rhs=zeros(T, N)
        ws.scale=ones(T, N)
        ws.saved_rhs=zeros(T, N)
        ws.refinement=zeros(T, N)
        ws.matrix=SparseArrays.spzeros(T, Int32, N, N)
        ws.linear_solver=nothing
        empty!(ws.entries)
        empty!(ws.positions)
    end
    # All scalar cone slacks are independent at the infeasible starting point.
    # Initialize their positive bounds with the same routine MadNLP uses.
    MadNLP.initialize_variables!(
        solver.z,
        zeros(T, solver.n_scalar),
        fill(T(Inf), solver.n_scalar),
        T(0.01),
        T(0.01),
    )
    fill!(ws.eta, zero(T))
    return
end

additional_barrier_degree(_) = 0
additional_complementarity(_) = 0
additional_affine_complementarity(_, alpha, beta) = 0
prepare_additional_cones!(_, solver) = true
additional_correction!(_) = nothing
additional_step_lengths(_, alpha, beta, tau) = (alpha, beta)
