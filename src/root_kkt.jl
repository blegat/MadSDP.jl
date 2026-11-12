# Augment with scaled cone directions rather than forming normal equations.
mutable struct RootKKTWorkspace{T,E}
    equality::E
    soc::Vector{SOCBlock{T}}
    entries::Vector{Tuple{Vector{Int},Matrix{T}}}
    matrix::SparseArrays.SparseMatrixCSC{T,Int32}
    positions::Dict{Tuple{Int,Int},Int}
    linear_solver::Any
    rhs::Vector{T}
    scale::Vector{T}
    saved_rhs::Vector{T}
    refinement::Vector{T}
    cone_offset::Int
    regularization::T
end
needs_schur_buffer(::RootKKT) = false
function check_schur(::RootKKT, ::AbstractAlgorithm)
    throw(ArgumentError("RootKKT requires PrimalDual with NT scaling."))
end
check_schur(::RootKKT, ::PrimalDual{<:NT}) = nothing
function schur_workspace(::RootKKT, ::Type{T}, n, model) where {T}
    eq = schur_workspace(AugmentedKKT(), T, n, model)
    return RootKKTWorkspace(
        eq,
        SOCBlock{T}[SOCBlock(eq.model, idx) for idx in LRO.second_order_indices(eq.model)],
        Tuple{Vector{Int},Matrix{T}}[],
        SparseArrays.spzeros(T, Int32, 0, 0),
        Dict{Tuple{Int,Int},Int}(),
        nothing,
        T[],
        T[],
        T[],
        T[],
        0,
        zero(T),
    )
end

function _initialize_root_kkt!(ws::RootKKTWorkspace{T}, solver) where {T}
    n = solver.m
    offset = ws.cone_offset = n + length(ws.equality.eta)
    pairs = Tuple{Int,Int}[]
    model = ws.equality.model
    for idx in LRO.matrix_indices(solver.model)
        d = LRO.side_dimension(solver.model, idx)
        first = model.dim.offsets[idx.value]+1
        J = model.jacobian[first:(first+d*d-1), :]
        cols = unique(SparseArrays.findnz(J)[2])
        push!(ws.entries, (cols, -Matrix(J[:, cols])))
        k = d*(d+1)÷2
        for j in cols, row = (offset+1):(offset+k)
            push!(pairs, (row, j))
        end
        offset += k
    end
    for b in ws.soc
        nrows = length(b.X)
        for j in b.columns, row = (offset+1):(offset+nrows)
            push!(pairs, (row, j))
        end
        offset += nrows
    end
    C = _inner(solver.model).C_lin
    for k in axes(C, 2), p in SparseArrays.nzrange(C, k)
        push!(pairs, (offset+k, C.rowval[p]))
    end
    offset += solver.n_scalar
    I, J, V = SparseArrays.findnz(ws.equality.E)
    for k in eachindex(V)
        push!(pairs, (n+I[k], J[k]))
    end
    for i = 1:offset
        push!(pairs, (i, i))
    end
    ws.matrix = SparseArrays.sparse(
        Int32[first(p) for p in pairs],
        Int32[last(p) for p in pairs],
        ones(T, length(pairs)),
        offset,
        offset,
    )
    for j in axes(ws.matrix, 2), p in SparseArrays.nzrange(ws.matrix, j)
        ws.positions[(ws.matrix.rowval[p], j)] = p
    end
    ws.rhs, ws.scale, ws.saved_rhs, ws.refinement =
        zeros(T, offset), ones(T, offset), zeros(T, offset), zeros(T, offset)
    ws.linear_solver = MadNLP.UmfpackSolver(ws.matrix)
    return
end

function factorize_schur!(ws::RootKKTWorkspace{T}, solver::MadSDPSolver) where {T}
    if isnothing(ws.linear_solver)
        _initialize_root_kkt!(ws, solver)
    end
    K = ws.matrix
    fill!(K.nzval, zero(T))
    offset = ws.cone_offset
    fill!(ws.scale, one(T))
    ws.scale[1:solver.m] .= zero(T)
    for (b, (cols, A)) in zip(solver.blocks, ws.entries)
        G = b.work.dir.G
        for (k, j) in enumerate(cols)
            LinearAlgebra.mul!(b.buf, reshape(view(A, :, k), b.n, b.n), G)
            LinearAlgebra.mul!(b.buf2, G', b.buf)
            row = offset
            for q = 1:b.n, p = 1:q
                row += 1
                value = b.buf2[p, q] * (p == q ? one(T) : sqrt(T(2)))
                K.nzval[ws.positions[(row, j)]] = value
                ws.scale[j] += abs2(value)
            end
        end
        offset += b.n*(b.n+1)÷2
    end
    for b in ws.soc
        for (k, j) in enumerate(b.columns), i in eachindex(b.X)
            value = b.weighted_A[i, k]
            K.nzval[ws.positions[(offset+i, j)]] = value
            ws.scale[j] += abs2(value)
        end
        offset += length(b.X)
    end
    C = _inner(solver.model).C_lin
    for k in axes(C, 2)
        weight = sqrt(solver.x[k]*solver.zinv[k])
        for p in SparseArrays.nzrange(C, k)
            j = C.rowval[p]
            value = weight*C.nzval[p]
            K.nzval[ws.positions[(offset+k, j)]] = value
            ws.scale[j] += abs2(value)
        end
    end
    for i = 1:solver.m
        ws.scale[i] = iszero(ws.scale[i]) ? one(T) : inv(sqrt(ws.scale[i]))
    end
    ws.scale[(solver.m+1):ws.cone_offset] .= zero(T)
    I, J, V = SparseArrays.findnz(ws.equality.E)
    for k in eachindex(V)
        K.nzval[ws.positions[(solver.m+I[k], J[k])]] = V[k]
        ws.scale[solver.m+I[k]] += abs2(V[k]*ws.scale[J[k]])
    end
    for i = (solver.m+1):ws.cone_offset
        ws.scale[i] = iszero(ws.scale[i]) ? one(T) : inv(sqrt(ws.scale[i]))
    end
    for i = (ws.cone_offset+1):length(ws.rhs)
        K.nzval[ws.positions[(i, i)]] = -one(T)
    end
    for j in axes(K, 2), p in SparseArrays.nzrange(K, j)
        K.nzval[p] *= ws.scale[K.rowval[p]]*ws.scale[j]
    end
    ws.regularization = zero(T)
    for attempt = 1:solver.opt.max_schur_reg
        MadNLP.factorize!(ws.linear_solver)
        if SparseArrays.UMFPACK.issuccess(ws.linear_solver.inner)
            return true
        end
        old = ws.regularization
        ws.regularization = iszero(old) ? max(eps(T), solver.opt.regularize_schur) : 10old
        for i = 1:ws.cone_offset
            K.nzval[ws.positions[(i, i)]] +=
                (i <= solver.m ? 1 : -1)*(ws.regularization-old)
        end
    end
    return false
end

function solve_directions!(
    ws::RootKKTWorkspace{T},
    solver::MadSDPSolver,
    mu::T,
    corrected::Bool,
) where {T}
    ws.rhs[1:solver.m] .= solver.Rp
    ws.rhs[(solver.m+1):ws.cone_offset] .= ws.equality.residual
    offset = ws.cone_offset
    for (i, b) in enumerate(solver.blocks)
        dir = b.work.dir
        # Negative scaled complementarity direction before the B*Δy term.
        fill!(b.buf2, zero(T))
        for k = 1:b.n
            b.buf2[k, k] = dir.D[k] - mu/dir.D[k]
        end
        if solver.rd_nonzero[i]
            LinearAlgebra.mul!(b.buf, b.Rd, dir.G)
            LinearAlgebra.mul!(b.buf2, dir.G', b.buf, one(T), one(T))
        end
        if corrected
            b.buf2 .-= dir.RNT
        end
        for q = 1:b.n, p = 1:q
            offset += 1
            ws.rhs[offset] = b.buf2[p, q]*(p == q ? one(T) : sqrt(T(2)))
        end
    end
    for b in ws.soc
        _soc_inverse!(b.work1, b.lambda)
        @. b.work1 = mu*b.work1-b.lambda
        LinearAlgebra.mul!(b.work2, b.G, b.Rd)
        b.work1 .-= b.work2
        if corrected
            b.work1 .+= b.correction
        end
        for i in eachindex(b.X)
            ws.rhs[offset+i] = -b.work1[i]
        end
        offset += length(b.X)
    end
    for k = 1:solver.n_scalar
        weight = sqrt(solver.x[k]*solver.zinv[k])
        q = mu*solver.zinv[k] - solver.x[k] - solver.x[k]*solver.zinv[k]*solver.Rd_lin[k]
        if corrected
            q += solver.corr_lin[k]
        end
        ws.rhs[offset+k] = -q/weight
    end
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
        for i = 1:ws.cone_offset
            ws.refinement[i] += (i <= solver.m ? 1 : -1)*ws.regularization*ws.rhs[i]
        end
        if maximum(abs(ws.refinement[i]/ws.scale[i]) for i in eachindex(ws.rhs)) <=
           1e-12*max(
            1,
            maximum(abs(ws.saved_rhs[i]/ws.scale[i]) for i in eachindex(ws.rhs)),
        )
            break
        end
        MadNLP.solve_linear_system!(ws.linear_solver, ws.refinement)
        ws.rhs .+= ws.refinement
    end
    ws.rhs .*= ws.scale
    solver.dely .= view(ws.rhs, 1:solver.m)
    ws.equality.delta_eta .= view(ws.rhs, (solver.m+1):ws.cone_offset)
    # Recover the cone direction directly; avoid subtracting large matrices.
    offset = ws.cone_offset
    for (i, b) in enumerate(solver.blocks)
        AT = LRO.unsafe_jtprod(solver.model, solver.dely, LRO.MatrixIndex(i))
        copyto!(b.work.delS, b.Rd)
        _axpy_sparse_dense!(b.work.delS, -one(T), AT)
        _symmetrize!(b.work.delS)
        for q = 1:b.n, p = 1:q
            offset += 1
            value = ws.rhs[offset]/(p == q ? one(T) : sqrt(T(2)))
            b.buf[p, q] = b.buf[q, p] = value
        end
        LinearAlgebra.mul!(b.buf2, b.work.dir.G, b.buf)
        LinearAlgebra.mul!(b.work.delX, b.buf2, b.work.dir.G')
        _symmetrize!(b.work.delX)
    end
    for b in ws.soc
        LinearAlgebra.mul!(b.delX, b.G, view(ws.rhs, (offset+1):(offset+length(b.X))))
        copyto!(b.delS, b.Rd)
        LinearAlgebra.mul!(b.delS, b.A, view(solver.dely, b.columns), -1, true)
        offset += length(b.X)
    end
    LinearAlgebra.mul!(solver.delz, _inner(solver.model).C_lin', solver.dely)
    solver.delz .= solver.Rd_lin .- solver.delz
    for k = 1:solver.n_scalar
        solver.delx[k] = sqrt(solver.x[k]*solver.zinv[k])*ws.rhs[offset+k]
    end
    return
end

function additional_residuals!(ws::RootKKTWorkspace, solver)
    additional_residuals!(ws.equality, solver)
    for b in ws.soc
        for (k, j) in enumerate(b.columns)
            solver.Rp[j] -= LinearAlgebra.dot(view(b.A, :, k), b.X)
        end
        b.Rd .= b.constant .- b.S
        LinearAlgebra.mul!(b.Rd, b.A, view(solver.y, b.columns), -1, true)
    end
    return
end
function additional_step!(ws::RootKKTWorkspace, alpha, beta)
    additional_step!(ws.equality, alpha, beta)
    for b in ws.soc
        @. b.X += alpha*b.delX
        @. b.S += beta*b.delS
    end
    return
end
additional_objective(ws::RootKKTWorkspace{T}) where {T} =
    additional_objective(ws.equality)+sum(
        b -> LinearAlgebra.dot(b.constant, b.X),
        ws.soc;
        init = zero(T),
    )
function additional_infeasibility(ws::RootKKTWorkspace)
    residual = additional_infeasibility(ws.equality)
    for b in ws.soc
        residual = max(
            residual,
            LinearAlgebra.norm(b.Rd, Inf)/max(1, LinearAlgebra.norm(b.constant, Inf)),
        )
    end
    return residual
end
function initialize_constraints!(ws::RootKKTWorkspace{T}, solver) where {T}
    initialize_constraints!(ws.equality, solver)
    scale =
        if solver.initial_point isa ScaledIdentity && !isnothing(solver.initial_point.scale)
            T(solver.initial_point.scale)
        else
            one(T)
        end
    for b in ws.soc
        fill!(b.X, zero(T))
        fill!(b.S, zero(T))
        b.X[1] = b.S[1] = scale
    end
    return
end
additional_barrier_degree(ws::RootKKTWorkspace) = length(ws.soc)
additional_complementarity(ws::RootKKTWorkspace{T}) where {T} =
    sum(b -> LinearAlgebra.dot(b.X, b.S), ws.soc; init = zero(T))
function additional_affine_complementarity(ws::RootKKTWorkspace, alpha, beta)
    result = zero(eltype(ws.rhs))
    for b in ws.soc
        result +=
            LinearAlgebra.dot(b.X, b.S)+beta*LinearAlgebra.dot(b.X, b.delS)+alpha*LinearAlgebra.dot(
                b.delX,
                b.S,
            )+alpha*beta*LinearAlgebra.dot(b.delX, b.delS)
    end
    return result
end
prepare_additional_cones!(ws::RootKKTWorkspace, solver) = all(prepare_soc!, ws.soc)
function additional_correction!(ws::RootKKTWorkspace)
    for b in ws.soc
        soc_correction!(b)
    end
    return
end
function additional_step_lengths(ws::RootKKTWorkspace, alpha, beta, tau)
    for b in ws.soc
        alpha = min(alpha, tau*_soc_max_step(b.X, b.delX))
        beta = min(beta, tau*_soc_max_step(b.S, b.delS))
    end
    return alpha, beta
end
equality_block(ws::RootKKTWorkspace) = ws.equality
equality_block(ws::AugmentedKKTWorkspace) = ws
