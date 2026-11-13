# Augment with scaled cone directions rather than forming normal equations.
struct RootKKTIndices
    psd::Vector{Matrix{Int}}
    soc::Vector{Matrix{Int}}
    scalar::Vector{Int}
    equality::Vector{Int}
    eq_rows::Vector{Int}
    eq_cols::Vector{Int}
    diagonal::Vector{Int}
    psd_hessian::Vector{Matrix{Int}}
    soc_hessian::Vector{Matrix{Int}}
end
RootKKTIndices() = RootKKTIndices(
    Matrix{Int}[],
    Matrix{Int}[],
    Int[],
    Int[],
    Int[],
    Int[],
    Int[],
    Matrix{Int}[],
    Matrix{Int}[],
)

mutable struct RootKKTWorkspace{T,E}
    equality::E
    soc::Vector{SOCBlock{T}}
    entries::Vector{Tuple{Vector{Int},Matrix{T}}}
    matrix::SparseArrays.SparseMatrixCSC{T,Int32}
    indices::RootKKTIndices
    linear_solver::Any
    linear_solver_type::Type
    linear_solver_options::NamedTuple
    rhs::Vector{T}
    scale::Vector{T}
    saved_rhs::Vector{T}
    refinement::Vector{T}
    trial::Vector{T}
    cone_offset::Int
    regularization::T
    cone_hessian::Bool
    equilibrate::Bool
    psd_cone_hessian::BitVector
    soc_cone_hessian::BitVector
end
needs_schur_buffer(::RootKKT) = false
needs_jacobian_buffers(::RootKKT) = false
function check_schur(::RootKKT, ::AbstractAlgorithm)
    throw(ArgumentError("RootKKT requires PrimalDual with NT scaling."))
end
check_schur(::RootKKT, ::PrimalDual{<:NT}) = nothing
function schur_workspace(component::RootKKT, ::Type{T}, n, model) where {T}
    eq = schur_workspace(AugmentedKKT(), T, n, model)
    return RootKKTWorkspace(
        eq,
        SOCBlock{T}[SOCBlock(eq.model, idx) for idx in LRO.second_order_indices(eq.model)],
        Tuple{Vector{Int},Matrix{T}}[],
        SparseArrays.spzeros(T, Int32, 0, 0),
        RootKKTIndices(),
        nothing,
        component.linear_solver,
        component.options,
        T[],
        T[],
        T[],
        T[],
        T[],
        0,
        zero(T),
        component.cone_hessian,
        component.equilibrate,
        BitVector(),
        BitVector(),
    )
end

function _initialize_root_kkt!(ws::RootKKTWorkspace{T}, solver) where {T}
    n = solver.m
    offset = ws.cone_offset = n + length(ws.equality.eta)
    pairs = Tuple{Int,Int}[]
    model = ws.equality.model
    transpose_jacobian = SparseArrays.sparse(model.jacobian')
    for idx in LRO.matrix_indices(solver.model)
        d = LRO.side_dimension(solver.model, idx)
        first = model.dim.offsets[idx.value]+1
        rows = first:(first+d*d-1)
        block = transpose_jacobian[:, rows]
        cols = sort!(unique(SparseArrays.rowvals(block)))
        A = -Matrix(transpose(block[cols, :]))
        push!(ws.entries, (cols, A))
        k = d*(d+1)÷2
        nonzeros=0
        for column in axes(A, 2), q = 1:d, p = 1:q
            nonzeros+=!iszero(A[p+(q-1)*d, column])
        end
        cone_hessian=ws.cone_hessian && k*(k+1)÷2+nonzeros < k*(length(cols)+1)
        push!(ws.psd_cone_hessian, cone_hessian)
        for (column, j) in enumerate(cols)
            row=offset
            for q = 1:d, p = 1:q
                row+=1
                if !cone_hessian || !iszero(A[p+(q-1)*d, column])
                    push!(pairs, (row, j))
                end
            end
        end
        if cone_hessian
            for column = 1:k, row = column:k
                push!(pairs, (offset+row, offset+column))
            end
        end
        offset += k
    end
    for b in ws.soc
        nrows = length(b.X)
        cone_hessian=ws.cone_hessian &&
                     nrows*(nrows+1)÷2+count(!iszero, b.A) < nrows*(length(b.columns)+1)
        push!(ws.soc_cone_hessian, cone_hessian)
        for (column, j) in enumerate(b.columns), i = 1:nrows
            if !cone_hessian || !iszero(b.A[i, column])
                push!(pairs, (offset+i, j))
            end
        end
        if cone_hessian
            for column = 1:nrows, row = column:nrows
                push!(pairs, (offset+row, offset+column))
            end
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
    positions = SparseArrays.SparseMatrixCSC(
        size(ws.matrix, 1),
        size(ws.matrix, 2),
        ws.matrix.colptr,
        ws.matrix.rowval,
        collect(1:SparseArrays.nnz(ws.matrix)),
    )
    psd_positions, soc_positions = Matrix{Int}[], Matrix{Int}[]
    offset = ws.cone_offset
    psd_hessian, soc_hessian=Matrix{Int}[], Matrix{Int}[]
    for (block_index, (b, (columns, A))) in enumerate(zip(solver.blocks, ws.entries))
        nrows = b.n*(b.n+1)÷2
        push!(psd_positions, [positions[offset+i, j] for i = 1:nrows, j in columns])
        push!(
            psd_hessian,
            ws.psd_cone_hessian[block_index] ?
            [i>=j ? positions[offset+i, offset+j] : 0 for i = 1:nrows, j = 1:nrows] :
            zeros(Int, 0, 0),
        )
        offset += nrows
    end
    for (block_index, b) in enumerate(ws.soc)
        nrows = length(b.X)
        push!(soc_positions, [positions[offset+i, j] for i = 1:nrows, j in b.columns])
        push!(
            soc_hessian,
            ws.soc_cone_hessian[block_index] ?
            [i>=j ? positions[offset+i, offset+j] : 0 for i = 1:nrows, j = 1:nrows] :
            zeros(Int, 0, 0),
        )
        offset += nrows
    end
    scalar_positions = zeros(Int, SparseArrays.nnz(C))
    for k in axes(C, 2), p in SparseArrays.nzrange(C, k)
        scalar_positions[p] = positions[offset+k, C.rowval[p]]
    end
    eq_positions = [positions[n+I[k], J[k]] for k in eachindex(V)]
    diagonal = [positions[i, i] for i in axes(ws.matrix, 1)]
    ws.indices = RootKKTIndices(
        psd_positions,
        soc_positions,
        scalar_positions,
        eq_positions,
        I,
        J,
        diagonal,
        psd_hessian,
        soc_hessian,
    )
    total = size(ws.matrix, 1)
    ws.rhs = zeros(T, total)
    ws.scale = ones(T, total)
    ws.saved_rhs = zeros(T, total)
    ws.refinement = zeros(T, total)
    ws.trial = zeros(T, total)
    # Static LDL constructors perform an initial numeric factorization.
    # A signed identity preserves the symbolic pattern and is nonsingular.
    fill!(ws.matrix.nzval, zero(T))
    for i in axes(ws.matrix, 1)
        ws.matrix.nzval[ws.indices.diagonal[i]] = i <= solver.m ? one(T) : -one(T)
    end
    ws.linear_solver = root_linear_solver(
        ws.linear_solver_type,
        ws.matrix,
        solver.m;
        ws.linear_solver_options...,
    )
    return
end

function factorize_schur!(ws::RootKKTWorkspace{T}, solver::MadSDPSolver) where {T}
    if isnothing(ws.linear_solver)
        _initialize_root_kkt!(ws, solver)
    end
    K = ws.matrix
    fill!(K.nzval, zero(T))
    for i = (ws.cone_offset+1):length(ws.rhs)
        K.nzval[ws.indices.diagonal[i]]=-one(T)
    end
    offset = ws.cone_offset
    fill!(ws.scale, one(T))
    ws.scale[1:solver.m] .= zero(T)
    for (block_index, (b, (cols, A))) in enumerate(zip(solver.blocks, ws.entries))
        G = b.work.dir.G
        if ws.psd_cone_hessian[block_index]
            LinearAlgebra.mul!(b.work.dir.W, b.work.dir.Gi', b.work.dir.Gi)
            _set_psd_hessian!(K, ws.indices.psd_hessian[block_index], b.work.dir.W)
        end
        for (k, j) in enumerate(cols)
            if ws.psd_cone_hessian[block_index]
                copyto!(b.buf2, reshape(view(A, :, k), b.n, b.n))
            else
                LinearAlgebra.mul!(b.buf, reshape(view(A, :, k), b.n, b.n), G)
                LinearAlgebra.mul!(b.buf2, G', b.buf)
            end
            row = offset
            for q = 1:b.n, p = 1:q
                row += 1
                value = b.buf2[p, q] * (p == q ? one(T) : sqrt(T(2)))
                index=ws.indices.psd[block_index][row-offset, k]
                if !iszero(index)
                    K.nzval[index]=value
                end
                ws.scale[j] += abs2(value)
            end
        end
        offset += b.n*(b.n+1)÷2
    end
    for (block_index, b) in enumerate(ws.soc)
        if ws.soc_cone_hessian[block_index]
            for j in eachindex(b.X), i = j:length(b.X)
                K.nzval[ws.indices.soc_hessian[block_index][i, j]]=-LinearAlgebra.dot(
                    view(b.Gi, :, i),
                    view(b.Gi, :, j),
                )
            end
        end
        for (k, j) in enumerate(b.columns), i in eachindex(b.X)
            value = ws.soc_cone_hessian[block_index] ? b.A[i, k] : b.weighted_A[i, k]
            index=ws.indices.soc[block_index][i, k]
            if !iszero(index)
                K.nzval[index]=value
            end
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
            K.nzval[ws.indices.scalar[p]] = value
            ws.scale[j] += abs2(value)
        end
    end
    if ws.cone_hessian
        ws.scale[1:solver.m] .= zero(T)
        for i = (ws.cone_offset+1):length(ws.rhs)
            ws.scale[i]=inv(sqrt(-K.nzval[ws.indices.diagonal[i]]))
        end
        for j = 1:solver.m, p in SparseArrays.nzrange(K, j)
            row=K.rowval[p]
            if row > ws.cone_offset
                ws.scale[j]+=abs2(K.nzval[p]*ws.scale[row])
            end
        end
    end
    for i = 1:solver.m
        ws.scale[i] = iszero(ws.scale[i]) ? one(T) : inv(sqrt(ws.scale[i]))
    end
    ws.scale[(solver.m+1):ws.cone_offset] .= zero(T)
    I, J, V = ws.indices.eq_rows, ws.indices.eq_cols, SparseArrays.nonzeros(ws.equality.E)
    for k in eachindex(V)
        K.nzval[ws.indices.equality[k]] = V[k]
        ws.scale[solver.m+I[k]] += abs2(V[k]*ws.scale[J[k]])
    end
    for i = (solver.m+1):ws.cone_offset
        ws.scale[i] = iszero(ws.scale[i]) ? one(T) : inv(sqrt(ws.scale[i]))
    end
    if !ws.equilibrate
        fill!(ws.scale, one(T))
    end
    for j in axes(K, 2), p in SparseArrays.nzrange(K, j)
        K.nzval[p] *= ws.scale[K.rowval[p]]*ws.scale[j]
    end
    ws.regularization = zero(T)
    for attempt = 1:solver.opt.max_schur_reg
        succeeded = try
            MadNLP.factorize!(ws.linear_solver)
            factorization_succeeded(ws.linear_solver)
        catch err
            if err isa MadNLP.FactorizationException
                false
            else
                rethrow()
            end
        end
        if succeeded && MadNLP.is_inertia(ws.linear_solver)
            succeeded =
                MadNLP.inertia(ws.linear_solver) == (solver.m, 0, length(ws.rhs)-solver.m)
        end
        if succeeded
            return true
        end
        old = ws.regularization
        ws.regularization = iszero(old) ? max(eps(T), solver.opt.regularize_schur) : 10old
        for i = 1:ws.cone_offset
            K.nzval[ws.indices.diagonal[i]] +=
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
    gamma=T(homogeneous_residual_factor(solver.state))
    ws.rhs[1:solver.m] .= gamma .* solver.Rp
    ws.rhs[(solver.m+1):ws.cone_offset] .= gamma .* ws.equality.residual
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
            LinearAlgebra.mul!(b.buf2, dir.G', b.buf, gamma, one(T))
        end
        if corrected
            b.buf2 .-= dir.RNT
        end
        if ws.psd_cone_hessian[i]
            LinearAlgebra.mul!(b.buf, dir.Gi', b.buf2)
            LinearAlgebra.mul!(b.buf2, b.buf, dir.Gi)
            _symmetrize!(b.buf2)
        end
        for q = 1:b.n, p = 1:q
            offset += 1
            ws.rhs[offset] = b.buf2[p, q]*(p == q ? one(T) : sqrt(T(2)))
        end
    end
    for (block_index, b) in enumerate(ws.soc)
        _soc_inverse!(b.work1, b.lambda)
        @. b.work1 = mu*b.work1-b.lambda
        LinearAlgebra.mul!(b.work2, b.G, b.Rd)
        b.work1 .-= gamma .* b.work2
        if corrected
            b.work1 .+= b.correction
        end
        if ws.soc_cone_hessian[block_index]
            LinearAlgebra.mul!(b.work2, b.Gi, b.work1)
        end
        for i in eachindex(b.X)
            ws.rhs[offset+i] = ws.soc_cone_hessian[block_index] ? -b.work2[i] : -b.work1[i]
        end
        offset += length(b.X)
    end
    for k = 1:solver.n_scalar
        weight = sqrt(solver.x[k]*solver.zinv[k])
        q =
            mu*solver.zinv[k] - solver.x[k] -
            solver.x[k]*solver.zinv[k]*gamma*solver.Rd_lin[k]
        if corrected
            q += solver.corr_lin[k]
        end
        ws.rhs[offset+k] = -q/weight
    end
    _root_linear_solve!(ws, solver, solver.state, mu, corrected)
    solver.dely .= view(ws.rhs, 1:solver.m)
    ws.equality.delta_eta .= view(ws.rhs, (solver.m+1):ws.cone_offset)
    # Recover the cone direction directly; avoid subtracting large matrices.
    offset = ws.cone_offset
    for (i, b) in enumerate(solver.blocks)
        columns, coefficients = ws.entries[i]
        b.work.delS .= gamma .* b.Rd
        LinearAlgebra.mul!(
            vec(b.work.delS),
            coefficients,
            view(solver.dely, columns),
            -one(T),
            one(T),
        )
        if !iszero(homogeneous_tau_direction(solver.state))
            _axpy_sparse_dense!(
                b.work.delS,
                homogeneous_tau_direction(solver.state),
                LRO.grad(solver.model, LRO.MatrixIndex(i)),
            )
        end
        _symmetrize!(b.work.delS)
        for q = 1:b.n, p = 1:q
            offset += 1
            value = ws.rhs[offset]/(p == q ? one(T) : sqrt(T(2)))
            b.buf[p, q] = b.buf[q, p] = value
        end
        if ws.psd_cone_hessian[i]
            copyto!(b.work.delX, b.buf)
            LinearAlgebra.mul!(b.buf2, b.work.dir.Gi, b.buf)
            LinearAlgebra.mul!(b.work.dir.M, b.buf2, b.work.dir.Gi')
        else
            copyto!(b.work.dir.M, b.buf)
            LinearAlgebra.mul!(b.buf2, b.work.dir.G, b.buf)
            LinearAlgebra.mul!(b.work.delX, b.buf2, b.work.dir.G')
        end
        _symmetrize!(b.work.delX)
        LinearAlgebra.mul!(b.buf2, b.work.delS, b.work.dir.G)
        LinearAlgebra.mul!(b.work.dir.tmp, b.work.dir.G', b.buf2)
    end
    for (block_index, b) in enumerate(ws.soc)
        if ws.soc_cone_hessian[block_index]
            copyto!(b.delX, view(ws.rhs, (offset+1):(offset+length(b.X))))
        else
            LinearAlgebra.mul!(b.delX, b.G, view(ws.rhs, (offset+1):(offset+length(b.X))))
        end
        b.delS .= gamma .* b.Rd
        LinearAlgebra.mul!(b.delS, b.A, view(solver.dely, b.columns), -1, true)
        b.delS .+= homogeneous_tau_direction(solver.state) .* b.constant
        offset += length(b.X)
    end
    LinearAlgebra.mul!(solver.delz, _inner(solver.model).C_lin', solver.dely)
    solver.delz .=
        gamma .* solver.Rd_lin .- solver.delz .+
        homogeneous_tau_direction(solver.state) .* _inner(solver.model).d_lin
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
        b.Rd .= cone_data_scale(solver.state) .* b.constant .- b.S
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
        residual = max(residual, LinearAlgebra.norm(b.Rd, Inf))
    end
    return residual
end
function initialize_constraints!(ws::RootKKTWorkspace{T}, solver) where {T}
    initialize_constraints!(ws.equality, solver)
    if !isnothing(ws.linear_solver) && ws.cone_offset != solver.m+length(ws.equality.eta)
        ws.linear_solver=nothing
        empty!(ws.entries)
        empty!(ws.psd_cone_hessian)
        empty!(ws.soc_cone_hessian)
    end
    if isnothing(ws.linear_solver)
        _initialize_root_kkt!(ws, solver)
    end
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

factorization_succeeded(::MadNLP.AbstractLinearSolver) = true
factorization_succeeded(solver::MadNLP.UmfpackSolver) =
    SparseArrays.UMFPACK.issuccess(solver.inner)
factorization_succeeded(solver::MadNLP.LDLSolver) =
    MadNLP.LDLF.factorized(solver.inner) && all(isfinite, solver.inner.d)

function root_linear_solver(constructor, matrix, npositive; parameters...)
    options = MadNLP.default_options(constructor)
    for (key, value) in parameters
        setproperty!(options, key, value)
    end
    return constructor(matrix; opt = options)
end
function root_linear_solver(
    ::Type{MadNLP.LDLSolver},
    matrix::SparseArrays.SparseMatrixCSC{T},
    npositive;
    parameters...,
) where {T}
    options = MadNLP.default_options(MadNLP.LDLSolver)
    options.ldl_regularization_positive = sqrt(eps(T))
    options.ldl_regularization_negative = -sqrt(eps(T))
    options.ldl_regularization_tolerance = eps(T)^(T(3)/4)
    for (key, value) in parameters
        setproperty!(options, key, value)
    end
    options.ldl_positive_size = npositive
    return MadNLP.LDLSolver(matrix; opt = options)
end
factorization_succeeded(solver::MadNLP.MumpsSolver) = !solver.is_singular

const RootKKTSolver =
    MadSDPSolver{T,M,B,A,AS,I,SW} where {T,M,B,A,AS,I,SW<:RootKKTWorkspace}

# Root KKT already stores each PSD map on its active variable columns.
# Use those maps for residuals and slacks, without legacy Jacobian caches.
function update_residuals!(solver::RootKKTSolver)
    ws = solver.schur
    solver.Rp .= cone_data_scale(solver.state) .* LRO.cons_constant(solver.model)
    for (i, b) in enumerate(solver.blocks)
        columns, A = ws.entries[i]
        for (k, j) in enumerate(columns)
            solver.Rp[j] -= LinearAlgebra.dot(view(A, :, k), vec(b.X))
        end
        _copy_to_dense!(b.Rd, LRO.grad(solver.model, LRO.MatrixIndex(i)))
        b.Rd .= cone_data_scale(solver.state) .* b.Rd .- b.S
        LinearAlgebra.mul!(vec(b.Rd), A, view(solver.y, columns), -1, true)
        solver.rd_nonzero[i] =
            LinearAlgebra.norm(b.Rd, Inf) >
            100eps(eltype(solver.y))*cone_data_scale(solver.state)*max(1, solver.cnorm[i])
    end
    if solver.n_scalar > 0
        data = _inner(solver.model)
        LinearAlgebra.mul!(solver.Rp, data.C_lin, solver.x, -1, true)
        LinearAlgebra.mul!(solver.Rd_lin, data.C_lin', solver.y)
        solver.Rd_lin .=
            cone_data_scale(solver.state) .* data.d_lin .- solver.z .- solver.Rd_lin
    end
    additional_residuals!(ws, solver)
    return
end

function _set_dual_slacks!(solver, ws::RootKKTWorkspace)
    if isnothing(ws.linear_solver)
        _initialize_root_kkt!(ws, solver)
    end
    for (i, b) in enumerate(solver.blocks)
        columns, A = ws.entries[i]
        _copy_to_dense!(b.S, LRO.grad(solver.model, LRO.MatrixIndex(i)))
        LinearAlgebra.mul!(vec(b.S), A, view(solver.y, columns), -1, true)
        _symmetrize!(b.S)
    end
    if solver.n_scalar > 0
        data = _inner(solver.model)
        LinearAlgebra.mul!(solver.z, data.C_lin', solver.y)
        solver.z .= data.d_lin .- solver.z
    end
    return
end

function root_linear_residual!(ws, solver, direction)
    copyto!(ws.refinement, ws.saved_rhs)
    LinearAlgebra.mul!(
        ws.refinement,
        LinearAlgebra.Symmetric(ws.matrix, :L),
        direction,
        -1,
        true,
    )
    for i = 1:ws.cone_offset
        ws.refinement[i] += (i <= solver.m ? 1 : -1)*ws.regularization*direction[i]
    end
    return maximum(abs(ws.refinement[i]/ws.scale[i]) for i in eachindex(ws.rhs))
end

factorization_succeeded(solver::MadNLP.QDLDLSolver) = !solver.is_singular
function root_linear_solver(::Type{MadNLP.QDLDLSolver}, matrix, npositive; parameters...)
    options = MadNLP.default_options(MadNLP.QDLDLSolver)
    options.qdldl_positive_size = npositive
    for (key, value) in parameters
        setproperty!(options, key, value)
    end
    return MadNLP.QDLDLSolver(matrix; opt = options)
end

needs_normal_equations(::RootKKTWorkspace) = false

function refine_root_direction!(ws::RootKKTWorkspace{T}, solver, linear_solver) where {T}
    MadNLP.solve_linear_system!(linear_solver, ws.rhs)
    residual = root_linear_residual!(ws, solver, ws.rhs)
    rhs_norm = maximum(abs(ws.saved_rhs[i]/ws.scale[i]) for i in eachindex(ws.rhs))
    tolerance =
        max(T(10)*eps(T), solver.opt.tol*T(1e-3))*max(
            T(cone_data_scale(solver.state)),
            rhs_norm,
        )
    for iteration = 1:5
        if residual <= tolerance
            break
        end
        MadNLP.solve_linear_system!(linear_solver, ws.refinement)
        ws.trial .= ws.rhs .+ ws.refinement
        next_residual = root_linear_residual!(ws, solver, ws.trial)
        if next_residual >= residual
            break
        end
        copyto!(ws.rhs, ws.trial)
        stagnated = next_residual >= residual/2
        residual = next_residual
        if stagnated
            break
        end
    end
    return residual, tolerance
end

# NT coordinates already have X = S = Diagonal(D). Reuse the scaled
# direction from the root solve instead of factoring X and solving with Lx.
prepare_step!(::EigenStep, ::SDPBlock, ::RootKKTWorkspace) = true
function _nt_step_bound!(workspace, direction, D, tau)
    matrix = workspace.eigen.scratch
    for j in axes(matrix, 2), i in axes(matrix, 1)
        matrix[i, j] = direction[i, j]/sqrt(D[i]*D[j])
    end
    lambda = LRO.eigmin!(workspace.eigen, matrix)
    return lambda >= 0 ? one(lambda) : min(one(lambda), -tau/lambda)
end
function step_lengths(step::EigenStep, solver::RootKKTSolver)
    T = eltype(solver.y)
    alpha = beta = one(T)
    for b in solver.blocks
        direction = b.work.dir
        alpha =
            min(alpha, _nt_step_bound!(b.work.step, direction.M, direction.D, T(step.τ)))
        beta =
            min(beta, _nt_step_bound!(b.work.step, direction.tmp, direction.D, T(step.τ)))
    end
    if solver.n_scalar > 0
        alpha = min(alpha, T(step.τ)*_max_scalar_step(solver.x, solver.delx))
        beta = min(beta, T(step.τ)*_max_scalar_step(solver.z, solver.delz))
    end
    return additional_step_lengths(solver.schur, alpha, beta, step.τ)
end

# svec(H*X*H) as a symmetric operator on the triangular coordinates of X.
function _set_psd_hessian!(matrix, indices, H)
    T=eltype(H)
    row=0
    for q in axes(H, 2), p = 1:q
        row+=1
        column=0
        for s in axes(H, 2), r = 1:s
            column+=1
            if row < column
                continue
            end
            value=H[p, r]*H[q, s]
            if r != s
                value=(value+H[p, s]*H[q, r])/sqrt(T(2))
            end
            if p != q
                value*=sqrt(T(2))
            end
            matrix.nzval[indices[row, column]]=-value
        end
    end
    return
end

needs_schur_buffer(::SparseKKT) = false
needs_jacobian_buffers(::SparseKKT) = false
check_schur(::SparseKKT, algorithm::AbstractAlgorithm) = check_schur(RootKKT(), algorithm)
function schur_workspace(component::SparseKKT, T, n, model)
    return schur_workspace(
        RootKKT(;
            linear_solver = component.linear_solver,
            cone_hessian = true,
            component.options...,
        ),
        T,
        n,
        model,
    )
end

function _base_root_linear_solve!(ws, solver)
    ws.rhs .*= ws.scale
    copyto!(ws.saved_rhs, ws.rhs)
    refine_root_direction!(ws, solver, ws.linear_solver)
    ws.rhs .*= ws.scale
    return
end
_root_linear_solve!(ws, solver, state, mu, corrected) = _base_root_linear_solve!(ws, solver)
