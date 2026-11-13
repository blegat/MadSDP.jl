# If B[:, j] = svec(G' Aⱼ G), including the scaled scalar block, then
# H = B' B. Column equilibration gives B D[:, p] = P' Q R.
# Projecting the primal direction with Q avoids recovering it by subtracting
# nearly equal terms after a normal-equation solve.
mutable struct QRSchurWorkspace{T}
    entries::Vector{Vector{Tuple{Int,Matrix{T}}}}
    factor::Union{Nothing,SparseArrays.SPQR.QRSparse{T,Int}}
    scale::Vector{T}
    base::Vector{T}
end

function check_schur(::QRSchur, ::AbstractAlgorithm)
    throw(ArgumentError("QRSchur requires PrimalDual with the NT direction."))
end
check_schur(::QRSchur, ::PrimalDual{<:NT}) = nothing

needs_schur_buffer(::QRSchur) = false

function schur_workspace(::QRSchur, ::Type{T}, m) where {T}
    if T != Float64
        throw(ArgumentError("QRSchur requires Float64 data."))
    end
    return QRSchurWorkspace{T}(Vector{Tuple{Int,Matrix{T}}}[], nothing, ones(T, m), T[])
end

function factorize_schur!(ws::QRSchurWorkspace{T}, solver::MadSDPSolver) where {T}
    if isempty(ws.entries)
        model = _inner(solver.model)
        for idx in LRO.matrix_indices(model)
            entries = Tuple{Int,Matrix{T}}[]
            for j = 1:solver.m
                A = LRO.jac(model, j, idx)
                if !iszero(A)
                    push!(entries, (j, Matrix(A)))
                end
            end
            push!(ws.entries, entries)
        end
    end
    rows, cols, values = Int[], Int[], T[]
    offset = 0
    for (i, b) in enumerate(solver.blocks)
        G = b.work.dir.G
        for (j, A) in ws.entries[i]
            LinearAlgebra.mul!(b.buf, A, G)
            LinearAlgebra.mul!(b.buf2, G', b.buf)
            k = 0
            for col = 1:b.n, row = 1:col
                k += 1
                v = b.buf2[row, col]
                if row != col
                    v *= sqrt(T(2))
                end
                if !iszero(v)
                    push!(rows, offset + k)
                    push!(cols, j)
                    push!(values, v)
                end
            end
        end
        offset += b.n * (b.n + 1) ÷ 2
    end
    C = _inner(solver.model).C_lin
    for k in axes(C, 2)
        weight = sqrt(solver.x[k] / solver.z[k])
        for p in SparseArrays.nzrange(C, k)
            push!(rows, offset + k)
            push!(cols, SparseArrays.rowvals(C)[p])
            push!(values, weight * SparseArrays.nonzeros(C)[p])
        end
    end
    offset += solver.n_scalar
    B = SparseArrays.sparse(rows, cols, values, offset, solver.m)
    resize!(ws.base, offset)
    for j in axes(B, 2)
        magnitude = LinearAlgebra.norm(view(B, :, j))
        if iszero(magnitude) || !isfinite(magnitude)
            return false
        end
        ws.scale[j] = inv(magnitude)
        for p in SparseArrays.nzrange(B, j)
            SparseArrays.nonzeros(B)[p] *= ws.scale[j]
        end
    end
    if offset < solver.m
        return false
    end
    ws.factor = LinearAlgebra.qr(B; tol = zero(T))
    return all(i -> !iszero(ws.factor.R[i, i]), 1:solver.m)
end

function solve_directions!(
    ws::QRSchurWorkspace{T},
    solver::MadSDPSolver,
    μ::T,
    corrected::Bool,
) where {T}
    base = ws.base
    fill!(base, zero(T))
    offset = 0
    for b in solver.blocks
        dir = b.work.dir
        LinearAlgebra.mul!(b.buf, b.Rd, dir.G)
        LinearAlgebra.mul!(b.buf2, dir.G', b.buf)
        b.buf2 .*= -one(T)
        for i = 1:b.n
            b.buf2[i, i] += μ / dir.D[i] - dir.D[i]
        end
        if corrected
            b.buf2 .+= dir.RNT
        end
        k = offset
        for col = 1:b.n, row = 1:col
            k += 1
            base[k] = row == col ? b.buf2[row, col] : sqrt(T(2)) * b.buf2[row, col]
        end
        offset += b.n * (b.n + 1) ÷ 2
    end
    for k = 1:solver.n_scalar
        weight = sqrt(solver.x[k] / solver.z[k])
        product = sqrt(solver.x[k] * solver.z[k])
        base[offset+k] = μ / product - product - weight * solver.Rd_lin[k]
        if corrected
            base[offset+k] += solver.corr_lin[k] / weight
        end
    end
    F = ws.factor::SparseArrays.SPQR.QRSparse{T,Int}
    R, p = LinearAlgebra.UpperTriangular(F.R), F.pcol
    # B' Δx̂ = Rp and Δx̂ = base + B Δy.
    # Work in permuted row coordinates: SPQR's Q does not include F.prow.
    qbase = F.Q' * base[F.prow]
    c = R' \ (ws.scale .* solver.Rp)[p]
    qbase[1:solver.m] .= c .- view(qbase, 1:solver.m)
    qbase[(solver.m+1):end] .= zero(T)
    solver.dely[p] = ws.scale[p] .* (R \ qbase[1:solver.m])
    direction = copy(base)
    direction[F.prow] .+= F.Q * qbase
    offset = 0
    for (i, b) in enumerate(solver.blocks)
        G = b.work.dir.G
        k = offset
        for col = 1:b.n, row = 1:col
            k += 1
            v = row == col ? direction[k] : direction[k] / sqrt(T(2))
            b.buf2[row, col] = b.buf2[col, row] = v
        end
        LinearAlgebra.mul!(b.buf, G, b.buf2)
        LinearAlgebra.mul!(b.work.delX, b.buf, G')
        _symmetrize!(b.work.delX)
        AT = LRO.unsafe_jtprod(solver.model, solver.dely, LRO.MatrixIndex(i))
        copyto!(b.work.delS, b.Rd)
        _axpy_sparse_dense!(b.work.delS, -one(T), AT)
        _symmetrize!(b.work.delS)
        offset += b.n * (b.n + 1) ÷ 2
    end
    LinearAlgebra.mul!(solver.delz, _inner(solver.model).C_lin', solver.dely)
    solver.delz .= solver.Rd_lin .- solver.delz
    for k = 1:solver.n_scalar
        solver.delx[k] = sqrt(solver.x[k] / solver.z[k]) * direction[offset+k]
    end
    return
end
