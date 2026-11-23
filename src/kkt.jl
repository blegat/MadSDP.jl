# The linearized KKT system of the interior-point methods as a
# `MadNLP.AbstractKKTSystem`, following MadIPM's `NormalKKTSystem`.
#
# With the complementarity `X S = μ I` linearized by a direction (see
# `AbstractDirection`), the Newton system on `(ΔX, Δx, Δy, ΔS, Δz)` is
#
#     ΔX + E(ΔS)       = r_c          Δx + (x ./ z) .* Δz = r_cl
#     A(ΔX) + C_lin Δx = r_p
#     A*(Δy) + ΔS      = r_d          C_linᵀ Δy + Δz      = r_dl
#
# where the right-hand side is stored in an `SDPKKTVector` `w` with
# `w.X = r_c`, `w.x = r_cl`, `w.y = r_p`, `w.S = r_d` and `w.z = r_dl`.
# `SchurKKTSystem` eliminates `ΔS`, `ΔX`, `Δz` and `Δx` to obtain the Schur
# complement system `H Δy = h` with
#
#     H = Σᵢ ⟨Aᵢⱼ, Lᵢ Aᵢₖ Rᵢ⟩ + C_lin diag(x ./ z) C_linᵀ
#     h = r_p - A(r_c - E(r_d)) - C_lin (r_cl - (x ./ z) .* r_dl)
#
# where `(Lᵢ, Rᵢ)` are the scalings of `E`, see `schur_scaling`.
#
# `LRO.schur_complement!` assembles the dense matrix `H`. The linear solver
# is built once on `aug_com`, which is `H` itself for a dense solver
# (`MadNLP.input_type(LS) == :dense`, e.g. LAPACK) and a `SparseMatrixCSC`
# storing the full lower triangle of `H` for a sparse one (`:csc`, e.g.
# CHOLMOD, MUMPS, LDLFactorizations or cuDSS), whose values are copied from `H`
# before each factorization.
#
# `H` is symmetric positive definite as long as the iterates lie in the
# interior. MadNLP's solvers do not throw if the factorization fails but
# report the inertia, so the factorization is considered successful if the
# solver reports `m` positive eigenvalues, see `is_factorized`.

struct SchurKKTSystem{
    T,
    MT<:AbstractMatrix{T},
    LS<:MadNLP.AbstractLinearSolver{T},
    M,
    B<:SDPBlock{T},
    SS,
} <: MadNLP.AbstractCondensedKKTSystem{T,Vector{T},MT,MadNLP.ExactHessian{T,Vector{T}}}
    m::Int
    H::Matrix{T}            # dense m×m Schur matrix (lower-triangular meaningful)
    aug_com::MT             # matrix factorized by `linear_solver`
    linear_solver::LS
    reg::Base.RefValue{T}   # regularization added to the diagonal of `H`
    # Data of the system, shared with `MadSDPSolver`.
    model::M                # LRO.BufferedModelForSchur
    blocks::Vector{B}       # `E` and the scratch matrices of each block
    L::SS                   # scalings `(L, R)` of the Schur complement, whose
    R::SS                   # scalar parts are `x` and `1 ./ z`
    # Blocks whose `r_d` is at the roundoff level, for which `E(r_d)` is
    # not computed by `solve_kkt!`. Set by `set_rhs!`.
    rd_negligible::Vector{Bool}
    buffer_m::Vector{T}
    buffer_scalar::Vector{T}
end

# Lower triangle of a dense `m × m` matrix in CSC format with `Int32` indices
# as expected by MadNLP's sparse solvers.
function _dense_tril_csc(::Type{T}, m::Integer) where {T}
    colptr = Vector{Int32}(undef, m + 1)
    rowval = Vector{Int32}(undef, m * (m + 1) ÷ 2)
    colptr[1] = 1
    k = 0
    for j = 1:m
        for i = j:m
            k += 1
            rowval[k] = i
        end
        colptr[j+1] = k + 1
    end
    return SparseArrays.SparseMatrixCSC{T,Int32}(m, m, colptr, rowval, zeros(T, k))
end

_aug_com(H::Matrix, ::Val{:dense}) = H
_aug_com(H::Matrix{T}, ::Val{:csc}) where {T} = _dense_tril_csc(T, size(H, 1))
function _aug_com(_, ::Val{S}) where {S}
    throw(ArgumentError("SchurKKTSystem: unsupported linear solver input type `:$S`."))
end

_transfer!(::Matrix, ::Matrix) = nothing
function _transfer!(aug_com::SparseArrays.SparseMatrixCSC, H::Matrix)
    nz = SparseArrays.nonzeros(aug_com)
    m = size(H, 1)
    k = 0
    @inbounds for j = 1:m, i = j:m
        k += 1
        nz[k] = H[i, j]
    end
    return
end

# Options of `linear_solver` with the LAPACK factorization `algorithm` if the
# solver is LAPACK.
function _lapack_options(linear_solver::Type, algorithm)
    opt = MadNLP.default_options(linear_solver)
    if hasproperty(opt, :lapack_algorithm)
        opt.lapack_algorithm = algorithm
    end
    return opt
end

# `H` is positive definite so Cholesky is the default for LAPACK.
default_kkt_options(linear_solver::Type) = _lapack_options(linear_solver, MadNLP.CHOLESKY)

"""
    MadNLP.create_kkt_system(
        ::Type{SchurKKTSystem}, model, blocks, L, R, linear_solver;
        opt_linear_solver = default_kkt_options(linear_solver),
    )

Schur complement system of the PSD blocks `blocks` and the scalar variables
of `model`, the Schur complement being `⟨Aⱼ, L Aₖ R⟩`. The vectors `L` and
`R` are shared with the solver, which updates them in place.
"""
function MadNLP.create_kkt_system(
    ::Type{SchurKKTSystem},
    model::LRO.BufferedModelForSchur{T},
    blocks::Vector{<:SDPBlock{T}},
    L,
    R,
    linear_solver::Type;
    opt_linear_solver = default_kkt_options(linear_solver),
) where {T}
    m = model.meta.ncon
    H = zeros(T, m, m)
    aug_com = _aug_com(H, Val(MadNLP.input_type(linear_solver)))
    # The Schur complement is definite: no dual block.
    MadNLP.set_block_structure!(opt_linear_solver, m, 0)
    ls = linear_solver(aug_com; opt = opt_linear_solver)
    return SchurKKTSystem(
        Int(m),
        H,
        aug_com,
        ls,
        Ref(zero(T)),
        model,
        blocks,
        L,
        R,
        zeros(Bool, length(blocks)),
        zeros(T, m),
        zeros(T, LRO.num_scalars(model)),
    )
end

MadNLP.num_variables(kkt::SchurKKTSystem) = kkt.m
MadNLP.get_kkt(kkt::SchurKKTSystem) = kkt.aug_com

function MadNLP.initialize!(kkt::SchurKKTSystem{T}) where {T}
    fill!(kkt.H, zero(T))
    kkt.reg[] = zero(T)
    fill!(kkt.rd_negligible, false)
    return
end

# Increments the regularization of the diagonal of `H`; `H` is definite so
# there is no dual regularization.
function MadNLP.regularize_diagonal!(kkt::SchurKKTSystem, primal, _)
    kkt.reg[] += primal
    return
end

function MadNLP.build_kkt!(kkt::SchurKKTSystem{T}) where {T}
    H = kkt.H
    LRO.schur_complement!(kkt.model, kkt.L, kkt.R, H)
    reg = kkt.reg[]
    if reg > zero(T)
        @inbounds for i in axes(H, 1)
            H[i, i] += reg
        end
    end
    _transfer!(kkt.aug_com, H)
    return
end

# Same condition as `MadNLP.is_inertia_correct(::NormalKKTSystem, ...)`.
function MadNLP.is_inertia_correct(kkt::SchurKKTSystem, num_pos, num_zero, _)
    return num_zero == 0 && num_pos == kkt.m
end

# Whether the last factorization of `kkt` succeeded. A solver that does not
# compute the inertia (e.g. LAPACK LU) cannot detect an indefinite `H`; as
# MadIPM, we assume it succeeded.
function is_factorized(kkt::MadNLP.AbstractKKTSystem)
    ls = kkt.linear_solver
    MadNLP.is_inertia(ls) || return true
    return MadNLP.is_inertia_correct(kkt, MadNLP.inertia(ls)...)
end

# Solves `H v = rhs` in place with the last factorization.
function solve_schur!(kkt::SchurKKTSystem, rhs::Vector)
    MadNLP.solve_linear_system!(kkt.linear_solver, rhs)
    return rhs
end

_scalar_scaling(kkt::SchurKKTSystem) = (kkt.L[LRO.ScalarIndex], kkt.R[LRO.ScalarIndex])

"""
    MadNLP.solve_kkt!(kkt::SchurKKTSystem, w::SDPKKTVector)

Replaces the right-hand side `w` of the linearized KKT system by its
solution `(ΔX, Δx, Δy, ΔS, Δz)`:
```
h  = r_p - A(r_c - E(r_d)) - C_lin (r_cl - (x ./ z) .* r_dl)
Δy = H \\ h
ΔX = r_c - E(r_d) + E(A*(Δy)),    ΔS = r_d - A*(Δy)
Δz = r_dl - C_linᵀ Δy,            Δx = r_cl - (x ./ z) .* Δz
```
`E(r_d)` and `E(A*(Δy))` are computed separately as `A*(Δy)` is often
sparser than `ΔS`.
"""
function MadNLP.solve_kkt!(kkt::SchurKKTSystem{T}, w::SDPKKTVector{T}) where {T}
    model = kkt.model
    h = w.y
    # h = r_p - A(r_c - E(r_d))
    for (i, b) in enumerate(kkt.blocks)
        @. b.buf = -w.X[i]
        if !kkt.rd_negligible[i]
            _apply_E!(b.buf2, b.work.dir, b, w.S[i])
            b.buf .+= b.buf2
        end
        LRO.add_jprod!(model, b.buf, h, LRO.MatrixIndex(i))
    end
    x, zinv = _scalar_scaling(kkt)
    if !isempty(w.x)
        # h -= C_lin (r_cl - (x ./ z) .* r_dl)
        @. kkt.buffer_scalar = x * zinv * w.z - w.x
        LinearAlgebra.mul!(h, _inner(model).C_lin, kkt.buffer_scalar, true, true)
    end
    solve_schur!(kkt, h)
    Δy = h
    for (i, b) in enumerate(kkt.blocks)
        ΔX, ΔS = w.X[i], w.S[i]
        if !kkt.rd_negligible[i]
            _apply_E!(b.buf2, b.work.dir, b, ΔS)
            ΔX .-= b.buf2
        end
        AT_dy = LRO.unsafe_jtprod(model, Δy, LRO.MatrixIndex(i))
        _apply_E!(b.buf2, b.work.dir, b, AT_dy)
        ΔX .+= b.buf2
        _symmetrize!(ΔX)
        _axpy_sparse_dense!(ΔS, -one(T), AT_dy)
        _symmetrize!(ΔS)
    end
    if !isempty(w.x)
        LinearAlgebra.mul!(w.z, _inner(model).C_lin', Δy, -one(T), one(T))
        @. w.x -= x * zinv * w.z
    end
    return w
end

"""
    MadNLP.mul!(w::SDPKKTVector, kkt::SchurKKTSystem, v::SDPKKTVector, α = 1, β = 0)

`w = α K v + β w` where `K` is the matrix of the linearized KKT system, so
`(K v).X = v.X + E(v.S)`, `(K v).x = v.x + (x ./ z) .* v.z`,
`(K v).y = A(v.X) + C_lin v.x`, `(K v).S = A*(v.y) + v.S` and
`(K v).z = C_linᵀ v.y + v.z`.
"""
function MadNLP.mul!(
    w::SDPKKTVector{T},
    kkt::SchurKKTSystem{T},
    v::SDPKKTVector{T},
    α = one(T),
    β = zero(T),
) where {T}
    model = kkt.model
    # A `β` of zero ignores the values of `w`, as for `LinearAlgebra.mul!`.
    iszero(β) && fill!(MadNLP.full(w), zero(T))
    Kv_y = kkt.buffer_m
    fill!(Kv_y, zero(T))
    for (i, b) in enumerate(kkt.blocks)
        _apply_E!(b.buf2, b.work.dir, b, v.S[i])
        @. w.X[i] = β * w.X[i] + α * (v.X[i] + b.buf2)
        LRO.add_jprod!(model, v.X[i], Kv_y, LRO.MatrixIndex(i))
        copyto!(b.buf, v.S[i])
        _axpy_sparse_dense!(b.buf, one(T), LRO.unsafe_jtprod(model, v.y, LRO.MatrixIndex(i)))
        @. w.S[i] = β * w.S[i] + α * b.buf
    end
    if !isempty(v.x)
        x, zinv = _scalar_scaling(kkt)
        C_lin = _inner(model).C_lin
        @. w.x = β * w.x + α * (v.x + x * zinv * v.z)
        LinearAlgebra.mul!(Kv_y, C_lin, v.x, true, true)
        LinearAlgebra.mul!(w.z, C_lin', v.y, α, β)
        @. w.z += α * v.z
    end
    @. w.y = β * w.y + α * Kv_y
    return w
end
