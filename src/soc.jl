# Lorentz-cone Jordan algebra, with e = (1,0,...) and Euclidean pairing.
# Its central path has s ∘ z = μe and dot(s,z) = μ.
_soc_det(x) =
    (x[1]-LinearAlgebra.norm(view(x, 2:length(x))))*(
        x[1]+LinearAlgebra.norm(view(x, 2:length(x)))
    )
_soc_interior(x) = x[1] > LinearAlgebra.norm(view(x, 2:length(x)))

function _soc_power!(out, x, inverse::Bool)
    radius = LinearAlgebra.norm(view(x, 2:length(x)))
    plus, minus = sqrt(x[1]+radius), sqrt(x[1]-radius)
    out[1] = inverse ? (inv(plus)+inv(minus))/2 : (plus+minus)/2
    coefficient = inverse ? -inv(plus*minus*(plus+minus)) : inv(plus+minus)
    for i = 2:length(x)
        out[i] = coefficient*x[i]
    end
    return out
end
function _soc_inverse!(out, x)
    determinant = _soc_det(x)
    out[1] = x[1]/determinant
    for i = 2:length(x)
        out[i] = -x[i]/determinant
    end
    return out
end
function _soc_product!(out, x, y)
    head = LinearAlgebra.dot(x, y)
    for i = 2:length(x)
        out[i] = x[1]*y[i]+y[1]*x[i]
    end
    out[1] = head
    return out
end
function _soc_ldiv!(out, x, y)
    head =
        (x[1]*y[1]-LinearAlgebra.dot(view(x, 2:length(x)), view(y, 2:length(y))))/_soc_det(
            x,
        )
    for i = 2:length(x)
        out[i] = (y[i]-x[i]*head)/x[1]
    end
    out[1] = head
    return out
end
function _soc_quadratic_mul!(out, w, x)
    tail_dot = LinearAlgebra.dot(view(w, 2:length(w)), view(x, 2:length(x)))
    determinant = _soc_det(w)
    head = (2w[1]^2-determinant)*x[1]+2w[1]*tail_dot
    factor = 2*(w[1]*x[1]+tail_dot)
    for i = 2:length(x)
        out[i] = determinant*x[i]+factor*w[i]
    end
    out[1] = head
    return out
end
function _soc_quadratic_matrix!(matrix, w)
    determinant = _soc_det(w)
    for j in eachindex(w), i in eachindex(w)
        matrix[i, j] = 2w[i]*w[j]
    end
    matrix[1, 1] -= determinant
    for i = 2:length(w)
        matrix[i, i] += determinant
    end
    return matrix
end

struct SOCBlock{T}
    constant::Vector{T}
    columns::Vector{Int}
    A::Matrix{T}
    weighted_A::Matrix{T}
    X::Vector{T}
    S::Vector{T}
    Rd::Vector{T}
    delX::Vector{T}
    delS::Vector{T}
    correction::Vector{T}
    G::Matrix{T}
    Gi::Matrix{T}
    lambda::Vector{T}
    work1::Vector{T}
    work2::Vector{T}
    work3::Vector{T}
end
function SOCBlock(model::LRO.Model{T}, index::LRO.SOCIndex) where {T}
    cone = LRO.VectorizedConeVector(model.constant, model.dim)
    rows = Base.to_index(cone, index)
    J = model.jacobian[rows, :]
    cols = unique(SparseArrays.findnz(J)[2])
    A = -Matrix(J[:, cols])
    n = length(rows)
    vector() = zeros(T, n)
    matrix() = zeros(T, n, n)
    return SOCBlock(
        collect(cone[index]),
        cols,
        A,
        similar(A),
        vector(),
        vector(),
        vector(),
        vector(),
        vector(),
        vector(),
        matrix(),
        matrix(),
        vector(),
        vector(),
        vector(),
        vector(),
    )
end
function prepare_soc!(b::SOCBlock)
    if !_soc_interior(b.X) || !_soc_interior(b.S)
        return false
    end
    _soc_power!(b.work1, b.X, false)
    _soc_quadratic_mul!(b.work2, b.work1, b.S)
    if !_soc_interior(b.work2)
        return false
    end
    _soc_power!(b.work3, b.work2, true)
    _soc_quadratic_mul!(b.work2, b.work1, b.work3)
    _soc_power!(b.work1, b.work2, false)
    _soc_quadratic_matrix!(b.G, b.work1)
    _soc_power!(b.work1, b.work2, true)
    _soc_quadratic_matrix!(b.Gi, b.work1)
    LinearAlgebra.mul!(b.lambda, b.G, b.S)
    LinearAlgebra.mul!(b.weighted_A, b.G, b.A)
    return true
end
function soc_correction!(b::SOCBlock)
    LinearAlgebra.mul!(b.work1, b.Gi, b.delX)
    LinearAlgebra.mul!(b.work2, b.G, b.delS)
    _soc_product!(b.work3, b.work1, b.work2)
    _soc_ldiv!(b.correction, b.lambda, b.work3)
    b.correction .*= -1
    return
end

# First positive determinant root, also respecting the head coordinate.
function _soc_max_step(x, dx)
    T = eltype(x)
    bound = dx[1] < 0 ? -x[1]/dx[1] : T(Inf)
    a = dx[1]^2-LinearAlgebra.dot(view(dx, 2:length(dx)), view(dx, 2:length(dx)))
    b = 2*(x[1]*dx[1]-LinearAlgebra.dot(view(x, 2:length(x)), view(dx, 2:length(dx))))
    c = _soc_det(x)
    if iszero(a)
        return b < 0 ? min(bound, -c/b) : bound
    end
    discriminant = b*b-4a*c
    if discriminant >= 0
        q = -(b+copysign(sqrt(discriminant), b))/2
        for root in (q/a, iszero(q) ? T(Inf) : c/q)
            if root > 0
                bound = min(bound, root)
            end
        end
    end
    return bound
end
