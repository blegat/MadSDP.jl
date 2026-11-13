# Equality-implied constant inequalities do not belong in the barrier. In
# particular, fixing all but one entry of a simplex also fixes its final entry.
# Retaining an active bound on that entry would destroy strict feasibility.
function scalar_block_bounds(model::LRO.Model{T}) where {T}
    C, d, rows, signs = LRO.scalar_bounds(model; equalities = false)
    eq = filter(i -> model.meta.lcon[i] == model.meta.ucon[i], 1:LRO.num_scalars(model))
    E = SparseArrays.sparse(model.jacobian[eq, :]')
    target = model.meta.lcon[eq] - model.constant[eq]
    fixed, values = falses(model.meta.nvar), zeros(T, model.meta.nvar)
    changed = true
    while changed
        changed = false
        for k in axes(E, 2)
            unknown, index, coefficient = 0, 0, zero(T)
            residual = target[k]
            for p in SparseArrays.nzrange(E, k)
                j, v = E.rowval[p], E.nzval[p]
                if fixed[j]
                    residual -= v*values[j]
                else
                    unknown += 1
                    index, coefficient = j, v
                end
            end
            if unknown == 1
                values[index] = residual/coefficient
                fixed[index] = true
                changed = true
            end
        end
    end
    keep = trues(length(rows))
    for k in eachindex(rows)
        constant = d[k]
        determined = true
        for p in SparseArrays.nzrange(C, k)
            j = C.rowval[p]
            if !fixed[j]
                determined = false
                break
            end
            constant -= C.nzval[p]*values[j]
        end
        if determined && constant >= -100eps(T)*max(one(T), abs(d[k]))
            keep[k] = false
        end
    end
    return C[:, keep], d[keep], rows[keep], signs[keep]
end
