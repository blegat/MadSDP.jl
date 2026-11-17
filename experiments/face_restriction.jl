# julia --project=test experiments/face_restriction.jl
using LinearAlgebra, SparseArrays, Printf
import MadSDP
import LowRankOpt as LRO

# No randomness or known support is passed to adaptive_face.
function face_example(; rotated = false)
    n = 5
    A = [Matrix(1.0I, n, n), zeros(n, n)]
    A[2][1, 2] = A[2][2, 1] = 0.5
    A[2][2, 2] = 1
    S = Matrix(Diagonal([0.0, 1, 3, 4, 5]))
    y = [0.2, -0.3]
    C = S + sum(y[i] * A[i] for i = 1:2)
    X = zeros(n, n)
    X[1, 1] = 1
    # A deterministic dense Householder orthogonal transform, rounded to Float64.
    Q = Matrix(1.0I, n, n)
    if rotated
        v = [1.0, 2, -3, 4, -2]
        Q -= 2 * (v * v') / dot(v, v)
    end
    transform(M) = Matrix(Symmetric(Q * M * Q'))
    C, A, X, S = transform(C), transform.(A), transform(X), transform(S)
    b = [1.0, 0.0] # Transforming the exact construction does not recompute b.
    Qexact = Matrix{Rational{BigInt}}(I, n, n)
    if rotated
        v = Rational{BigInt}.([1, 2, -3, 4, -2])
        Qexact -= 2 * (v * v') / dot(v, v)
    end
    intended_X = Qexact[:, 1] * Qexact[:, 1]'
    return (; C, A, b, X, y, S, intended_X, support = Q[:, 1:1])
end

function maxcut_example()
    W = [0.0 5 7 6; 5 0 0 1; 7 0 0 1; 6 1 1 0]
    C = -Matrix(Diagonal(vec(sum(W; dims = 2))) - W) / 4
    A = [Matrix(Diagonal([i == j ? 1.0 : 0.0 for i = 1:4])) for j = 1:4]
    v = [1.0, -1, -1, -1]
    y = (C * v) ./ v
    S = C - Diagonal(y)
    return (;
        C,
        A,
        b = ones(4),
        X = v * v',
        y,
        S = Matrix(S),
        support = reshape(v / 2, 4, 1),
    )
end

function stored_reference(e)
    return setprecision(256) do
        C, A, b = BigFloat.(e.C), [BigFloat.(a) for a in e.A], BigFloat.(e.b)
        result = MadSDP.refine_rank_one(C, A, b, BigFloat.(e.X), BigFloat.(e.y); steps = 8)
        r = MadSDP.face_residuals(C, A, b, result.X, result.y, result.S)
        return (; result..., residuals = r, objective = dot(C, result.X))
    end
end

function evaluate(e, X, y, S)
    return setprecision(256) do
        r = MadSDP.face_residuals(
            BigFloat.(e.C),
            [BigFloat.(a) for a in e.A],
            BigFloat.(e.b),
            BigFloat.(X),
            BigFloat.(y),
            BigFloat.(S),
        )
        target = hasproperty(e, :intended_X) ? BigFloat.(e.intended_X) : BigFloat.(e.X)
        forward = norm(BigFloat.(X) - target) / max(1, norm(target))
        forward_stored =
            hasproperty(e, :reference) ?
            norm(BigFloat.(X) - e.reference.X) / max(1, norm(e.reference.X)) : forward
        return (; r..., forward, forward_stored)
    end
end

function baseline(model, schur)
    solver = MadSDP.MadSDPSolver(
        model;
        step = MadSDP.CholeskyBacktrack(),
        schur,
        tol = 1e-16,
        mu_min = 0.0,
        regularize_schur = 0.0,
        max_iter = 100,
    )
    MadSDP.solve!(solver; verbose = false)
    block = only(solver.blocks)
    G = block.work.dir.G
    weighted_condition = cond(
        reduce(
            hcat,
            [
                vec(G' * Matrix(LRO.jac(model, j, LRO.MatrixIndex(1))) * G) for
                j = 1:model.meta.ncon
            ],
        ),
    )
    return (;
        weighted_condition,
        X = block.X,
        y = solver.y,
        S = block.S,
        iterations = solver.iter,
        status = solver.status,
    )
end

function comparison(e, model, method)
    if startswith(method, "NT-")
        result = baseline(
            model,
            method == "NT-Cholesky" ? MadSDP.DenseCholesky() : MadSDP.QRSchur(),
        )
        if method == "NT-QR+round"
            X = MadSDP.face_round(e.A, e.b, Matrix(1.0I, size(e.C)...), result.X)
            y, S = MadSDP.face_dual(e.C, e.A, X)
            result = (; result..., X, y, S)
        elseif method == "NT-QR+rotate"
            rounded = MadSDP.refine_rank_one(e.C, e.A, e.b, result.X, result.y)
            result = (; result..., rounded...)
        end
        return result
    elseif method == "oracle"
        return MadSDP.face_candidate(
            e.C,
            e.A,
            e.b,
            e.support,
            Matrix(1.0I, size(e.C)...),
            zeros(length(e.b));
            tol = 1e-16,
        )
    end
    face = MadSDP.FaceOptions(;
        discard = method == "aggressive" ? 0.9 : 1e-12,
        projector_svd = method != "conservative-eigen",
    )
    return MadSDP.adaptive_face(
        model;
        face,
        tol = 1e-16,
        refine = method != "conservative-linear",
        rotate = method == "conservative+rotate",
    )
end

function run_experiments(io = stdout)
    println(
        io,
        "| example | method | primal | dual | PSD X | PSD S | gap | XS | forward | forward stored | rank | dim | angle | iterations | events | ms |",
    )
    println(
        io,
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    )
    details = IOBuffer()
    for (name, e) in (
        ("block", face_example()),
        ("dense", face_example(; rotated = true)),
        ("maxcut", maxcut_example()),
    )
        e = (; e..., reference = stored_reference(e))
        println(
            details,
            "$name 256-bit reference residuals: ",
            map(Float64, e.reference.residuals),
        )
        model = MadSDP.face_model(e.C, e.A, e.b)
        for method in (
            "NT-Cholesky",
            "NT-QR",
            "NT-QR+round",
            "NT-QR+rotate",
            "conservative-linear",
            "conservative-eigen",
            "conservative",
            "conservative+rotate",
            "aggressive",
            "oracle",
        )
            elapsed = @elapsed result = comparison(e, model, method)
            dim, angle, events = size(e.C, 1), NaN, "—"
            if startswith(method, "NT-")
                println(
                    details,
                    "$name $method status=$(result.status) cond(B)=$(result.weighted_condition) cond(B)^2=$(result.weighted_condition^2) cond(X)=$(cond(result.X)) cond(S)=$(cond(result.S))",
                )
            elseif method == "oracle"
                dim, angle, events = 1, 0.0, string(result.reason)
            else
                labels = String[]
                for ev in result.events
                    c = ev.candidate
                    dim = size(c.V, 2)
                    angle = norm(e.support - c.V * (c.V' * e.support))
                    push!(labels, "$(ev.iteration):$(dim):$(c.reason)")
                    println(
                        details,
                        "$name $method iteration=$(ev.iteration) mu=$(ev.mu) spectrum=$(ev.spectrum) angle=$angle",
                    )
                    if hasproperty(c, :residuals)
                        println(
                            details,
                            "  candidate metrics: ",
                            map(Float64, evaluate(e, c.X, c.y, c.S)),
                            " condition=",
                            c.condition,
                            " status=",
                            c.status,
                            " raw merit=",
                            MadSDP.face_merit(c.raw_residuals),
                            " rotated=",
                            c.rotated,
                        )
                    end
                end
                events = join(labels, ",")
            end
            r = evaluate(e, result.X, result.y, result.S)
            rank = count(>(1e-8 * max(1, norm(result.X))), eigvals(Symmetric(result.X)))
            @printf(
                io,
                "| %s | %s | %.2e | %.2e | %.2e | %.2e | %.2e | %.2e | %.2e | %.2e | %d | %d | %.2e | %d | %s | %.2f |\n",
                name,
                method,
                r.primal,
                r.dual,
                r.primal_psd,
                r.dual_psd,
                r.gap,
                r.complementarity,
                r.forward,
                r.forward_stored,
                rank,
                dim,
                angle,
                result.iterations,
                events,
                elapsed * 1000
            )
        end
        E = eigen(Symmetric(Float64.(e.reference.X)))
        v = E.vectors[:, end] * sqrt(E.values[end])
        Av = reduce(hcat, [a * v for a in e.A])
        J = [2Av' zeros(length(e.b), length(e.b)); Float64.(e.reference.S) -Av]
        println(
            details,
            "$name reference cond(KKT)=",
            cond(J),
            " slack spectrum=",
            eigvals(Symmetric(Float64.(e.reference.S))),
        )
        bounded = MadSDP.adaptive_face(model; tol = 1e-16, bounds = true)
        println(
            details,
            "$name exact rational original dual bound=",
            isnothing(bounded.dual_bound) ? NaN : Float64(bounded.dual_bound),
            " reference minus bound=",
            isnothing(bounded.dual_bound) ? NaN :
            e.reference.objective - BigFloat(bounded.dual_bound),
        )
        println(
            details,
            "$name construction defect: ",
            map(Float64, evaluate(e, e.X, e.y, e.S)),
        )
        println(
            details,
            "$name unweighted constraint condition: ",
            cond(reduce(hcat, vec.(e.A))),
        )
    end
    println(io, "\n```text\n", String(take!(details)), "```")
end

if abspath(PROGRAM_FILE) == @__FILE__
    BLAS.set_num_threads(1)
    # Warm up all paths; reported runtimes exclude compilation.
    run_experiments(devnull)
    open(joinpath(@__DIR__, "face_results.md"), "w") do io
        run_experiments(io)
    end
end
