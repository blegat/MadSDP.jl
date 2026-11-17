# Adaptive PSD faces: a small dense experiment

The experiment supports a limited positive result and an important negative one.
A conservative enclosing face can preserve rotating optimal directions that an
early rank-one restriction excludes. It does **not** by itself cure the accuracy
floor of an interior-point solve. Final equality/dual cleanup, and sometimes
rotation of the support, are needed. Full-space rank-one KKT refinement without
restriction is an effective control on these examples; there is no evidence here
of an accuracy or runtime advantage over that control.

## Reproduce

The implementation is optional: `adaptive_face` leaves the ordinary `solve!`
path unchanged. It supports one Float64 PSD block, no scalar cone, and small
dense problems. It uses NT, QRSchur, and Cholesky backtracking. It does not
change the original model or expose the known optimal support to the heuristic.

This branch includes the existing `dense_condition` compatibility commit
484d413 (cherry-picked as 77374a3). Experiments used LowRankOpt 0.2.6 checkout
9c72c86, MadSDP's original main bae693b, Julia 1.13.3, JuliaFormatter 2.14.10,
and an Intel i7-12700K. The local test manifest originally pinned incompatible
LowRankOpt 0.2.4; it was resolved against the local checkout. Neither dependency
source nor another checkout's branch was changed. The minimum LowRankOpt
compatibility is now 0.2.6 because the inherited buffering API change requires it.

From the MadSDP checkout, with the compatible LowRankOpt checkout alongside it:

```sh
julia --project=test -e 'using Pkg; Pkg.develop(path="../LowRankOpt"); Pkg.instantiate()'
julia --project=test test/faces.jl
julia --project=test experiments/face_restriction.jl
```

The last command warms every comparison path, sets BLAS threads to one, and
writes [face_results.md](face_results.md). It contains all original-data metrics,
failed trial metrics, projector spectra, solver statuses, conditioning estimates,
reference checks, and exact original dual-bound checks. Timings are single warm
measurements including the mode's verification and failed trials, excluding the
common independent reporting/reference solve. These tiny timings are noisy;
they are not a speed benchmark claim. Test commands additionally exercised
`test/qr_schur.jl` and the compatibility commit's `test/MOI_wrapper.jl`.
The 50 focused face assertions, 113 QR assertions, and 1,059 MOI assertions
passed. Temporary `@show @__LINE__` markers covered every instrumented path,
including direct projector diagonalization, equation compression, dual checks,
rounding, and rollback; they were removed afterwards. The Jacobi iteration-limit
error and rethrowing an unexpected programming exception were not forced.

## The models and what is proved

For the controlled example, n=5, A₁=I, A₂ has entries A₂[1,2]=A₂[2,1]=1/2 and
A₂[2,2]=1. Set b=(1,0), X*=e₁e₁ᵀ, S*=diag(0,1,3,4,5), y*=(1/5,−3/10),
and C=A*(y*)+S*. Thus X* is primal feasible, S* is dual feasible, and X*S*=0.
These facts alone would not establish uniqueness or Slater's condition:

- A strictly feasible primal matrix has leading block
  `[0.6 -0.1; -0.1 0.1]` and remaining diagonal entries 0.1. Its trace is one,
  its second equality holds, and its leading determinant is 0.05.
- y=y*−(1,0) gives strictly positive slack S*+I.
- Every optimum has ⟨S*,X⟩=0. PSD forces rows/columns 2–5 to vanish; trace fixes
  X[1,1]=1. Thus the primal optimum is unique. S e₁=0 fixes both dual
  multipliers, so the dual optimum is also unique. Strict complementarity holds.

The example deliberately has a fixed invariant two-dimensional block. This is
an advantage, not generic facial identification. In an interior feasible iterate,
X[1,2]=−X[2,2]≠0: the candidate rank-one support rotates as the boundary is
approached. The enclosing first two dimensions contain its limit exactly in
exact arithmetic. Scaled-identity initialization is infeasible (trace 50 rather
than 1) and is not optimal. The boundary difficulty is visible in worsening
scaled Newton conditioning, dual cross-block errors, and factorization failure
as small primal/slack eigenvalues vanish, even with zero Schur regularization.

The dense example applies the Householder Q=I−2vvᵀ/(vᵀv), v=(1,2,−3,4,−2),
to C and each A. In exact arithmetic, orthogonal congruence preserves both
Slater witnesses, uniqueness, and the optimal value. Q has rational entries
and all its columns are dense. Float64 multiplication and symmetrization
produce a nearby problem: **exact equivalence is claimed only before rounding**.
b stays (1,0); it is not recomputed to make the transformed proposed optimum
feasible. `forward` compares against the rational intended Qe₁e₁ᵀQᵀ;
`forward stored` compares against a 256-bit reference solution of the actual
stored Float64 data. The reference uses eight Newton steps on the rank-one KKT
system, with a known optimum only as a reference seed, not in adaptive solving.
Its original-space KKT residuals and cone errors are recorded. This is a
high-precision numerical reference, not a formal existence/uniqueness proof
for the rounded perturbed SDP. Nonsingularity of the reference KKT system and
the positive nonzero slack spectrum support the local continuation argument.

The follow-up Max-Cut relaxation uses the four-node graph with edge weights
w₁₂=5, w₁₃=7, w₁₄=6, w₂₄=w₃₄=1, and all other weights zero. Its cost is −L/4,
constraints diag(X)=1, and optimum X*=vvᵀ for v=(1,−1,−1,−1). The dual vector
is (C v)./v. Its slack has kernel span(v) and is positive definite on v⊥,
so complementarity plus the diagonal constraints proves uniqueness and value
−18. I is strictly primal feasible; subtracting a positive multiple of the
all-ones vector from y gives strictly dual feasible slack. The generator and
reference checks establish this certificate independently of a solver gap.

## Heuristic and numerical safeguards

At feasible-enough full-space iterates, form the spectral projector onto
slack eigenvalues at most

```text
min(0.2 scaleS, 5 √(μ scaleS / max(1, ‖X‖F))), scaleS=max(1, ‖S‖F).
```

The feasibility gate is max(solver primal/dual residual)≤1e-3. Start trials at
μ≤1e-4 with three samples. The square-root threshold is a scale-aware heuristic,
not a central-path bound; no strict-complementarity guarantee is assumed by
the code. A cap prevents including every large slack eigenvalue. There is no
universal spectral-gap rule here. At the first controlled trial the projector
average has spectrum approximately (0,0,0,4.33e-6,0.999996). Keeping only values
near one freezes a rotating rank-one support. Keeping values above 1e-12
retains the two-dimensional enclosure. The aggressive control uses 0.9.

The default uses the left singular vectors of `[P₁ P₂ P₃]`. Because Pᵢ²=Pᵢ,
its squared singular values divided by three equal the spectrum of the average
projector. This respects arbitrary eigenvector signs and rotations within
clusters. It avoids explicitly forming normal equations for the small
rotational component. `projector_svd=false` implements direct diagonalization
of the average as a numerical control. Finite history never certifies an
invariant space; retaining a near-zero component is inherently threshold-sensitive.

A trial reduces every matrix by VᵀAV and maps X by VᵀXV. An SVD compresses
the reduced equality rows, checking the discarded right-hand-side component
before proceeding. This is numerical rank detection at 1e-12, **not** exact
symbolic equivalence. The lifted result must satisfy every original equality
at the requested target. Projected X is shifted into the relative interior;
if the mapped dual is not interior, the ordinary scaled-identity start is used.
The infeasible-start IPM handles projection-induced equality loss. Neither b
nor the original objective is changed.

Multipliers map back through the equality-row basis, retaining the old null
component. A separate least-squares candidate solves `(C−A*(y))W≈0` in the
**full space**, including cross blocks, using the numerical primal support W.
It is kept only if it improves the maximum original residual. Every slack is
then reconstructed in the original space; PSD of VᵀSV alone is never enough.

The modes distinguish the following operations:

| Mode | Final processing |
| --- | --- |
| NT-Cholesky / NT-QR | Unmodified solver result; identical tolerances and step rule |
| NT-QR+round | Linear equality correction and full-space dual recovery |
| NT-QR+rotate | Five full-space rank-one KKT Newton steps |
| conservative-linear | Restricted IPM, linear equality correction, dual recovery |
| conservative-eigen | Fixed-face rank-one Newton rounding; average-projector eigenvectors |
| conservative | Fixed-face rank-one Newton rounding; stacked-projector SVD |
| conservative+rotate | Additionally permit five full-space rank-one Newton steps |
| aggressive | Same fixed-face processing with discard=0.9 |
| oracle | Rounded intended true support, diagnostic only |

Rank-one Newton rounding solves A(vvᵀ)=b and (C−A*(y))v=0 locally. Within a
fixed V it cannot escape that face; in full space it can rotate. It is a local
nonlinear refinement after a convex SDP solve, **not** another convex restriction
or a global convergence method. Higher numerical rank falls back to linear
rounding. Linear rounding need not preserve PSD; original-space verification
rejects an inaccurate result either way.

All comparisons request tol=1e-16, mu_min=0, zero initial Schur regularization,
and at most 100 IPM iterations per solve. The existing Cholesky backend may
attempt emergency regularization on factorization failure; QR does not use it.
No tolerance was relaxed to accept a face. Trials failing any of the six
original-space tests roll back without changing the saved full-space solver.
Later trials use fresh history after μ decreases by another factor 100; this
allows expansion back to the full cone. There are at most three trials.
`events` retains all candidates, including those rejected near the Float64 floor.
A returned full-space fallback may still fail the target; `verified` says so.
A rotated accepted result sets `restricted=false`, `rotated=true`.

With `bounds=true`, rational LDL elimination checks the **exact binary input**
slack for each saved multiplier. It retains the best valid original dual bound
as an exact Rational{BigInt}, independently of whether the returned approximate
slack is PSD. Zero pivots require zero remaining cross entries. No shifting of
the objective or assertion of dual feasibility from a reduced slack is used.
This deliberately expensive diagnostic is disabled in the timed modes and run
separately. The reference-minus-bound margins are recorded in the results.

## Measurements and interpretation

These residuals measure backward accuracy, not a general forward-error bound.
The known solution and stored-data reference supply separate forward checks.
Accepted tiny PSD violations mean approximate feasibility, not exact cone
membership. Only the optional rationally checked multiplier gives an exact
original dual bound. No data perturbation is used in the solves to obtain these
numbers.

All final metrics and attempted candidate metrics are recomputed from original
stored data in 256-bit BigFloat arithmetic. The small Jacobi eigensolver checks
PSD at the same precision; apparent violations around 1e-77 are its arithmetic
floor. Exact rational dual bounds do not rely on that eigensolver.

For vector Euclidean and matrix Frobenius norms, the reported quantities are:

```text
primal = ‖A(X)−b‖ / max(1, ‖b‖)
dual = ‖C−A*(y)−S‖F / max(1, ‖C‖F)
PSD X = max(0, −λmin(X)) / max(1, ‖X‖F)
PSD S = max(0, −λmin(S)) / max(1, ‖S‖F)
gap = |⟨C,X⟩−bᵀy| / max(1, |⟨C,X⟩|+|bᵀy|)
XS = ‖XS‖F / max(1, ‖X‖F ‖S‖F)
forward = ‖X−X*‖F / max(1, ‖X*‖F).
```

`rank` counts primal eigenvalues above 1e-8 max(1,‖X‖F). `dim` is the last
attempted face dimension, even on rollback; `angle` is the sine of the largest
angle between the known rank-one support and that attempted enclosure,
computed as ‖(I−VVᵀ)u*‖. NaN means no trial. Event notation is
`full-iteration:dimension:reason`. Iteration totals include failed trial IPM
work, but exclude the explicitly described five-step rounding operations.
Recorded cond(B) is for the final prepared NT weighted Jacobian
`vec(GᵀAᵢG)`; cond(B)² estimates its normal-equation condition. Reduced equation
conditions are ratios of retained singular values, excluding discarded zeros.

The full numerical table and trial diagnostics are in
[face_results.md](face_results.md). The following compact table is generated
from those same measurements; `KKT` is the maximum of the six original-space
residuals, not the solver's reported gap.

| Example | Method | Max KKT | Forward (stored data) | IPM iterations | ms |
| --- | --- | ---: | ---: | ---: | ---: |
| block | NT-Cholesky | 1.71e-11 | 1.35e-19 | 29 | 0.24 |
| block | NT-QR | 9.58e-11 | 3.44e-18 | 27 | 0.55 |
| block | NT-QR+round | 3.86e-17 | 3.44e-18 | 27 | 0.47 |
| block | NT-QR+rotate | 3.86e-17 | 0.00e+00 | 27 | 0.44 |
| block | conservative | 3.86e-17 | 9.64e-18 | 60 | 2.56 |
| block | conservative+rotate | 3.86e-17 | 0.00e+00 | 34 | 1.62 |
| block | aggressive | 9.58e-11 | 3.44e-18 | 27 | 0.83 |
| block | oracle | 3.86e-17 | 0.00e+00 | 4 | 0.75 |
| dense | NT-Cholesky | 2.80e-13 | 6.39e-13 | 35 | 0.30 |
| dense | NT-QR | 7.83e-11 | 1.49e-11 | 25 | 0.41 |
| dense | NT-QR+round | 5.92e-12 | 1.49e-11 | 25 | 0.43 |
| dense | NT-QR+rotate | 4.06e-17 | 8.85e-17 | 25 | 0.46 |
| dense | conservative | 7.83e-11 | 1.49e-11 | 83 | 14.19 |
| dense | conservative+rotate | 3.99e-17 | 1.11e-16 | 50 | 20.48 |
| dense | aggressive | 7.83e-11 | 1.49e-11 | 25 | 2.09 |
| dense | oracle | 1.26e-16 | 6.87e-17 | 5 | 3.20 |
| maxcut | NT-Cholesky | 4.90e-07 | 1.32e-11 | 100 | 0.73 |
| maxcut | NT-QR | 1.11e-09 | 8.41e-16 | 26 | 0.43 |
| maxcut | NT-QR+round | 1.72e-16 | 8.40e-16 | 26 | 0.44 |
| maxcut | NT-QR+rotate | 4.32e-78 | 0.00e+00 | 26 | 0.46 |
| maxcut | conservative | 1.11e-09 | 8.41e-16 | 62 | 5.41 |
| maxcut | conservative+rotate | 4.32e-78 | 0.00e+00 | 26 | 2.71 |
| maxcut | aggressive | 1.11e-09 | 8.41e-16 | 26 | 1.26 |
| maxcut | oracle | 1.60e-16 | 0.00e+00 | 6 | 1.31 |

Values below 1e-70 are the Jacobi verification floor, not solver precision;
the refined Max-Cut primal actually equals the integer cut matrix exactly.

| Example | cond(A), unweighted | Last NT-QR cond(B)² | Reference rank-one KKT condition | Reference objective minus saved exact dual bound |
| --- | ---: | ---: | ---: | ---: |
| block | 2.07 | 5.66e17 | 14.60 | 0 |
| dense | 2.07 | 2.66e16 | 14.60 | 5.94e-17 |
| Max-Cut | 1.00 | 2.04e16 | 20.66 | 4.44e-16 |

This separates an ill-conditioned boundary Newton representation from the
well-conditioned local optimality system. At the dense example's first trial,
direct average-projector eigenvectors give an enclosing-support angle of
1.53e-14 and rounded forward error 2.16e-14. Stacked-projector SVD gives
1.04e-15 and about 1.1e-15, respectively; later fixed candidates approach
6e-16 forward error but still fail the uncompromised 1e-16 residual target.
These candidate results are kept even when the driver returns its fallback.

The controlled example's raw primal matrix is already extremely accurate;
its dual cross-block error dominates complementarity. Simple dual recovery
fixes that baseline. Thus successful face rounding is **not evidence of an
accuracy gain over a properly postprocessed full-space method**.

The dense example separates three effects. Full IPM termination can have a tiny
gap while forward/complementarity errors remain orders of magnitude larger.
Direct average-projector diagonalization loses the small rotational eigenspace;
stacked-projector SVD substantially reduces that enclosure error. A fixed
approximate face still imposes a forward-error floor proportional to its
principal-angle error. Full-space Newton refinement removes that geometric
obstruction and reaches approximately machine accuracy. Some excellent fixed
candidates are correctly rejected at the strict 1e-16 target: rounding in dense
congruences and lifted Float64 matrices is already of that order. Failure of
that target does not mean a 1e-15 result is inaccurate, and a computed zero gap
does not imply an exact optimum of the stored problem.

The Max-Cut control shows that more stable QR linear algebra matters. Its early
conservative faces do not reliably enclose the optimum accurately enough;
fixed-face trials fail full-space checks. Allowing rotation, or applying the
same local KKT refinement to full-space QR, recovers the certified cut solution.
This graph is small and intentionally has a certifiable rank-one SDP optimum;
it does not establish robustness on general Max-Cut instances.

The experiment does not justify promising 1e-16 on arbitrary Float64 SDPs.
Its useful result is a safeguarded conservative restart and a reproducible
separation of linear algebra, enclosure error, final rounding, and stored-data
error. More aggressive facial identification and a tiny duality gap are both
insufficient. A larger face helps preserve room for rotation, but numerical
subspace extraction and a full-space check remain essential.

## Relation to prior work

- Ding, Yurtsever, Cevher, Tropp & Udell (2021),
  [DOI 10.1137/19M1244603](https://doi.org/10.1137/19M1244603),
  [paper](https://tropp.caltech.edu/papers/DYCTU21-Optimal-Storage-Approach.pdf):
  §4.1 compresses onto small-slack eigenspaces; Theorems 4.1 and 4.6 distinguish
  rank-sensitive feasibility recovery and more robust objective recovery.
  §§5.2–5.4 discuss rank, infeasibility, and quality checks; §6 combines dual
  iterations with recovery. Their MinFeasSDP minimizes equality error and
  MinObjSDP permits nonzero equality error. In particular §5.3 warns that zero
  allowed infeasibility is generally infeasible. Those recovery bounds do not
  justify our exact equality restriction. Our rollback is empirical protection,
  not a new version of their theorem.
- Mohammad-Nezhad & Terlaky,
  [On the Identification of the Optimal Partition for Semidefinite Optimization](https://optimization-online.org/wp-content/uploads/2016/09/5626.pdf),
  §§3.3–3.4, Theorems 3.8–3.9 and 3.11–3.13, and §4, Theorem 4.3:
  eigenvalue separation and subspace proximity depend on central-path or
  neighborhood conditions and error bounds. The transition subspace matters
  without strict complementarity. Our finite history at arbitrary IPM iterates
  has no such certificate, even though these examples are strictly complementary.
- Mohammad-Nezhad & Terlaky,
  [A Rounding Procedure for Semidefinite Optimization](https://engineering.lehigh.edu/sites/engineering.lehigh.edu/files/_DEPARTMENTS/ise/pdf/tech-papers/17/17T_009_0.pdf),
  §3, equations (11)–(15) and the primal/dual least-squares constructions:
  dropping unknown cross-subspace terms need not leave solvable systems;
  complementary boundary points generally retain feasibility error. Our
  explicit rounding and original-space rejection follow that distinction;
  we do not claim their guarantees for our Newton cleanup.
- Bellavia, Gondzio & Porcelli (2021),
  [A Relaxed Interior Point Method for Low-Rank Semidefinite Programming Problems with Applications to Matrix Completion](https://pmc.ncbi.nlm.nih.gov/articles/PMC8504795/),
  §§2–3: nearly low-rank primal representations and relaxed Newton equations
  allow adaptive rank revelation. This is related motivation, not the same
  algorithm as a fixed convex face followed by local support refinement.
- Todd, Toh & Tütüncü (1998),
  [DOI 10.1137/S105262349630060X](https://doi.org/10.1137/S105262349630060X),
  [manuscript](https://ecommons.cornell.edu/server/api/core/bitstreams/57c1e75e-dcaf-4d06-930a-eabec36721eb/content),
  manuscript pp. 14–15, equations (38)–(40): the Schur system is the normal
  equation for a weighted least-squares problem, permitting QR on the weighted
  Jacobian instead. The discussion immediately following contrasts finite
  precision conditioning with Cholesky. This motivated keeping QR as an
  essential baseline instead of attributing all boundary difficulties to rank.
