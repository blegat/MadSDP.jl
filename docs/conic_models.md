# Conic model interface

`LowRankOpt.Model` uses ordinary free scalar decision variables. Its
NLPModels constraint outputs contain bounded scalar rows first, followed
by SOC vectors and full column-major PSD matrices. The scalar rows retain their equality,
lower, upper, or interval bounds in `meta.lcon` and `meta.ucon`.

MadSDP keeps equality multipliers outside the barrier calculation. The
default NT method uses `RootKKT()`: a sparse augmented system containing
scaled cone directions and equality rows. It avoids normal equations and
recovers the cone directions directly. `AugmentedKKT()` assembles the
weighted Schur block instead and also supports HKM directions. Both reuse
MadNLP's sparse UMFPACK solver and bound initialization routine. KKT
regularization is applied when a root system is singular; iterative
refinement checks equations in their original units.

Scalar presolve omits inequality slacks that equalities already fix to
feasible constants. This includes bounds on the final free entry of a
simplex when its other entries are fixed. These constraints remain in the
public model, and their multipliers are returned as zero. Equalities
remain native rows of the augmented system.

`DenseCholesky()`, `MadNLPSchur()`, and `QRSchur()` remain available for the
explicit `LowRankOpt.DualModel` formulation. Selecting one through the
primal interface requests that explicit dual conversion. Loraine and
BurerMonteiro also request the conversion explicitly; finite scalar bounds
are expanded there, and multipliers are mapped back to the public rows.

## Native SOC blocks

`RootKKT()` handles Lorentz cones directly, using their Jordan product,
NT scaling, predictor-corrector equations, and analytic boundary steps.
The SOC barrier degree is one with the Euclidean pairing. Rotated cones
use an orthogonal transformation in the model interface. The public
constraint order is scalar rows, SOC vectors, then PSD matrices.

Components using HKM directions explicitly request the PSD-arrow adapter
for SOCs. Loraine and BurerMonteiro use that adapter as well. The PSD
representation is a computational choice of those components.

Unsuccessful iterates are checked for valid Farkas or recession rays.
A Farkas check during iteration prevents an infeasible SOC path from
overflowing. Reported certificates are normalized and independently
verified against the original conic model.
