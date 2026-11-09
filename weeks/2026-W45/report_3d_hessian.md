# IEEE PES | 3D quadratic Hessian from regional means: controlled test and CART inspection

Date: 2026-11-05. Role in this session: implementer/tester of the construction Tony
specified (aligned, equally spaced regions; regional means; second-order finite
differences). No estimator beyond that construction was used to produce a number
reported as an estimate.

Target: `f(x) = c + b^T x + 1/2 x^T Q x` on `[0,1]^3` with

```
Q = [[4.0, 1.5, 0.5],      b = [-1, 2, 0.5],   c = 0.7
     [1.5, 3.0, -1.0],
     [0.5, -1.0, 2.0]]
```

## 1. What already existed and is reusable

| Item | Status |
|---|---|
| `docs/quadratic_hessian.ipynb` | 2D only; *point-evaluation* centred differences, not regional means. Reusable idea: the mixed entry from a directional stencil. Not reused directly. |
| `hessian_derivation.tex (RAship export, external derivation notes)`, `hessian_2d.tex` | Maple-authored derivations (box-mean lemma, effective-centre proposition, uneven-split correction). Per the working agreement they are to be reproduced by Tony before use; this session used only the *even-split* special case they share with Tony's specification. |
| `tree_hessian/` named in `working_agreement.md` | Not in the repository export. Rebuilt here from scratch. |
| Wycoff (2024) eq. 3 | `gamma_i = 2 (mu_r - mu_l) / (u - l)`; used verbatim as `stencils.gradient_estimate`. |
| Python environment | No scikit-learn on the machine; created `.venv` (gitignored) with numpy 2.0.2, scikit-learn 1.6.1, pytest. |

## 2. What was implemented

```
tree_hessian/quadratic.py      Quadratic f; exact box mean (2-pt Gauss-Legendre, exact for quadratics);
                               uniform sampling; sample box mean.
tree_hessian/stencils.py       gradient_estimate (eq. 3); diag_three_cell; diag_four_cell; mixed_four_cell;
                               cell constructors with midpoint defaults and optional off-midpoint thresholds.
tree_hessian/tree_extract.py   Walk a fitted sklearn tree -> boxes; split table; midpoint deviation;
                               find_diag_configs / find_mixed_configs (accept only within tol of midpoints);
                               assemble (NaN where nothing accepted).
tree_hessian/tests/            20 tests: analytic values, predicted failures, tree extraction.
experiments/run_controlled.py  Parts A/B/C below.   -> experiments/results/controlled.txt
experiments/run_cart.py        CART inspection.      -> experiments/results/cart.txt
```

Formulas (all are differences of two eq.-3 gradient estimates divided by the
distance between the regions they belong to):

* Diagonal, four equal cells of width `w/4` along `x_i` from a midpoint split and
  two midpoint child splits: `Q_ii = (gamma_R - gamma_L)/(w/2) = 8(mu_RR - mu_RL - mu_LR + mu_LL)/w^2`.
* Diagonal, three adjacent equal cells of width `h`: `Q_ii = (mu_+ - 2 mu_0 + mu_-)/h^2`.
* Mixed, 2x2 equal cells (midpoint split along `x_j`, both halves split along `x_i`
  at the common midpoint): `Q_ij = (gamma_i(T) - gamma_i(B))/(w_j/2) = 4(mu_TR - mu_TL - mu_BR + mu_BL)/(w_i w_j)`.

Assumptions as implemented: (A1) cells are axis-aligned boxes sharing every extent
except along the differenced axes; (A2) equal widths along the differenced axes.

Run:

```
python -m pytest -q tree_hessian/tests
python experiments/run_controlled.py
python experiments/run_cart.py
```

## 3. Results

### 3A. Exact regional averaging: deterministic approximation error

All six entries recovered on `[0,1]^3` with Frobenius error 1.4e-14; over 200
random sub-boxes and cell widths the worst entry error (three- and four-cell
stencils) was 9.5e-12. For a quadratic the approximation error of the specified
construction is zero; only rounding remains.

### 3B. Exact averaging with (A2) violated: the bias is deterministic and linear

| violation | error / delta | note |
|---|---|---|
| diag `x_0`, left child split at `0.25 + delta` | -2.667 = -(2/3) Q_00 | error exactly linear in delta |
| mixed `(0,1)`, `tau_B = 0.5 - delta`, `tau_T = 0.5 + delta` | +5.333 = (4/3) Q_00 | error linear in the misalignment, scaled by the *diagonal* entry |

So a 5 % threshold misplacement gives errors of 0.13 (diag) and 0.27 (mixed) on
entries of size 4 and 1.5. This is the quantity at stake when CART thresholds are
used as if they were midpoints.

Two observations the tests record but that have **not been derived** (B3, B4 in
`controlled.txt`; tests marked OBSERVATION): the mixed stencil stays exact when the
outer `x_j` split is anywhere, as long as the two inner `x_i` thresholds coincide;
and the diagonal four-cell stencil stays exact for an off-centre outer split when
each child is split at its own midpoint and the divisor is the actual distance
between the child centroids. Both are stronger than (A2).

### 3C. Uniform sampling: sampling error (RMSE over 10 seeds, noiseless `y`)

Stencil box `[0,w]^3`; cells have width `w/4` (diag) or `w/2` (mixed) along the
differenced axes.

| w | n | RMSE Q_00 | RMSE Q_01 | Frobenius RMSE |
|---|---|---|---|---|
| 1.00 | 1e3 | 1.13 | 0.42 | 2.17 |
| 1.00 | 1e4 | 0.26 | 0.13 | 0.67 |
| 1.00 | 1e5 | 0.064 | 0.030 | 0.17 |
| 1.00 | 1e6 | 0.033 | 0.016 | 0.069 |
| 0.50 | 1e6 | 0.125 | 0.080 | 0.31 |
| 0.25 | 1e6 | 0.57 | 0.27 | 1.02 |

Full table (all entries, both noise levels) in `experiments/results/controlled.txt`.
Points to note:

* Even with noiseless `y`, sample means carry Monte Carlo error because `f` varies
  inside a cell; adding observation noise of sd 0.1 changes the RMSE by a few
  percent at most. In this regime the within-cell variation of `f`, not the noise,
  dominates.
* Error falls roughly like `n^{-1/2}` at fixed `w` and grows fast as `w` shrinks
  (fewer points per cell and division by `w^2`). At `w = 0.25, n = 1e3` some cells
  were empty. No rate is claimed beyond what the table shows.

### 3D. CART trees on the same quadratic

`DecisionTreeRegressor(max_depth=K)` on `N` uniform points, noiseless unless stated.
Domain `[0,1]^3` used as the root extent. Full output in `experiments/results/cart.txt`.

Split variables: the root splits `x_1`, depth 1 both nodes split `x_0`, depth 2 all
four split `x_1`; deeper levels mix all three. Thresholds: median relative midpoint
deviation `|tau - mid|/width` is 5 to 9 % of the node width; the fraction of
internal nodes within 1 % of the midpoint is 0 to 10 %.

Subtree shapes present, ignoring where thresholds fall:

| tree | diag shape (node `i`, both children `i`) | node `i` with at least one child `i` | mixed shapes (0,1)/(0,2)/(1,2) |
|---|---|---|---|
| K=3, N=2e4 | 0 / 0 / 0 | 0 / 0 / 0 | 3 / 0 / 0 |
| K=5, N=2e4 | 0 / 0 / 0 | 0 / 0 / 1 | 5 / 3 / 2 |
| K=7, N=2e4 | 0 / 0 / 0 | 1 / 2 / 1 | 19 / 12 / 14 |
| K=5, N=2e4, noise 0.1 | 0 / 0 / 0 | 0 / 0 / 1 | 5 / 1 / 2 |
| K=5, N=2e5 | 0 / 0 / 0 | 0 / 0 / 1 | 6 / 1 / 2 |
| K=10, N=2e5 | 0 / 1 / 0 | 7 / 4 / 6 | 143 / 142 / 149 |

Configurations accepted under the specified construction (`tol = 0`, thresholds at
midpoints to rounding): **none, in every tree.** No entry of `Q` was estimated from
a CART tree under the stated assumptions.

Diagnostic only (equal-width formulas applied to configurations whose thresholds are
within `tol` of the midpoints; these are *not* valid estimates under the
construction, and 3B says what the bias is): at `tol = 0.10` the K=5/N=2e5 tree
yields `Q_01 = 1.20` (true 1.5) from 3 subtrees and `Q_12 = -0.96` (true -1.0) from
1; at `tol = 0.25` all three mixed entries get values with errors 0.04 to 0.17 on
that tree and 0.07 to 0.18 on the K=7 tree, while individual subtree estimates range
from -1.7 to 4.1 for `Q_01`. The single diagonal-shaped subtree in the whole study
(K=10, node 852) gives `Q_11 = -10.7` against 3.0. The Frobenius error is undefined
in every tree because the diagonal is never available.

## 4. Research summary

**1. Entries recovered.** Under the controlled construction (aligned, equal cells,
exact or sampled regional means): all six independent entries, exactly with exact
means and with the sampling RMSE of 3C. From CART trees under the specified
assumptions: no entry.

**2. Assumptions under which the results hold.** `f` exactly quadratic; cells are
axis-aligned boxes sharing all extents except along the differenced axis or axes
(A1); equal widths along those axes (A2); regional means are either exact or sample
means of uniform draws over the cell (so that the sample mean estimates the uniform
regional mean, which is what makes the eq.-3 quantity a gradient estimate). The
root extent must be known: the tree extraction uses the sampling domain `[0,1]^3`,
which a real dataset would not supply.

**3. Entries or subtree configurations that could not be recovered.**

* All diagonal entries from CART: the shape "node splits `x_i`, both children split
  `x_i`" did not occur in trees of depth 3 to 7, and occurred once at depth 10 with
  thresholds far from midpoints. CART on this quadratic alternates split variables
  (the gain of splitting a variable again is reduced once its extent is halved);
  consecutive same-variable splits appear only deep in the tree where cells are
  small and sample means noisy.
* All mixed entries from CART under (A1)+(A2): the mixed *shape* is common
  (alternating `x_1`/`x_0` splits), but the two children's inner thresholds never
  coincide and are never at the midpoint, so the equal-cell cross stencil does not
  apply. The diagnostic numbers show how far it is off when applied anyway.
* Increasing depth increased the count of mixed shapes but did not move thresholds
  toward midpoints (median deviation 5 to 9 % at every depth) and produced one badly
  wrong diagonal estimate. Nothing here supports a depth-based convergence claim.

**4. Mathematically or methodologically unresolved; decisions for Tony.**

1. *Non-midpoint thresholds.* The specified construction needs midpoints; CART's
   thresholds are midpoints between adjacent sample values and essentially never
   coincide with box midpoints. Options I did not choose between: (a) keep the
   construction and constrain the tree (e.g. a midpoint-split tree, or data on a
   grid so that candidate thresholds are cell boundaries); (b) extend the
   construction to arbitrary thresholds. `hessian_derivation.tex (RAship export, external derivation notes)`
   contains such an extension (effective centre `(l + tau + u)/3`; it predicts the
   3B mixed bias `2.667 (tau_T - tau_B)` exactly), but the working agreement lists
   it as not yet reproduced by you, so it was not used.
2. *Diagonal entries from a tree.* Even with arbitrary thresholds handled, a
   diagonal entry needs two consecutive same-variable splits, which CART rarely
   makes on this target. Whether to accept this as a limitation, to use a
   three-cell configuration from "node `i` with one child `i`" (needs its own
   formula; widths are unequal), or to combine gradient estimates from
   non-nested nodes is a modelling choice.
3. *Aggregation across subtrees.* `assemble` takes the unweighted mean of accepted
   estimates. With sample means, subtrees with fewer points are noisier; a weighting
   (and the target quantity, pointwise `Q` versus `E[nabla^2 f(X)]`) is not decided.
4. *Observations 3B-B3/B4* (stencils exact under weaker conditions than (A2)) are
   numerical facts awaiting a derivation.
5. *Root extent* for real data.

## 6. Extension to uneven splits (added later on 2026-11-05, at Tony's choice)

Formulation used, taken from `hessian_derivation.tex (RAship export, external derivation notes)` (Proposition
"Effective center" and the "General dimension" section); eq. 3 is unchanged:

* a node with extent `[l, u]` splitting `x_s` at `tau` carries
  `gamma = 2(mu_r - mu_l)/(u_s - l_s) = d f/d x_s (c)` with
  `c_s = (l_s + tau + u_s)/3` and `c_p = (l_p + u_p)/2` for `p != s`;
* for two nodes `a, b` splitting the same `x_s`: `gamma_b - gamma_a = (c_b - c_a)^T Q e_s`;
* all pairs of nodes splitting `x_s` are stacked and row `s` of `Q` is the least-squares
  solution, reported only if the displacement matrix has rank 3; the matrix is then
  symmetrised and the raw asymmetry reported as a consistency check.

Code: `tree_hessian/uneven.py`, `tree_hessian/synthetic.py` (hand-built trees),
`experiments/run_uneven.py` -> `experiments/results/uneven.txt`; 6 further tests
(effective centre identity; exactness on random uneven trees; the `1/2` variant is
biased; rank-deficient rows return NaN; CART geometry with exact means is exact;
error shrinks with N). Also `extract_nodes_from_cart` reads Tony's own
`cart_regressor.py` tree; on N = 2000 its splits are identical to sklearn's.

**Results.**

| setting | result |
|---|---|
| CART geometry, exact box means, depth 2 or 3 | rows unidentified (rank < 3): too few nodes split each variable, or their centres are coplanar. Reported as NaN. |
| CART geometry, exact box means, depth 5 and 7 | all six entries exact, Frobenius error 1e-15, asymmetry 1e-15 |
| CART sample means, N = 2e4, depth 5 / 7 / 10 | Frobenius error 0.23 / 0.17 / 0.35; max entry error 0.14 / 0.08 / 0.17 |
| CART sample means, depth 6, N = 2e3 / 2e4 / 2e5 (10 seeds) | mean Frobenius 1.02 / 0.33 / 0.11 |
| CART sample means, depth 8, N = 2e3 / 2e4 / 2e5 | 2.74 / 0.42 / 0.09 |
| depth 4, any N | unidentified in every seed |
| observation noise sd 0.1 | changes the above by at most ~0.1 |
| Tony's CART, N = 2e3, depth 6 | exact means: exact; sample means: Frobenius 1.34 |

**What this establishes.** With the effective-centre location, the obstacle of
section 4 (CART thresholds are not midpoints) disappears for a quadratic: the tree
geometry CART actually produces identifies all six entries exactly from exact
regional means, including the diagonal entries that the aligned construction could
never obtain. The residual error on real trees is entirely sampling error in the
leaf means, and it falls with N at fixed depth. It does **not** fall monotonically
with depth at fixed N (depth 10 is worse than depth 7 at N = 2e4; depth 8 is worse
than depth 6 at N = 2e3), because deeper cells hold fewer points and the pair
displacements shrink.

**Identifiability condition (observed, to be stated and proved by Tony).** Row `s`
is identified iff the effective centres of the nodes splitting `x_s` affinely span
`R^3`, which needs at least four such nodes with centres not all in one plane.
Shallow trees fail this; the alternating split pattern of section 3D means the
nodes splitting `x_0` at depth 1 are only two, with centres differing in `x_1`
only.

**Still open after this step.**

1. The derivation this rests on (box-mean lemma, effective-centre proposition) is
   Maple-authored and, per the working agreement, still to be reproduced by Tony.
   The tests `test_gamma_equals_partial_at_effective_center` and
   `test_half_instead_of_third_is_biased` are what a reproduction must agree with.
2. Weighting of pairs: all pairs weighted equally. Cells with few points are
   noisier and dominate at depth 10; a weighting (by cell counts, or by the
   variance of `gamma`) is a modelling choice not made here.
3. Locality: all nodes splitting `x_s` anywhere in the domain are pooled, which is
   right for a quadratic (constant `Q`) and wrong in general. For a non-quadratic
   target the pooling radius, or a per-subtree solve, must be chosen.
4. The root extent `[0,1]^3` is still taken from the sampling design.

## 5. Reviewer notes on existing material

* `hessian_derivation.tex (RAship export, external derivation notes)`, line 113: the aligned-case cross stencil is
  written with a factor 8; `hessian_2d.tex` (Remark "Even splits") and the code
  here give 4. The 4 is correct for `v_2 = (u_2 - l_2)/2` and
  `gamma = 2 Delta mu/(u_1 - l_1)`. Not edited.
* `working_agreement.md` "Current state" refers to `tree_hessian/`, which was not in
  the export; the directory now holds this session's code, which is different from
  the 2D toy implementation described there.
