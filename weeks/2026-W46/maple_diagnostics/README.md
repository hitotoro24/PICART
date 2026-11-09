# Week 46 — Maple diagnostic study: CART effective-centre Hessian vs. residual-based PHD, OPFData IEEE-118

Independent Maple re-implementation and diagnostic evaluation of the 2D and 3D comparison. The CART reconstruction
is **Algorithm 1 of the manuscript, implemented line by line** in `hessian_tools.algorithm1_hessian`; its per-node
quantities (eq. 3 gradient, eq. 4 effective location) come from the Week 45 package, which is imported unchanged.
PHD is implemented here from scratch. Nothing in Week 45, cedar_baseline, the CSV, or any manuscript file is
modified.

### Algorithm 1 as implemented

| pseudocode line | code |
|---|---|
| 1–4: for each internal node read s, τ, ℓ, u, μ_L, μ_R; compute γ by (3), z by (4) | `node_gradient`, `effective_center` (Week 45) |
| 5: empty V_s, y_s | dict of lists per axis |
| 6–11: for each pair of distinct nodes (A, B) with s_A = s_B and z_A ≠ z_B, append (z_B − z_A)ᵀ to V_s and γ_B − γ_A to y_s | double loop over ordered pairs, equality test on z |
| 12–18: if rank(V_s) < d → "not identified"; else least squares → row s of Q̃ | `np.linalg.matrix_rank`, `np.linalg.lstsq`; returns `identified=False` and `Q_hat=None` |
| 19: return Q̂ = (Q̃ + Q̃ᵀ)/2 | `Q_hat` |

Q̃ (the matrix before line 19) is also returned and is what the symmetry tables inspect, because Q̂ by construction
cannot show the inconsistency. Numerically, Algorithm 1 reproduces the Week 45 solver (unordered pairs) and the
Cedar centred regression to ~1e-9 in both 2D and 3D, so all three are the same least-squares problem.

## Files

| file | purpose |
|---|---|
| `maple_opfdata_hessian_2d.ipynb` | 2D study (CSV columns 1–2), executed with outputs |
| `maple_opfdata_hessian_3d.ipynb` | 3D study (CSV columns 1–3), executed with outputs |
| `hessian_tools.py` | all functions used by the notebooks, including `algorithm1_hessian`; every function has a docstring (purpose, inputs, outputs) |
| `make_notebooks.py` | regenerates and executes both notebooks (`python make_notebooks.py` from this directory) |

Run a notebook from this directory; it needs `numpy pandas scikit-learn matplotlib`. The CSV is read from
`/Users/tonys/Desktop/Class/Decision Trees/datasets/118_ieee_smol.csv` (fallback: `118_ieee_smol.csv` in this
directory). Only the first 3 or 4 columns are read; the 2 GB file is not used.

## Column convention

Headerless CSV, 5,000 × 100. Column 0 = response (operating cost of the solved OPF). Columns 1..d = predictors,
labelled *Load k* for CSV column k. Columns 1–99 are consistent with per-unit active demand at the 99 load
elements of the IEEE-118 case (column means sum to 42.40 p.u. = 4,240 MW; nominal case total 4,242 MW). Bus
identities per column are not established.

## What this study adds beyond the Cedar baseline

| | Cedar baseline | Maple diagnostics |
|---|---|---|
| CART reconstruction | re-implemented inline (`cart_hessian`, centred regression per axis) | Algorithm 1 implemented line by line (`algorithm1_hessian`); checked against the Week 45 solver and the Cedar centred regression, all agree to ~1e-9 |
| tree settings | `max_depth=7`, default `min_samples_leaf=1` only | setting **A** (same as cedar_baseline, for comparability) and setting **B** `min_samples_leaf=50`, side by side |
| first-derivative step | computed inside `cart_hessian` | tabulated per node: extent, τ, child means, γ, effective centre |
| PHD | Σ⁻¹MΣ⁻¹ then symmetrised | same estimator, conventions stated (1/n covariance, OLS residual on intercept + linear terms), returned unsymmetrised with its rounding asymmetry reported (~1e-16) |
| symmetry report | pairs, mean/median/max abs gap | same, plus gap relative to the larger entry; the 2D case is stated to have one pair and the 3D case three; all math in the notebooks is LaTeX and figures follow the Cedar `plot_hessian` style |
| PHD comparison | max abs difference | full entry-by-entry table for raw CART, and separately for ½(H+Hᵀ) labelled as a derived step |
| diagnosis | none | §10 controlled check on the fitted partitions with a known quadratic: (a) exact box means, (b) empirical node mean responses at the real rows, (c) uniform resample with matched counts; plus within-cell centroid offsets |
| notebook outputs | committed without executed outputs | executed end to end; software versions, shapes, ranks, raw values and tables stored |

## Results on the 5,000-row subset

All numbers are entries of Q̃ in original units (cost per predictor-unit²). Setting A reproduces cedar_baseline exactly.

**2D** (one distinct off-diagonal pair)

| setting | H[1,2] | H[2,1] | abs gap | gap / max entry |
|---|---|---|---|---|
| A | −733,998.62 | −3,154,084.83 | 2,420,086.20 | 0.77 |
| B | −192,840.71 | −124,378.14 | 68,462.57 | 0.36 |

PHD: [[140,086.44, −474,727.79], [−474,727.79, 830,031.37]]. CART H[2,2] is negative in both settings (−3.9M, −0.53M)
where PHD is positive.

**3D** (three distinct pairs; abs gaps for (1,2), (1,3), (2,3))

| setting | abs gaps | mean / median / max |
|---|---|---|
| A | 2,175,983.07 / 381,057.08 / 2,242,116.58 | 1,599,718.91 / 2,175,983.07 / 2,242,116.58 |
| B | 1,126,179.39 / 280,901.65 / 765,021.07 | 724,034.04 / 765,021.07 / 1,126,179.39 |

Rank is full (d) in every row for both settings and both dimensions.

## Findings

1. **Controlled algebra validation (not a general guarantee of bug-free code).** Pairwise and centred solves agree to rounding; with exact box means on the fitted
   partitions the known quadratic is recovered to ~1e-14 in 2D and 3D. The implementation follows the derivation.
2. **The raw CART matrix is not symmetric on this data**, with gaps of 0.3–1.9 times the larger of the two entries.
   Since H[i,j] and H[j,i] are inferred from different node sets, the gap measures internal asymmetry, **not error relative to an unknown true OPF Hessian**.
3. **Evidence about the cause (not a definitive decomposition).** empirical node mean responses of a noise-free known quadratic at the real rows already give relative errors of 1–10×
   (2D) and 2–10× (3D). a uniform resampling experiment with matched counts gives errors of the same order, which supports the role of finite observations within small cells at n = 5,000, without proving that any one cause dominates; at `min_samples_leaf=50` the real rows are worse than the
   uniform resample in every seed, so the non-uniform placement of the predictors inside boxes (mean centroid
   offset 7–11 % of box width) is an additional, smaller term.
4. **PHD comparison.** PHD is symmetric with entries of order 10⁵–10⁶ and disagrees with CART in magnitude and in
   some signs. Given (2)–(3) this disagreement is within the CART estimate's own inconsistency and does not yet
   discriminate between the two methods' assumptions.

## Naming and provenance

This folder is the **Maple diagnostic study**, not a numbered successor implying improved accuracy. Setting A reproduces the [Cedar baseline](../cedar_baseline/README.md); setting B changes the minimum leaf size. The controlled known-quadratic experiments are additional diagnostics, not OPF ground truth. Original directory naming is preserved in Git history.

**Mathematical caution.** The effective location and regional-mean derivative identity follow from a quadratic with *uniform volume averages over its child rectangles*. Actual CART child-node mean responses average only the observations assigned to the children. If those observations are concentrated unevenly, the contrast can already fail to equal the derivative at the geometric effective location, before second derivatives are reconstructed. Nonuniformity alone does not identify which estimator correction is appropriate.

## Limitations

5,000 rows at depth 7 is far too few for second differences of cell means; the full dataset was deliberately
excluded here. The effective-centre location assumes a uniform within-box distribution that the correlated
predictors do not satisfy. All node pairs are weighted equally and pooled over the whole domain (global quadratic
assumption). PHD assumes a Gaussian design. No eigenanalysis, bootstrap, prediction metric, or alternative
estimator is included.
