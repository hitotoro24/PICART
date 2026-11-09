# Week 46 — attempt_02: CART effective-centre Hessian vs. residual-based PHD, OPFData IEEE-118

Independent re-implementation and evaluation of the 2D and 3D comparison. The CART reconstruction is **the
existing Week 45 method**, imported unchanged from `weeks/2026-W45/tree_hessian`; PHD is implemented here from
scratch. Nothing in Week 45, attempt_01, the CSV, or any manuscript file is modified.

## Files

| file | purpose |
|---|---|
| `ieee118_cart_phd_2d.ipynb` | 2D study (CSV columns 1–2), executed with outputs |
| `ieee118_cart_phd_3d.ipynb` | 3D study (CSV columns 1–3), executed with outputs |
| `hessian_tools.py` | all functions used by the notebooks; every function has a docstring (purpose, inputs, outputs) |
| `make_notebooks.py` | regenerates and executes both notebooks (`python make_notebooks.py` from this directory) |

Run a notebook from this directory; it needs `numpy pandas scikit-learn matplotlib`. The CSV is read from
`/Users/tonys/Desktop/Class/Decision Trees/datasets/118_ieee_smol.csv` (fallback: `118_ieee_smol.csv` in this
directory). Only the first 3 or 4 columns are read; the 2 GB file is not used.

## Column convention

Headerless CSV, 5,000 × 100. Column 0 = response (operating cost of the solved OPF). Columns 1..d = predictors,
labelled *Load k* for CSV column k. Columns 1–99 are consistent with per-unit active demand at the 99 load
elements of the IEEE-118 case (column means sum to 42.40 p.u. = 4,240 MW; nominal case total 4,242 MW). Bus
identities per column are not established.

## What differs from attempt_01

| | attempt_01 | attempt_02 |
|---|---|---|
| CART reconstruction | re-implemented inline (`cart_hessian`, centred regression per axis) | Week 45 `tree_hessian.uneven` imported and called directly (pairwise system, row-wise lstsq); the two formulations are checked numerically and agree to ~1e-9 |
| tree settings | `max_depth=7`, default `min_samples_leaf=1` only | setting **A** (same as attempt_01, for comparability) and setting **B** `min_samples_leaf=50`, side by side |
| first-derivative step | computed inside `cart_hessian` | tabulated per node: extent, τ, child means, γ, effective centre |
| PHD | Σ⁻¹MΣ⁻¹ then symmetrised | same estimator, conventions stated (1/n covariance, OLS residual on intercept + linear terms), returned unsymmetrised with its rounding asymmetry reported (~1e-16) |
| symmetry report | pairs, mean/median/max abs gap | same, plus gap relative to the larger entry; the 2D case is stated to have one pair and the 3D case three |
| PHD comparison | max abs difference | full entry-by-entry table for raw CART, and separately for ½(H+Hᵀ) labelled as a derived step |
| diagnosis | none | §8 controlled check on the fitted partitions with a known quadratic: (a) exact box means, (b) sample means at the real rows, (c) uniform resample with matched counts; plus within-cell centroid offsets |
| notebook outputs | committed without executed outputs | executed end to end; software versions, shapes, ranks, raw values and tables stored |

## Results on the 5,000-row subset

All numbers in original units (cost per predictor-unit²). Setting A reproduces attempt_01 exactly.

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

1. **No coding defect.** Pairwise and centred solves agree to rounding; with exact box means on the fitted
   partitions the known quadratic is recovered to ~1e-14 in 2D and 3D. The implementation follows the derivation.
2. **The raw CART matrix is not symmetric on this data**, with gaps of 0.3–1.9 times the larger of the two entries.
   Since H[i,j] and H[j,i] come from disjoint node sets, the gap is an assumption-free measure of estimation error.
3. **Cause.** Sample means of a noise-free known quadratic at the real rows already give relative errors of 1–10×
   (2D) and 2–10× (3D). A uniform resample with matched counts gives errors of the same order, so the dominant term
   at n = 5,000 is sample-mean variability in small cells; at `min_samples_leaf=50` the real rows are worse than the
   uniform resample in every seed, so the non-uniform placement of the predictors inside boxes (mean centroid
   offset 7–11 % of box width) is an additional, smaller term.
4. **PHD comparison.** PHD is symmetric with entries of order 10⁵–10⁶ and disagrees with CART in magnitude and in
   some signs. Given (2)–(3) this disagreement is within the CART estimate's own inconsistency and does not yet
   discriminate between the two methods' assumptions.

## Limitations

5,000 rows at depth 7 is far too few for second differences of cell means; the full dataset was deliberately
excluded here. The effective-centre location assumes a uniform within-box distribution that the correlated
predictors do not satisfy. All node pairs are weighted equally and pooled over the whole domain (global quadratic
assumption). PHD assumes a Gaussian design. No eigenanalysis, bootstrap, prediction metric, or alternative
estimator is included.
